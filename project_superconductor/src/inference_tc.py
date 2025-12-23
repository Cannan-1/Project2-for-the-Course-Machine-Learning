"""
超导材料临界温度(Tc)预测模型推理模块
实现训练好的模型加载、化学式解析和实时预测功能
"""

import torch
import pandas as pd
import numpy as np
import os
import sys

# ==================== 系统路径配置 ====================
# 确保模块导入路径正确，支持不同运行环境
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 动态导入自定义模块（兼容不同项目结构）
try:
    from data_processor import SuperconDataProcessor
    from model import TcPredictorAdvanced
except ImportError:
    from src.data_processor import SuperconDataProcessor
    from src.model import TcPredictorAdvanced


class Inference:
    """
    超导Tc预测推理引擎
    负责加载训练好的模型并进行实时预测
    
    核心功能:
    1. 初始化数据处理器（需使用训练数据校准标准化器）
    2. 加载并验证预训练模型权重
    3. 解析化学式字符串并生成Tc预测
    """
    
    def __init__(self, model_path, train_data_path):
        """
        初始化推理引擎
        
        关键点: 必须使用训练数据校准数据处理器中的标准化器，
               确保输入特征与训练阶段具有相同的数值分布
        
        参数:
            model_path (str): 预训练模型权重文件路径
            train_data_path (str): 训练数据文件路径（用于校准标准化器）
        """
        # 强制使用CPU设备（避免GPU驱动兼容性问题）
        self.device = torch.device("cpu")
        self.model_path = model_path
        
        print("⚙️ 推理引擎初始化中...")
        
        # 1. 初始化并校准数据处理器
        # 必须启用高级特征以保持与训练时的一致性
        self.processor = SuperconDataProcessor(use_advanced_features=True)
        print("   校准: 使用训练数据拟合标准化器...")
        
        # 加载训练数据以获取标准化器的统计参数和特征维度
        # 注: 此处的数据处理仅用于校准，不用于预测
        X_temp, _, _, _ = self.processor.load_and_process_data(train_data_path)
        
        # 自动检测输入特征维度
        self.input_size = X_temp.shape[1]
        print(f"   检测到输入特征维度: {self.input_size}")

        # 2. 加载预训练模型
        # 根据检测到的特征维度初始化模型结构
        self.model = TcPredictorAdvanced(input_size=self.input_size)
        
        try:
            # 从文件加载模型权重（强制映射到CPU设备）
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.to(self.device)  # 确保模型在CPU上运行
            self.model.eval()  # 切换到推理模式
            print(f"✅ 模型成功加载自: {model_path}")
            
        except Exception as e:
            # 模型加载失败处理（通常是架构不匹配或文件损坏）
            print(f"❌ 模型加载失败: {e}")
            print("🛑 严重错误: 模型权重加载失败，推理终止")
            print("👉 解决方案: 重新运行 'python main.py' 训练正确架构的模型")
            sys.exit(1)

    def predict(self, formula):
        """
        根据化学式预测超导临界温度
        
        参数:
            formula (str): 超导材料的化学式字符串（如"YBa2Cu3O7"）
            
        返回:
            float: 预测的临界温度Tc（单位：K），失败时返回None或0.0
            
        处理流程:
            1. 解析化学式 → 特征字典
            2. 转换为DataFrame
            3. 特征对齐与填充
            4. 标准化处理
            5. 模型预测
            6. 物理后处理
        """
        # 1. 化学式解析：将字符串转换为特征字典
        features_dict = self.processor._parse_formula(formula)
        
        if features_dict is None:
            # 化学式格式无效或无法解析
            print(f"⚠️ 无效的化学式: {formula}")
            return None
            
        # 2. 数据结构转换：字典 → DataFrame（单样本）
        df = pd.DataFrame([features_dict])
        
        # 3. 特征列对齐与填充（关键修复步骤）
        # 问题：某些简单化合物可能缺少高级特征列（如'X_variance'）
        # 解决方案：强制对齐训练时的特征列顺序，缺失值填充为0.0
        df = df.reindex(columns=self.processor.feature_columns, fill_value=0.0)
        
        # 4. 特征标准化：应用训练时学习的标准化参数
        try:
            X_np = self.processor.scaler.transform(df.values)
        except Exception as e:
            # 标准化过程异常（通常由特征维度不匹配引起）
            print(f"❌ 特征标准化错误 {formula}: {e}")
            return 0.0  # 返回安全默认值
        
        # 5. 模型推理
        with torch.no_grad():  # 禁用梯度计算以提升推理效率
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            prediction = self.model(X_tensor).item()  # 提取标量值
            
        # 6. 物理合理性后处理：Tc不能为负值
        return max(0.0, prediction)


# ==================== 主程序测试 ====================
if __name__ == "__main__":
    """
    推理模块自测试：验证模型加载和基本预测功能
    """
    
    # 配置路径（假定在项目根目录运行）
    MODEL_PATH = "best_model.pth" 
    TRAIN_PATH = "train.tsv" if os.path.exists("train.tsv") else "data/train.tsv"

    # 验证必需文件存在性
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误: 未找到预训练模型文件 'best_model.pth'")
        print("💡 提示: 请先运行 'python main.py' 训练模型")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_PATH):
        print("❌ 错误: 未找到训练数据文件")
        print("💡 提示: 请确保 'train.tsv' 文件在正确位置")
        sys.exit(1)

    # 初始化推理引擎
    print("🔧 正在初始化推理引擎...")
    engine = Inference(MODEL_PATH, TRAIN_PATH)
    print("✅ 推理引擎初始化完成")

    # 测试用例：典型超导材料的化学式
    test_materials = [
        "MgB2",           # 二硼化镁，传统超导体 (Tc ≈ 39K)
        "YBa2Cu3O7",      # YBCO，铜氧化物高温超导体 (Tc ≈ 92K)
        "HgBa2Ca2Cu3O8",  # 汞系铜氧化物，记录高温超导体 (Tc ≈ 134K)
        "FeSe",           # 铁硒化物，铁基超导体 (Tc ≈ 8K)
        "Al",             # 铝元素，I型超导体 (Tc ≈ 1.2K)
        "Pb",             # 铅元素，传统超导体 (Tc ≈ 7.2K)
        "Nb3Sn",          # 铌三锡，A15型超导体 (Tc ≈ 18K)
    ]

    # 格式化输出预测结果
    print("\n" + "="*55)
    print("🧪 超导临界温度实时预测系统 🧪")
    print("="*55)
    
    for formula in test_materials:
        tc = engine.predict(formula)
        # 对齐输出格式，增强可读性
        print(f"材料: {formula:<18} → 预测 Tc: {tc:>6.2f} K")
    
    print("="*55)
