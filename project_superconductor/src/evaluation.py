"""
超导材料临界温度预测模型性能评估模块
提供模型定量评估指标计算、报告生成和错误样本分析功能
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import pandas as pd
import sys

# ==================== 系统路径配置 ====================
# 确保项目模块正确导入，支持多种运行环境
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 动态导入自定义模块（兼容不同目录结构）
try:
    from data_processor import SuperconDataProcessor
    from model import TcPredictorAdvanced
except ImportError:
    from src.data_processor import SuperconDataProcessor
    from src.model import TcPredictorAdvanced


def evaluate_model(model, X_test, y_test, model_name="TcPredictorAdvanced"):
    """
    模型性能综合评估函数
    
    计算回归任务关键评估指标，包括R²分数、均方误差、均方根误差和平均绝对误差
    
    参数:
        model (nn.Module): 已训练的PyTorch模型
        X_test (np.ndarray/torch.Tensor): 测试集特征矩阵
        y_test (np.ndarray/torch.Tensor): 测试集真实标签
        model_name (str): 模型名称标识，用于报告
        
    返回:
        dict: 包含各项评估指标的字典
    """
    # 模型切换到评估模式（禁用Dropout等训练特定层）
    model.eval()
    
    # 强制使用CPU设备确保兼容性
    device = torch.device("cpu")
    model.to(device)
    
    # 数据类型统一：确保输入为PyTorch张量
    if isinstance(X_test, np.ndarray): 
        X_test = torch.FloatTensor(X_test)
    if isinstance(y_test, np.ndarray): 
        y_test = torch.FloatTensor(y_test)
    
    # 数据转移到指定设备
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # 推理阶段：禁用梯度计算以提升性能
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    # 转换为numpy数组便于计算指标
    y_test = y_test.cpu().numpy().flatten()
    
    # 计算各项回归评估指标
    return {
        "model_name": model_name,                            # 模型标识
        "n_samples": len(y_test),                           # 测试样本数量
        "r2_score": r2_score(y_test, y_pred),               # R²决定系数（0-1，越高越好）
        "mse": mean_squared_error(y_test, y_pred),          # 均方误差
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), # 均方根误差（单位与目标相同）
        "mae": mean_absolute_error(y_test, y_pred)          # 平均绝对误差（鲁棒性更好）
    }


def save_evaluation_report(metrics, filepath="evaluation_report.txt"):
    """
    生成并保存格式化评估报告
    
    参数:
        metrics (dict): evaluate_model函数返回的指标字典
        filepath (str): 报告文件保存路径，默认"evaluation_report.txt"
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # 报告头部信息
        f.write("="*60 + "\n")
        f.write("       超导材料临界温度(Tc)预测模型评估报告\n")
        f.write("="*60 + "\n\n")
        
        # 基础信息部分
        f.write(f"📊 模型基本信息\n")
        f.write(f"   - 模型名称:      {metrics['model_name']}\n")
        f.write(f"   - 测试样本数:    {metrics['n_samples']:,} 条\n\n")
        
        # 性能指标部分
        f.write(f"📈 模型性能指标\n")
        f.write("-" * 50 + "\n")
        f.write(f"   R²决定系数:     {metrics['r2_score']:.4f}\n")
        f.write(f"   RMSE(均方根误差): {metrics['rmse']:.4f} K\n")
        f.write(f"   MAE(平均绝对误差): {metrics['mae']:.4f} K\n")
        f.write(f"   MSE(均方误差):    {metrics['mse']:.4f}\n")
        f.write("-" * 50 + "\n\n")
        
        # 性能解读指导
        f.write(f"📋 指标解读说明\n")
        f.write(f"   • R²分数范围[0,1]，越接近1表示模型解释力越强\n")
        f.write(f"   • RMSE和MAE单位均为开尔文(K)，数值越小预测越精准\n")
        f.write(f"   • 建议对比不同模型架构的指标以选择最佳方案\n")
    
    print(f"📄 详细评估报告已保存至: {filepath}")


def analyze_worst_predictions(model, processor, data_path, top_n=10):
    """
    错误样本深度分析：识别预测误差最大的样本
    
    参数:
        model (nn.Module): 已训练的预测模型
        processor (SuperconDataProcessor): 数据处理器实例
        data_path (str): 原始数据文件路径
        top_n (int): 分析的最大误差样本数量，默认10个
    """
    print("\n🔍 正在执行预测误差分析（识别最差预测样本）...")
    
    try:
        # 尝试CSV格式读取（逗号分隔）
        df = pd.read_csv(data_path, sep=',')
    except:
        # 回退到TSV格式读取（制表符分隔）
        df = pd.read_csv(data_path, sep='\t')
    
    # 智能检测Tc标签列名（支持多种命名约定）
    tc_col = next(
        (c for c in df.columns if c.lower() in ['tc', 'critical_temp', 'temp', 'critical_temperature']), 
        None
    )
    
    if tc_col is None:
        print("⚠️ 警告: 未检测到Tc标签列，错误分析终止")
        return
    
    # 获取与训练时一致的数据分割索引
    # 确保错误分析针对相同的测试集样本
    from sklearn.model_selection import train_test_split
    feature_df, valid_rows = processor.extract_features(df)
    df_clean = df.iloc[valid_rows].reset_index(drop=True)
    indices = np.arange(len(df_clean))
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 提取测试集数据
    df_test = df_clean.iloc[test_indices].copy()
    X_test_np, _, _, _ = processor.load_and_process_data(data_path)
    # X_test_np 已按相同分割比例处理为测试集
    
    # 模型推理
    X_test_tensor = torch.FloatTensor(X_test_np).to(torch.device("cpu"))
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy().flatten()
    
    # 数据长度对齐检查
    if len(preds) != len(df_test):
        min_len = min(len(preds), len(df_test))
        df_test = df_test.iloc[:min_len]
        preds = preds[:min_len]

    # 计算绝对误差并排序
    df_test['Predicted_Tc'] = preds
    df_test['Abs_Error'] = np.abs(df_test[tc_col] - df_test['Predicted_Tc'])
    
    # 获取误差最大的top_n个样本
    worst_cases = df_test.sort_values(by='Abs_Error', ascending=False).head(top_n)
    
    # 格式化输出误差分析结果
    print(f"\n🏆 预测误差最大的{top_n}个样本:")
    print(f"{'化学式':<20} | {'真实Tc':<10} | {'预测Tc':<10} | {'绝对误差':<10}")
    print("-" * 60)
    
    for _, row in worst_cases.iterrows():
        # 智能检测化学式列名
        formula_col = next(
            (c for c in row.index if c.lower() in ['formula', 'name', 'material']), 
            'N/A'
        )
        formula = row[formula_col] if formula_col != 'N/A' else "N/A"
        
        # 行格式输出
        print(f"{str(formula):<20} | {row[tc_col]:<10.2f} | {row['Predicted_Tc']:<10.2f} | {row['Abs_Error']:<10.2f}")
    
    # 保存详细错误分析结果到CSV文件
    worst_cases.to_csv("error_analysis_worst_cases.csv", index=False, encoding='utf-8')
    print(f"\n✅ 误差最大样本已保存至 'error_analysis_worst_cases.csv'")


# ==================== 独立运行模式 ====================
if __name__ == "__main__":
    """
    评估模块独立运行模式
    用于在不启动完整训练流程的情况下执行模型性能评估
    """
    print("--- 模型评估模块独立运行模式 ---")
    
    # 1. 自动检测数据文件和模型文件路径
    train_path = "train.tsv" if os.path.exists("train.tsv") else "data/train.tsv"
    model_path = "best_model.pth"
    
    # 文件存在性验证
    if not os.path.exists(train_path):
        print("❌ 错误: 未找到训练数据文件 train.tsv")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print("❌ 错误: 未找到预训练模型文件 best_model.pth")
        print("💡 提示: 请先运行 main.py 训练模型")
        sys.exit(1)
    
    # 2. 数据预处理（使用与训练时相同的高级特征配置）
    # 确保标准化器参数与训练阶段完全一致
    processor = SuperconDataProcessor(use_advanced_features=True)
    print("📊 正在处理数据（确保标准化器参数与训练时一致）...")
    _, X_test, _, y_test = processor.load_and_process_data(train_path)
    
    # 3. 模型加载
    input_dim = X_test.shape[1]  # 从数据自动推断输入维度
    model = TcPredictorAdvanced(input_size=input_dim)
    
    try:
        # 加载预训练权重（强制CPU设备以确保兼容性）
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        print("✅ 模型加载成功")
        
        # 4. 执行综合评估并生成报告
        metrics = evaluate_model(model, X_test, y_test)
        save_evaluation_report(metrics)  # 生成evaluation_report.txt
        
        # 5. 执行错误样本分析
        analyze_worst_predictions(model, processor, train_path)  # 生成error_analysis_worst_cases.csv
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        print("💡 可能原因: 模型架构与权重不匹配或数据预处理不一致")