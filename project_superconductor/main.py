"""
超导材料临界温度(Tc)预测项目主程序
集成数据处理、模型训练、错误分析和预测生成全流程
"""

import os
import sys
import pandas as pd
import torch

# 系统路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

# 导入自定义模块
try:
    from src.data_processor import SuperconDataProcessor
    from src.model import TcPredictorAdvanced
    from src.train_tc_prediction import train_model
    from src.eda import perform_eda
    from src.evaluation import analyze_worst_predictions
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


def main():
    """项目主执行函数"""
    print("=== 超导材料临界温度(Tc)预测系统 ===")

    # 1. 数据文件路径自动检测
    # 优先检查当前目录，其次检查data子目录
    train_path = "train.tsv" if os.path.exists("train.tsv") else "data/train.tsv"
    test_path = "test.tsv" if os.path.exists("test.tsv") else "data/test.tsv"

    # 验证训练数据文件存在性
    if not os.path.exists(train_path):
        print("❌ 错误: 未找到训练数据文件 train.tsv")
        return

    # [阶段0] 探索性数据分析 - 全量数据
    print("\n[阶段0] 执行探索性数据分析...")
    perform_eda(train_path)

    # [阶段1] 模型训练 - 启用物理增强特征与约束
    print("\n[阶段1] 训练物理增强模型...")
    
    # 模型训练配置
    model, r2, rmse = train_model(
        data_path=train_path, 
        model_class=TcPredictorAdvanced, 
        epochs=300, 
        do_cv=True,  # 启用交叉验证
        use_advanced_features=True,    # 启用物理衍生特征
        use_physics_constraints=True,  # 启用物理约束损失
        constraint_weights={  # 约束权重配置
            'non_negative': 0.2,  # 非负约束权重
            'upper_bound': 0.1    # 上界约束权重
        }
    )
    
    # 获取模型输入维度
    input_dim = model.input_layer[0].in_features

    # [阶段2] 模型性能错误分析
    print("\n[阶段2] 执行预测错误分析...")
    if os.path.exists('best_model.pth'):
        # 重新初始化数据处理器（确保特征一致性）
        processor = SuperconDataProcessor(use_advanced_features=True)
        processor.load_and_process_data(train_path) 
        analyze_worst_predictions(model, processor, train_path)

    # [阶段3] 生成测试集预测结果
    print("\n[阶段3] 生成测试集预测结果...")
    if not os.path.exists(test_path):
        print("⚠️ 警告: 未找到测试数据文件，跳过预测生成")
        return

    # 测试集数据处理（使用与训练集相同的特征处理器）
    processor_test = SuperconDataProcessor(use_advanced_features=True)
    processor_test.load_and_process_data(train_path)  # 基于训练数据拟合标准化器
    X_test_np, _, _, _ = processor_test.load_and_process_data(test_path)
    
    if X_test_np is not None:
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            # 批量生成预测结果
            preds = model(torch.FloatTensor(X_test_np)).numpy().flatten()
        
        # 物理合理性后处理：修正负预测值
        preds[preds < 0] = 0.0
        
        # 生成提交文件
        submission_file = "submission.csv"
        pd.DataFrame({
            'Index': range(len(preds)), 
            'Predicted_Tc': preds
        }).to_csv(submission_file, index=False)
        print(f"✅ 成功生成预测文件: '{submission_file}'")


if __name__ == "__main__":
    main()