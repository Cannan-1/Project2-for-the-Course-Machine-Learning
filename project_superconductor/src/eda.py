"""
超导材料数据集探索性数据分析(EDA)模块
执行数据质量检查、分布可视化和特征相关性分析
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# ==================== 系统路径配置 ====================
# 确保项目模块正确导入，支持多种运行环境
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 动态导入自定义模块（兼容不同项目结构）
try:
    from data_processor import SuperconDataProcessor
except ImportError:
    from src.data_processor import SuperconDataProcessor


def perform_eda(data_path):
    """
    执行全面的探索性数据分析
    
    分析流程:
    1. 数据质量与分布检查
    2. 目标变量(Tc)分布可视化
    3. 特征相关性分析
    
    参数:
        data_path (str): 数据文件路径（支持.csv和.tsv格式）
    """
    print("📊 开始全数据集探索性数据分析(EDA)...")
    
    # 创建可视化图表输出目录
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    # 1. 原始数据加载与格式识别
    # 临时初始化数据处理器用于检测数据格式
    temp_processor = SuperconDataProcessor(use_advanced_features=False)
    
    try:
        # 自动检测分隔符：CSV文件使用逗号，TSV文件使用制表符
        df = pd.read_csv(data_path, sep=',' if data_path.endswith('csv') else '\t')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 基础数据统计信息输出
    print(f"📈 数据集基本信息:")
    print(f"   样本数量: {df.shape[0]:,} 行")
    print(f"   特征数量: {df.shape[1]} 列")
    print(f"   列名: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"   列名: {list(df.columns)}")

    # 智能检测Tc（临界温度）列名（支持多种命名约定）
    tc_col = next(
        (c for c in df.columns if c.lower() in ['tc', 'critical_temp', 'critical_temperature']), 
        None
    )
    if not tc_col:
        print("❌ EDA错误: 未检测到Tc临界温度标签列")
        return
    
    print(f"✅ 检测到Tc列: '{tc_col}'")

    # 2. Tc目标变量分布可视化（直方图与核密度估计）
    plt.figure(figsize=(10, 6))
    
    # 创建直方图（100个分箱以显示细节）叠加核密度估计曲线
    sns.histplot(df[tc_col], bins=100, kde=True, color='skyblue', alpha=0.7)
    
    # 标注统计信息
    mean_tc = df[tc_col].mean()
    median_tc = df[tc_col].median()
    plt.axvline(mean_tc, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_tc:.1f}K')
    plt.axvline(median_tc, color='green', linestyle='--', linewidth=1.5, label=f'中位数: {median_tc:.1f}K')
    
    plt.title(f'临界温度({tc_col})分布直方图', fontsize=14)
    plt.xlabel('临界温度 Tc (K)', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存高质量PDF格式图表
    plt.savefig("./figures/eda_tc_distribution.pdf", dpi=300, bbox_inches='tight')
    print("✅ Tc分布直方图已保存至 ./figures/eda_tc_distribution.pdf")
    plt.close()

    # 3. 特征相关性热力图分析（使用完整数据集）
    print("🔍 提取物理特征进行相关性分析（全量数据集）...")
    
    # 重新初始化高级特征处理器（用于特征工程）
    processor = SuperconDataProcessor(use_advanced_features=True)
    
    # 提取高级物理特征（使用完整数据集，不进行采样）
    feature_df, valid_rows = processor.extract_features(df)
    
    print(f"   有效特征提取样本数: {len(valid_rows):,} / {len(df):,}")
    
    # 将目标变量Tc加入特征DataFrame用于相关性分析
    target_values = df.iloc[valid_rows][tc_col].values
    feature_df['target_Tc'] = target_values
    
    # 计算完整的Pearson相关系数矩阵
    corr_matrix = feature_df.corr()
    
    # 特征选择：提取与Tc相关性最高的特征子集（防止热力图过于拥挤）
    if 'target_Tc' in corr_matrix.columns:
        # 选择与Tc绝对相关性最高的16个特征（包含Tc自身）
        top_corr_features = corr_matrix['target_Tc'].abs().sort_values(ascending=False).head(16).index
        corr_subset = corr_matrix.loc[top_corr_features, top_corr_features]
        
        # 相关性统计信息
        print(f"   Tc最相关特征:")
        for feat in top_corr_features[1:6]:  # 跳过自身，显示前5个
            if feat != 'target_Tc':
                corr_value = corr_matrix.loc['target_Tc', feat]
                print(f"      {feat}: {corr_value:+.3f}")
    else:
        corr_subset = corr_matrix

    # 创建相关性热力图
    plt.figure(figsize=(14, 12))
    
    # 使用seaborn热力图，添加数值标注和颜色映射
    sns.heatmap(
        corr_subset, 
        annot=True,                # 显示相关系数值
        cmap='coolwarm',           # 红蓝双色系（红色正相关，蓝色负相关）
        fmt=".2f",                 # 数值格式：两位小数
        linewidths=0.5,            # 单元格间线条宽度
        annot_kws={"size": 8},     # 标注文字大小
        cbar_kws={"shrink": 0.8}   # 颜色条尺寸调整
    )
    
    plt.title('关键特征与Tc的相关系数矩阵', fontsize=16, pad=20)
    plt.tight_layout()
    
    # 保存相关性热力图
    plt.savefig("./figures/eda_feature_correlation.pdf", dpi=300, bbox_inches='tight')
    print("✅ 特征相关性热力图已保存至 ./figures/eda_feature_correlation.pdf")
    plt.close()
    
    print("🎉 EDA探索性数据分析完成")


# ==================== 模块独立测试 ====================
if __name__ == "__main__":
    """
    EDA模块独立测试模式
    用于直接执行数据集探索分析，无需启动完整训练流程
    """
    
    # 尝试定位训练数据文件（支持多种路径和格式）
    data_files = ["train.tsv", "train.csv", "data/train.tsv", "data/train.csv"]
    
    data_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"📂 发现数据文件: {data_file}")
            perform_eda(data_file)
            data_found = True
            break
    
    if not data_found:
        print("❌ 错误: 未找到训练数据文件")
        print("💡 提示: 请确保以下文件之一存在:")
        for data_file in data_files[:4]:
            print(f"     - {data_file}")