"""
超导材料临界温度预测模型训练模块
实现物理约束损失、交叉验证和完整训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
import numpy as np
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
    from model import TcPredictor, TcPredictorAdvanced
except ImportError:
    from src.data_processor import SuperconDataProcessor
    from src.model import TcPredictor, TcPredictorAdvanced


# ==================== 物理约束损失函数类 ====================
class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束增强损失函数
    在标准回归损失基础上添加超导物理先验约束
    """
    
    def __init__(self, base_loss='mse', constraint_weights=None):
        """
        初始化物理约束损失
        
        参数:
            base_loss (str): 基础损失函数类型，可选'mse'或'l1'
            constraint_weights (dict): 各约束项的权重配置
        """
        super().__init__()
        # 基础损失函数选择
        self.base_loss_fn = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()
        
        # 默认约束权重（非负约束和上界约束）
        self.weights = {'non_negative': 0.1, 'upper_bound': 0.1}
        
        # 更新用户自定义权重配置
        if constraint_weights: 
            self.weights.update(constraint_weights)
    
    def forward(self, predictions, targets, features=None):
        """
        前向传播计算总损失
        
        参数:
            predictions (Tensor): 模型预测值
            targets (Tensor): 真实标签值
            features (Tensor): 输入特征（可选，用于扩展约束）
            
        返回:
            total_loss (Tensor): 总损失值
            loss_components (dict): 各损失分项统计
        """
        # 1. 计算基础回归损失（均方误差）
        base_loss = self.base_loss_fn(predictions, targets)
        
        # 2. 约束1: 非负性约束 (Tc ≥ 0 K)
        neg_loss = torch.mean(torch.relu(-predictions) ** 2) if self.weights['non_negative'] > 0 else 0.0
        
        # 3. 约束2: 合理上界约束 (Tc < 350 K，基于已知超导材料温度上限)
        high_loss = torch.mean(torch.relu(predictions - 350.0) ** 2) if self.weights['upper_bound'] > 0 else 0.0
        
        # 4. 加权组合各损失项
        total_loss = base_loss + self.weights['non_negative'] * neg_loss + self.weights['upper_bound'] * high_loss
        
        # 返回总损失及各分项统计（用于监控）
        return total_loss, {'base': base_loss.item(), 'neg': neg_loss, 'high': high_loss}


# ==================== 交叉验证流程函数 ====================
def run_cross_validation(X, y, model_class, k_folds=5, epochs=50, 
                         batch_size=64, learning_rate=0.001, device='cpu', 
                         use_physics_constraints=False):
    """
    执行K折交叉验证评估模型性能
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 标签向量
        model_class: 模型类构造函数
        k_folds (int): 交叉验证折数
        epochs (int): 每折训练轮次
        batch_size (int): 批量大小
        learning_rate (float): 学习率
        device (str): 计算设备
        use_physics_constraints (bool): 是否启用物理约束损失
        
    返回:
        avg_r2 (float): 平均R²分数
        avg_rmse (float): 平均RMSE误差
        model_history (list): 各折训练历史（暂未返回）
    """
    print(f"\n🔄 开始{k_folds}折交叉验证...")
    
    # 强制使用CPU设备（解决RTX 5070等新硬件的CUDA兼容性问题）
    device = torch.device("cpu")
    
    # 初始化K折分割器
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []  # 存储各折评估结果
    
    # 逐折训练和评估
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        # 数据分割与设备转换
        X_train = torch.FloatTensor(X[train_ids]).to(device)
        y_train = torch.FloatTensor(y[train_ids]).to(device)
        X_val = torch.FloatTensor(X[val_ids]).to(device)
        y_val = torch.FloatTensor(y[val_ids]).to(device)
        
        # 模型初始化
        model = model_class(input_size=X.shape[1], dropout_rate=0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 损失函数选择（物理约束或标准损失）
        criterion = PhysicsConstrainedLoss() if use_physics_constraints else nn.MSELoss()
        
        # 模型训练阶段
        model.train()
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 简化的训练循环（交叉验证轮次较少）
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                # 物理约束损失返回元组，标准损失返回标量
                loss = criterion(out, by)[0] if use_physics_constraints else criterion(out, by)
                loss.backward()
                optimizer.step()
        
        # 模型评估阶段
        model.eval()
        with torch.no_grad():
            preds = model(X_val).cpu().numpy().flatten()
            targets = y_val.cpu().numpy().flatten()
            
            # 计算评估指标
            r2 = r2_score(targets, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            neg_count = np.sum(preds < 0)  # 统计不合理负预测
            
            fold_results.append({'r2': r2, 'rmse': rmse, 'neg': neg_count})
            
        # 输出本折结果
        print(f"   折{fold+1}: R²={r2:.4f}, RMSE={rmse:.4f}, 负值数={neg_count}")
    
    # 计算交叉验证平均性能
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    print(f"✅ 交叉验证平均: R²={avg_r2:.4f}, RMSE={avg_rmse:.4f}")
    
    return avg_r2, avg_rmse, None


# ==================== 主训练流程函数 ====================
def train_model(data_path, model_class=TcPredictorAdvanced, epochs=500, batch_size=64, 
                learning_rate=0.001, do_cv=True, use_advanced_features=True, 
                use_physics_constraints=False, constraint_weights=None):
    """
    完整模型训练流程，包括数据准备、交叉验证和最终训练
    
    参数:
        data_path (str): 数据文件路径
        model_class: 模型类（默认使用高级模型）
        epochs (int): 总训练轮次
        batch_size (int): 训练批量大小
        learning_rate (float): 初始学习率
        do_cv (bool): 是否执行交叉验证
        use_advanced_features (bool): 是否使用高级物理特征
        use_physics_constraints (bool): 是否启用物理约束
        constraint_weights (dict): 物理约束权重配置
        
    返回:
        model (nn.Module): 训练完成的最佳模型
        final_r2 (float): 最终R²分数
        final_rmse (float): 最终RMSE误差
    """
    print(f"--- 训练流程启动 ---")
    print(f"配置: 高级特征={use_advanced_features}, 物理损失={use_physics_constraints}")
    
    # 强制使用CPU模式（避免RTX 5070等新显卡的CUDA版本兼容性问题）
    device = torch.device("cpu")
    print(f"⚠️ 强制启用CPU模式（因RTX 5070 CUDA版本不兼容）")
    print(f"使用设备: {device}")

    # 1. 数据预处理与特征工程
    processor = SuperconDataProcessor(use_advanced_features=use_advanced_features)
    X_train_np, X_test_np, y_train_np, y_test_np = processor.load_and_process_data(data_path)
    
    # 2. 交叉验证评估（可选）
    if do_cv:
        run_cross_validation(X_train_np, y_train_np, model_class, k_folds=5, epochs=50, 
                             device=device, use_physics_constraints=use_physics_constraints)

    # 3. 全量数据最终训练
    print("\n🚀 开始最终训练...")
    
    # 数据张量转换
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    
    # 模型初始化
    model = model_class(input_size=X_train.shape[1], dropout_rate=0.2).to(device)
    
    # 优化器配置（添加L2正则化防止过拟合）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 学习率调度器（基于验证损失动态调整学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=15)
    
    # 损失函数选择
    if use_physics_constraints:
        criterion = PhysicsConstrainedLoss(constraint_weights=constraint_weights)
    else:
        criterion = nn.MSELoss()
    
    # 训练状态跟踪
    best_val_loss = float('inf')
    train_losses, val_losses, r2_scores = [], [], []

    # 数据加载器
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=batch_size, shuffle=True)
    
    # 主训练循环
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        
        # 批量训练
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            
            # 损失计算（区分物理约束和标准损失）
            if use_physics_constraints:
                loss, _ = criterion(out, by)
            else:
                loss = criterion(out, by)
            
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        # 验证阶段
        model.eval()
        with torch.no_grad():
            out_val = model(X_test)
            
            # 验证损失计算
            if use_physics_constraints:
                val_loss_item = criterion(out_val, y_test)[0].item()
            else:
                val_loss_item = criterion(out_val, y_test).item()
            
            # 验证集预测性能
            val_preds = out_val.cpu().numpy().flatten()
            val_r2 = r2_score(y_test.cpu().numpy().flatten(), val_preds)
            
            # 学习率调度
            scheduler.step(val_loss_item)
            
            # 最佳模型保存
            if val_loss_item < best_val_loss:
                best_val_loss = val_loss_item
                torch.save(model.state_dict(), 'best_model.pth')

        # 记录训练指标
        train_losses.append(ep_loss / len(train_loader))
        val_losses.append(val_loss_item)
        r2_scores.append(val_r2)

        # 定期输出训练进度
        if (epoch+1) % 20 == 0:
            print(f"轮次 [{epoch+1}/{epochs}] 验证损失: {val_loss_item:.4f}, R²: {val_r2:.4f}")

    # 保存训练过程可视化图表
    save_metrics_curves(train_losses, val_losses, r2_scores, epochs, use_physics_constraints)
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test).cpu().numpy().flatten()
        
        # 生成预测结果可视化
        create_prediction_plot(y_test.cpu().numpy().flatten(), 
                               final_preds, r2_scores[-1], use_physics_constraints)
        
    return model, r2_scores[-1], np.sqrt(best_val_loss)


def save_metrics_curves(train, val, r2, epochs, physics):
    """
    保存训练过程中的损失和R²曲线图
    
    参数:
        train (list): 训练损失历史
        val (list): 验证损失历史
        r2 (list): R²分数历史
        epochs (int): 训练总轮次
        physics (bool): 是否使用物理约束（用于文件名区分）
    """
    # 根据是否使用物理约束确定文件名后缀
    suffix = "_physics" if physics else "_baseline"
    
    # 确保图表目录存在
    if not os.path.exists("./figures"): 
        os.makedirs("./figures")
    
    # 创建双面板图表
    plt.figure(figsize=(10, 5))
    
    # 左图：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train, label='训练损失')
    plt.plot(val, label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：R²分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(r2, color='orange', label='R²分数')
    plt.xlabel('训练轮次')
    plt.ylabel('R²分数')
    plt.title('预测性能曲线')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"./figures/training_curve{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_plot(true, pred, r2, physics):
    """
    生成预测值与真实值的散点对比图
    
    参数:
        true (np.ndarray): 真实值数组
        pred (np.ndarray): 预测值数组
        r2 (float): R²分数
        physics (bool): 是否使用物理约束（用于文件名区分）
    """
    suffix = "_physics" if physics else "_baseline"
    
    plt.figure(figsize=(6, 6))
    
    # 散点图，颜色表示预测误差大小
    plt.scatter(true, pred, alpha=0.5, c=np.abs(true-pred), 
                cmap='viridis', s=20, edgecolors='black', linewidth=0.5)
    
    # 理想预测线（y=x）
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--', linewidth=2)
    
    plt.title(f"预测值 vs 真实值 (R²={r2:.3f})", fontsize=14)
    plt.xlabel("真实 Tc (K)", fontsize=12)
    plt.ylabel("预测 Tc (K)", fontsize=12)
    
    # 添加颜色条表示误差大小
    plt.colorbar(label='绝对误差 (K)')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./figures/predictions{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()