"""
超导材料临界温度预测神经网络模型定义
包含基准模型与高级残差网络模型两种架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基准模型 (标准多层感知机)
# 保留用于代码兼容性和性能对比基准
# ============================================================
class TcPredictor(nn.Module):
    """
    标准多层感知机基准模型
    采用经典全连接网络结构，作为性能对比的参考基线
    
    网络结构: Input -> [Linear -> BatchNorm -> ReLU -> Dropout]×N -> Output
    """
    
    def __init__(self, input_size, hidden_sizes=None, dropout_rate=0.2):
        """
        初始化基准模型架构
        
        参数:
            input_size (int): 输入特征维度
            hidden_sizes (list): 各隐藏层的神经元数量，默认[256, 128, 64]
            dropout_rate (float): Dropout正则化比例，默认0.2
        """
        super(TcPredictor, self).__init__()
        
        # 默认隐藏层配置：三层递减结构
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        
        # 动态构建隐藏层序列
        layers = []
        prev_size = input_size  # 前一层神经元数量
        
        # 逐层构建网络模块
        for size in hidden_sizes:
            # 线性变换层
            layers.append(nn.Linear(prev_size, size))
            # 批量归一化层（加速收敛，提升稳定性）
            layers.append(nn.BatchNorm1d(size))
            # ReLU激活函数（引入非线性）
            layers.append(nn.ReLU())
            # Dropout正则化（防止过拟合）
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size  # 更新前一层尺寸
            
        # 将层序列封装为顺序模块
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层（回归任务，输出维度为1）
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        """
        前向传播过程
        
        参数:
            x (torch.Tensor): 输入特征张量，形状为(batch_size, input_size)
            
        返回:
            torch.Tensor: 预测的Tc值，形状为(batch_size, 1)
        """
        # 通过隐藏层序列提取特征
        x = self.hidden_layers(x)
        # 通过输出层生成最终预测
        return self.output_layer(x)


# ============================================================
# 2. 高级模型 (残差网络增强版)
# 采用残差连接、GELU激活和更深更宽的网络结构
# ============================================================
class ResidualBlock(nn.Module):
    """
    双层残差连接块
    结构: Input -> [Linear -> BN -> GELU -> Dropout]×2 -> Add(Input) -> Output
    
    残差连接设计能够缓解深层网络梯度消失问题，提升训练稳定性
    """
    
    def __init__(self, in_size, out_size, dropout_rate=0.2):
        """
        初始化残差块
        
        参数:
            in_size (int): 输入特征维度
            out_size (int): 输出特征维度
            dropout_rate (float): Dropout比例，默认0.2
        """
        super(ResidualBlock, self).__init__()
        
        # 第一变换层：线性+批量归一化+GELU激活+Dropout
        self.layer1 = nn.Sequential(
            nn.Linear(in_size, out_size),      # 线性变换
            nn.BatchNorm1d(out_size),          # 批量归一化
            nn.GELU(),                         # GELU激活函数（优于ReLU）
            nn.Dropout(dropout_rate)           # Dropout正则化
        )
        
        # 第二变换层：保持维度不变的变换
        self.layer2 = nn.Sequential(
            nn.Linear(out_size, out_size),     # 维度保持
            nn.BatchNorm1d(out_size),          # 批量归一化
            nn.GELU(),                         # GELU激活
            nn.Dropout(dropout_rate)           # Dropout正则化
        )
        
        # 捷径连接：输入到输出的直接映射
        # 当输入输出维度不匹配时，需要进行投影变换
        if in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Linear(in_size, out_size),  # 维度投影
                nn.BatchNorm1d(out_size)       # 归一化
            )
        else:
            # 维度相同时使用恒等映射
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        残差块前向传播
        
        参数:
            x (torch.Tensor): 输入特征张量
            
        返回:
            torch.Tensor: 残差连接后的输出
        """
        identity = self.shortcut(x)   # 捷径连接输出
        out = self.layer1(x)          # 第一层变换
        out = self.layer2(out)        # 第二层变换
        return out + identity         # 残差相加：F(x) + x


class TcPredictorAdvanced(nn.Module):
    """
    高级Tc预测模型 (TcPredictorPro)
    采用更深、更宽的残差网络架构，增强特征提取能力
    
    网络结构: 
        Input -> 输入嵌入层 -> [残差块]×N -> 输出层
    """
    
    def __init__(self, input_size, hidden_sizes=None, dropout_rate=0.2):
        """
        初始化高级模型
        
        参数:
            input_size (int): 输入特征维度
            hidden_sizes (list): 隐藏层维度序列，默认[1024, 512, 256, 128]
            dropout_rate (float): Dropout比例，默认0.2
        """
        super(TcPredictorAdvanced, self).__init__()
        
        # 1. 网络宽度扩展：默认使用更宽的层结构
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256, 128] 
            
        # 2. 输入嵌入层：初始特征变换与降维
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),  # 扩展到高维空间
            nn.BatchNorm1d(hidden_sizes[0]),         # 归一化
            nn.GELU(),                               # GELU激活
            nn.Dropout(dropout_rate)                 # 正则化
        )
        
        # 3. 残差块堆叠：构建深度特征提取网络
        self.blocks = nn.ModuleList()
        prev_size = hidden_sizes[0]
        
        # 依次添加残差块，逐步降低特征维度
        for size in hidden_sizes[1:]:
            self.blocks.append(
                ResidualBlock(prev_size, size, dropout_rate)
            )
            prev_size = size  # 更新前一层维度
            
        # 4. 输出层：回归预测
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        """
        高级模型前向传播
        
        参数:
            x (torch.Tensor): 输入特征张量
            
        返回:
            torch.Tensor: 预测的Tc值
        """
        # 初始特征变换
        x = self.input_layer(x)
        
        # 通过残差块序列提取深层特征
        for block in self.blocks:
            x = block(x)
            
        # 生成最终预测
        return self.output_layer(x)


# ============================================================
# 模块功能测试
# ============================================================
if __name__ == "__main__":
    """
    模块自测试：验证模型定义的正确性和实例化能力
    """
    try:
        # 测试基准模型实例化
        m1 = TcPredictor(21)
        print("✅ TcPredictor实例化成功")
        
        # 测试高级模型实例化
        m2 = TcPredictorAdvanced(21)
        print("✅ TcPredictorAdvanced实例化成功")
        
        # 测试前向传播（使用随机输入）
        test_input = torch.randn(4, 21)  # 批量大小=4，特征维度=21
        
        output1 = m1(test_input)
        print(f"   基准模型输出形状: {output1.shape}")
        
        output2 = m2(test_input)
        print(f"   高级模型输出形状: {output2.shape}")
        
        print("✅ model.py模块测试通过！")
        
    except Exception as e:
        # 捕获并报告任何实例化错误
        print(f"❌ 模型实例化错误: {e}")