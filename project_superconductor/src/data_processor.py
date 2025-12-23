"""
超导材料数据处理器模块
实现化学式解析、物理特征工程和标准化预处理全流程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition, Element


class SuperconDataProcessor:
    """
    超导材料数据集处理器
    核心功能：化学式解析、物理特征工程、数据标准化
    
    设计特点:
    1. 化学式列自动检测，支持多种命名格式
    2. 基于元素周期表的物理属性特征提取
    3. 高级物理启发特征工程（可选）
    4. 自动处理缺失值和异常数据
    """
    
    def __init__(self, use_advanced_features=True):
        """
        初始化数据处理器
        
        参数:
            use_advanced_features (bool): 是否启用高级物理启发特征，默认True
        """
        self.scaler = StandardScaler()  # 特征标准化器
        self.feature_columns = None     # 特征列名缓存
        self.use_advanced_features = use_advanced_features  # 高级特征开关
        
        # 元素物理属性集合（基于pymatgen库）
        self.properties = [
            'atomic_mass',          # 原子质量
            'atomic_radius',        # 原子半径
            'X',                    # 电负性 (Pauling标度)
            'number',               # 原子序数
            'mendeleev_no',         # 门捷列夫序号
            'melting_point',        # 熔点 (K)
            'density_of_solid',     # 固体密度 (g/cm³)
            'thermal_conductivity', # 热导率 (W/m·K)
            'row',                  # 周期表行数
            'group'                 # 周期表族数
        ]
        
        self.physics_features = []  # 高级物理特征名缓存

    def _get_element_prop(self, element, prop):
        """
        安全获取元素物理属性值
        
        参数:
            element (Element): pymatgen元素对象
            prop (str): 属性名称
            
        返回:
            float: 属性值，获取失败时返回0.0
        """
        try:
            val = getattr(element, prop)
            # 处理None值，确保返回数值类型
            return float(val) if val is not None else 0.0
        except (AttributeError, ValueError):
            # 属性不存在或类型转换失败时返回0.0
            return 0.0

    def _create_physics_features(self, element_dict, weights_dict, total_atoms):
        """
        生成物理启发特征（基于材料科学先验知识）
        
        参数:
            element_dict (dict): 元素属性字典
            weights_dict (dict): 元素组成权重字典
            total_atoms (int): 总原子数
            
        返回:
            dict: 物理启发特征字典
        """
        features = {}
        
        # 高级特征开关检查
        if not self.use_advanced_features:
            return features
        
        # 按属性组织元素值列表
        vals = {prop: [element_dict[el].get(prop, 0) for el in element_dict] 
                for prop in self.properties}
        weights = list(weights_dict.values())  # 原子分数权重
        
        # 特征1: 德拜温度代理特征 (Debye Temperature Proxy)
        # 物理基础：德拜温度 ∝ √(熔点/原子质量)
        try:
            avg_mass = np.average(vals['atomic_mass'], weights=weights)
            avg_melt = np.average(vals['melting_point'], weights=weights)
            if avg_mass > 1e-3:  # 避免除零
                features['debye_proxy'] = np.sqrt(avg_melt / avg_mass)
        except (ZeroDivisionError, ValueError):
            pass

        # 特征2: 电子-声子耦合潜力指标
        # 物理基础：电负性差异影响电子-声子耦合强度
        try:
            x_vals = vals['X']  # 电负性值列表
            if len(x_vals) > 1:
                features['X_variance'] = np.var(x_vals)  # 电负性方差
                features['X_range'] = max(x_vals) - min(x_vals)  # 电负性范围
        except (ValueError, TypeError):
            pass

        # 特征3: 晶格刚性指标
        # 物理基础：熔点与原子半径立方之比反映晶格结合强度
        try:
            avg_radius = np.average(vals['atomic_radius'], weights=weights)
            if avg_radius > 1e-3:
                features['lattice_stiffness'] = avg_melt / (avg_radius ** 3)
        except (ZeroDivisionError, ValueError):
            pass
            
        # 特征4: 电子密度代理
        # 物理基础：原子序数与密度的乘积反映电子密度
        try:
            avg_number = np.average(vals['number'], weights=weights)
            avg_density = np.average(vals['density_of_solid'], weights=weights)
            features['electron_density_proxy'] = avg_number * avg_density
        except (ValueError, TypeError):
            pass

        # 缓存高级特征名称
        self.physics_features = list(features.keys())
        return features

    def _parse_formula(self, formula):
        """
        解析化学式字符串并生成特征字典
        
        参数:
            formula (str): 化学式字符串，如"YBa2Cu3O7"
            
        返回:
            dict: 特征字典，解析失败时返回None
        """
        try:
            formula = str(formula).strip()
            
            # 输入验证：纯数字不是有效化学式
            if formula.isdigit(): 
                return None
            
            # 使用pymatgen解析化学式
            comp = Composition(formula)
            total_atoms = comp.num_atoms  # 总原子数
            element_fractions = comp.get_el_amt_dict()  # 元素组成字典
            
            # 预提取所有元素的属性（减少重复查询）
            element_dict = {}
            for el_name in element_fractions.keys():
                el = Element(el_name)
                element_dict[el_name] = {
                    p: self._get_element_prop(el, p) for p in self.properties
                }
            
            features = {}
            
            # 生成基础统计特征（各属性的加权平均值和范围）
            for prop in self.properties:
                values = [element_dict[el][prop] for el in element_fractions]
                weights = [element_fractions[el] for el in element_fractions]
                
                if weights:  # 确保有权重数据
                    # 加权平均值（反映整体性质）
                    features[f'mean_{prop}'] = np.average(values, weights=weights)
                    # 范围值（反映元素间差异）
                    features[f'range_{prop}'] = max(values) - min(values) if len(values) > 1 else 0.0
                else:
                    features[f'mean_{prop}'] = 0.0
                    features[f'range_{prop}'] = 0.0
            
            # 生成高级物理启发特征（可选）
            physics_feats = self._create_physics_features(
                element_dict, element_fractions, total_atoms
            )
            features.update(physics_feats)
            
            # 添加原子数特征
            features['num_atoms'] = total_atoms
            
            return features
            
        except (ValueError, KeyError, AttributeError):
            # 化学式解析失败（格式错误或pymatgen不支持）
            return None

    def _detect_formula_column(self, df):
        """
        智能检测数据框中的化学式列
        
        算法原理:
        1. 优先检查常见列名（formula, chemical_formula等）
        2. 对候选列进行有效性验证（包含字母且非纯数字）
        3. 选择有效样本最多的列作为化学式列
        
        参数:
            df (pd.DataFrame): 输入数据框
            
        返回:
            str: 检测到的化学式列名
        """
        best_col = None
        max_valid_score = -1
        
        # 第一优先级：常见化学式列名
        candidates = [c for c in df.columns 
                     if c.lower() in ['formula', 'chemical_formula', 'material', 'name']]
        # 第二优先级：其他所有列
        others = [c for c in df.columns if c not in candidates]
        
        # 逐列评估有效性
        for col in candidates + others:
            try:
                # 采样前100个非空值进行评估
                sample = df[col].dropna().astype(str).head(100)
                valid_count = 0
                
                for val in sample:
                    val = val.strip()
                    # 有效化学式应包含字母且非纯数字
                    if any(c.isalpha() for c in val) and not val.isdigit():
                        valid_count += 1
                
                # 选择有效样本最多的列
                if valid_count > max_valid_score:
                    max_valid_score = valid_count
                    best_col = col
            except (KeyError, AttributeError):
                continue
        
        # 默认回退策略
        if not best_col: 
            return 'formula' if 'formula' in df.columns else df.columns[0]
            
        print(f"🔍 自动检测到化学式列: '{best_col}'")
        return best_col

    def extract_features(self, df):
        """
        从原始数据框提取特征矩阵
        
        参数:
            df (pd.DataFrame): 包含化学式列的原始数据框
            
        返回:
            feature_df (pd.DataFrame): 特征数据框
            valid_rows (list): 有效样本的行索引列表
        """
        # 1. 智能检测化学式列
        formula_col = self._detect_formula_column(df)
        
        feature_list = []
        valid_rows = []
        
        print(f"正在处理化学式列 '{formula_col}'...")
        
        # 2. 逐行解析化学式
        for idx, formula in enumerate(df[formula_col]):
            feats = self._parse_formula(formula)
            if feats is not None:
                feature_list.append(feats)
                valid_rows.append(idx)
        
        # 3. 验证特征提取结果
        if not feature_list:
            raise ValueError("错误: 未能从数据中提取到有效特征")
            
        # 4. 转换为数据框并处理缺失值
        feature_df = pd.DataFrame(feature_list)
        feature_df = feature_df.fillna(0.0)  # 缺失值填充为0
        
        # 5. 输出特征统计信息
        print(f"✅ 成功创建 {feature_df.shape[1]} 个特征 "
              f"（包含 {len(self.physics_features)} 个物理启发特征）")
        
        return feature_df, valid_rows

    def load_and_process_data(self, data_path, test_size=0.2):
        """
        从文件加载数据并完成预处理全流程
        
        参数:
            data_path (str): 数据文件路径
            test_size (float): 测试集比例，默认0.2
            
        返回:
            X_train, X_test, y_train, y_test: 标准化后的训练/测试数据
            或对于无标签数据，返回标准化后的特征矩阵
        """
        print(f"📂 从 {data_path} 加载数据...")
        
        # 1. 数据文件读取（自动检测分隔符）
        try:
            df = pd.read_csv(data_path, sep=',' if data_path.endswith('csv') else '\t')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # 回退到制表符分隔
            df = pd.read_csv(data_path, sep='\t') 
            
        # 2. 智能检测Tc（临界温度）标签列
        tc_col = next(
            (c for c in df.columns if c.lower() in ['tc', 'critical_temp', 'critical_temperature']), 
            None
        )
        
        # 3. 特征提取
        feature_df, valid_rows = self.extract_features(df)
        df_clean = df.iloc[valid_rows].reset_index(drop=True)
        
        # 4. 特征列名缓存（用于推理时的特征对齐）
        self.feature_columns = feature_df.columns.tolist()
        X = feature_df.values.astype(np.float32)
        
        # 5. 数据分割与标准化
        if tc_col:
            y = df_clean[tc_col].values.astype(np.float32).reshape(-1, 1)
            
            # 划分训练测试集（固定随机种子确保可复现性）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 基于训练集拟合标准化器，并应用于训练/测试集
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            return X_train, X_test, y_train, y_test
        else:
            # 无标签数据（推理模式）
            X = self.scaler.fit_transform(X)
            return X, None, None, None