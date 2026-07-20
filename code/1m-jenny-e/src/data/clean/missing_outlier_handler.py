"""
缺失值 & 异常值处理模块

提供数据清洗中缺失值和异常值的检测与处理功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List

from src.utils.config_utils import config
from src.utils.logger_utils import logger


# ==========================================
# 缺失值处理器
# ==========================================
class MissingValueHandler:
    """缺失值处理器"""

    def __init__(self):
        self.verbose = config.get('project.verbose')
    
    def handle(
        self,
        df: pd.DataFrame,
        tar_col: Optional[str] = config.get('columns.qty'),
        method: Optional[str] = config.get('data_clean.missing_method'),
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据
            tar_col: 目标列名
            method: 缺失值处理方法
            group_cols: 分组列名（不为空时启用分组填充）
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if self.verbose:
            logger.info(f"开始处理缺失值，数据规模: {df.shape}")
        
        # 检测&处理缺失值
        df_pcd = self._handle_missing_values(df, tar_col, method, group_cols)
        
        if self.verbose:
            logger.info(f"缺失值处理完成，数据规模: {df_pcd.shape}")

        return df_pcd
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        tar_col: str,
        method: str,
        group_cols: List[str]
    ) -> pd.DataFrame:
        """
        处理缺失值，支持分组填充

        支持的处理策略：
            - 'ffill': 前向填充
            - 'bfill': 后向填充
            - 'interpolate': 线性插值
            - 'mean': 替换为均值
            - 'median': 替换为中位数
            - 'zero': 替换为0
            - 'drop': 直接删除

        Args:
            df: 输入数据
            tar_col: 目标列名
            method: 缺失值处理方法
            group_cols: 分组列名（不为空时启用分组填充）

        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 创建副本避免修改原始数据框
        df = df.copy()

        # 检测缺失值
        missing_count = df[tar_col].isna().sum()
        if missing_count == 0:
            if self.verbose:
                logger.info(f"列 {tar_col} 未检测到缺失值")
            return df
        if self.verbose:
            logger.info(f"列 {tar_col} 检测到 {missing_count} 条缺失值记录")
        
        # 处理缺失值
        if group_cols:
            # 分组填充
            if method == 'ffill':
                df[tar_col] = df.groupby(group_cols)[tar_col].ffill()
            elif method == 'bfill':
                df[tar_col] = df.groupby(group_cols)[tar_col].bfill()
            elif method == 'interpolate':
                df[tar_col] = df.groupby(group_cols)[tar_col].apply(
                    lambda x: x.interpolate(method='linear')
                )
            elif method == 'mean':
                df[tar_col] = df.groupby(group_cols)[tar_col].transform(
                    lambda x: x.fillna(x.mean())
                )
            elif method == 'median':
                df[tar_col] = df.groupby(group_cols)[tar_col].transform(
                    lambda x: x.fillna(x.median())
                )
            elif method == 'zero':
                df[tar_col] = df[tar_col].fillna(0)
            else:
                raise ValueError(f"未知的缺失值处理策略: {method}")
        else:
            # 直接填充
            if method == 'ffill':
                df[tar_col] = df[tar_col].ffill()
            elif method == 'bfill':
                df[tar_col] = df[tar_col].bfill()
            elif method == 'interpolate':
                df[tar_col] = df[tar_col].interpolate(method='linear')
            elif method == 'mean':
                df[tar_col] = df[tar_col].fillna(df[tar_col].mean())
            elif method == 'median':
                df[tar_col] = df[tar_col].fillna(df[tar_col].median())
            elif method == 'zero':
                df[tar_col] = df[tar_col].fillna(0)
            else:
                raise ValueError(f"未知的缺失值处理策略: {method}")
        
        # 计算剩余缺失值数量
        remaining = df[tar_col].isna().sum()
        if self.verbose:
            logger.info(f"使用 {method} 策略处理了 {missing_count} 条缺失值记录，剩余 {remaining} 条缺失值记录")

        return df


# ==========================================
# 异常值处理
# ==========================================
class OutlierHandler:
    """异常值处理器"""
    
    def __init__(self):
        self.verbose = config.get('project.verbose')
    
    def handle(
        self,
        df: pd.DataFrame,
        tar_col: Optional[str] = config.get('columns.qty'),
        detect_method: Optional[str] = config.get('data_clean.outlier_detect'),
        handle_method: Optional[str] = config.get('data_clean.outlier_handle'),
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        处理异常值

        Args:
            df: 输入数据
            tar_col: 目标列名
            detect_method: 异常值检测方法
            handle_method: 异常值处理方法
            **kwargs: 其他参数 (threshold | iqr_window | lower_percentile | upper_percentile | smooth_window)

        Returns:
            Tuple[处理后的数据, 异常值信息]
        """
        if self.verbose:
            logger.info(f"开始检测异常值，数据规模: {df.shape}")

        # 检测异常值
        outliers = self._detect_outliers(df, tar_col, detect_method, **kwargs)

        # 提取异常值索引
        outlier_indices = outliers.get(tar_col, [])

        # 记录异常值检测结果
        if outlier_indices:
            if self.verbose:
                logger.info(f"列 {tar_col} 检测到 {len(outlier_indices)} 条异常值记录")
            outlier_info = df.loc[outlier_indices].copy()
        else:
            if self.verbose:
                logger.info(f"列 {tar_col} 未检测到异常值")
            outlier_info = pd.DataFrame(columns=df.columns)
            return df, outlier_info

        if self.verbose:
            logger.info(f"开始处理异常值，数据规模: {df.shape}")

        # 处理异常值
        df_pcd = self._handle_outliers(df, tar_col, outliers, handle_method, **kwargs)

        if self.verbose:
            logger.info(f"异常值处理完成，数据规模: {df_pcd.shape}")

        return df_pcd, outlier_info

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        tar_col: str,
        method: str,
        **kwargs
    ) -> Dict[str, List]:
        """
        检测异常值

        支持的检测方法：
            - 'iqr': 四分位距法（每个点依次作为滚动窗口的中心，对滚动窗口应用四分位距法判断该点是否为异常值）
            - 'zscore': Z-Score法
            - 'percentile': 百分位数法

        Args:
            df: 输入数据
            tar_col: 目标列名
            method: 异常值检测方法
            **kwargs: 其他参数 (threshold | iqr_window | lower_percentile | upper_percentile)

        Returns:
            Dict[str, List]: 异常值索引字典 {列名: [索引列表]}
        """
        outliers = {}
        col_data = df[tar_col]

        # 检测异常值
        if method == 'iqr':
            threshold = kwargs.get('threshold', config.get('data_clean.iqr_threshold'))
            window_size = kwargs.get('iqr_window', config.get('data_clean.iqr_window_size'))
            if window_size > 0:
                # 使用滚动窗口检测异常值
                outlier_indices = []
                for i in range(len(col_data)):
                    start = max(0, i - window_size // 2)
                    end = min(len(col_data), i + window_size // 2 + 1)
                    window = col_data.iloc[start:end]
                    q1 = window.quantile(0.25)
                    q3 = window.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    if col_data.iloc[i] < lower_bound or col_data.iloc[i] > upper_bound:
                        outlier_indices.append(col_data.index[i])
                outliers[tar_col] = outlier_indices
            else:
                # 使用全局检测异常值
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_indices = df[(col_data < lower_bound) | (col_data > upper_bound)].index.tolist()
                outliers[tar_col] = outlier_indices

        elif method == 'zscore':
            threshold = kwargs.get('threshold', config.get('data_clean.zscore_threshold'))
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outlier_indices = df[z_scores > threshold].index.tolist()
            outliers[tar_col] = outlier_indices

        elif method == 'percentile':
            lower_percentile = kwargs.get('lower_percentile', config.get('data_clean.lower_percentile'))
            upper_percentile = kwargs.get('upper_percentile', config.get('data_clean.upper_percentile'))
            lower_bound = col_data.quantile(lower_percentile)
            upper_bound = col_data.quantile(upper_percentile)
            outlier_indices = df[(col_data < lower_bound) | (col_data > upper_bound)].index.tolist()
            outliers[tar_col] = outlier_indices

        else:
            raise ValueError(f"未知的异常值检测方法: {method}")

        total_outliers = sum(len(indices) for indices in outliers.values())
        if self.verbose:
            logger.info(f"使用 {method} 方法检测到 {total_outliers} 个异常值")

        return outliers

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        tar_col: str,
        outliers: Dict[str, List],
        method: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        处理异常值

        支持的处理策略：
            - 'clip': 截断为非异常值边界值
            - 'drop': 直接删除
            - 'interpolate': 线性插值
            - 'ffill': 前向填充
            - 'bfill': 后向填充
            - 'mean': 替换为非异常值的均值
            - 'median': 替换为非异常值的中位数
            - 'smooth': 替换为非异常值的移动平均值

        Args:
            df: 输入数据
            tar_col: 目标列名
            outliers: 异常值索引字典
            method: 异常值处理方法
            **kwargs: 其他参数 (smooth_window)

        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 创建副本避免修改原始数据框
        df = df.copy()

        # 提取异常值&非异常值索引
        outlier_indices = outliers[tar_col]
        non_outlier_indices = list(set(df.index) - set(outlier_indices))
        
        # 处理异常值
        if method == 'clip':
            # 找到非异常值的边界
            if non_outlier_indices:
                non_outlier_values = df.loc[non_outlier_indices, tar_col]
                min_val = non_outlier_values.min()
                max_val = non_outlier_values.max()
            else:
                logger.warning(f"列 {tar_col} 未找到非异常值，使用全局边界值")
                min_val = df[tar_col].min()
                max_val = df[tar_col].max()
            for idx in outlier_indices:
                val = df.at[idx, tar_col]
                if val < min_val:
                    df.at[idx, tar_col] = min_val
                elif val > max_val:
                    df.at[idx, tar_col] = max_val

        elif method == 'drop':
            df = df.drop(outlier_indices)

        elif method == 'interpolate':
            df.loc[outlier_indices, tar_col] = np.nan
            df[tar_col] = df[tar_col].interpolate(method='linear')
            # 检查首尾异常值并提醒
            if pd.isna(df[tar_col].iloc[0]) or pd.isna(df[tar_col].iloc[-1]):
                logger.warning(f"注意: 异常值位于序列首尾位置，使用 {method} 方法会产生NaN值")

        elif method == 'ffill':
            df.loc[outlier_indices, tar_col] = np.nan
            df[tar_col] = df[tar_col].ffill()
            # 检查首尾异常值并提醒
            if pd.isna(df[tar_col].iloc[0]):
                logger.warning(f"注意: 异常值位于序列起始位置，使用 {method} 方法会产生NaN值")

        elif method == 'bfill':
            df.loc[outlier_indices, tar_col] = np.nan
            df[tar_col] = df[tar_col].bfill()
            # 检查首尾异常值并提醒
            if pd.isna(df[tar_col].iloc[-1]):
                logger.warning(f"注意: 异常值位于序列结束位置，使用 {method} 方法会产生NaN值")

        elif method == 'mean':
            if non_outlier_indices:
                mean_val = df.loc[non_outlier_indices, tar_col].mean()
            else:
                logger.warning(f"列 {tar_col} 未找到非异常值，使用全局均值")
                mean_val = df[tar_col].mean()
            for idx in outlier_indices:
                df.at[idx, tar_col] = mean_val

        elif method == 'median':
            if non_outlier_indices:
                median_val = df.loc[non_outlier_indices, tar_col].median()
            else:
                logger.warning(f"列 {tar_col} 未找到非异常值，使用全局中位数")
                median_val = df[tar_col].median()
            for idx in outlier_indices:
                df.at[idx, tar_col] = median_val

        elif method == 'smooth':
            smooth_window = kwargs.get('smooth_window', config.get('data_clean.smooth_window_size'))
            smoothed = df[tar_col].rolling(
                window=smooth_window, center=True, min_periods=1
            ).mean()
            for idx in outlier_indices:
                df.at[idx, tar_col] = smoothed.at[idx]

        else:
            raise ValueError(f"未知的异常值处理方法: {method}")

        if self.verbose:
            logger.info(f"使用 {method} 方法处理了 {len(outlier_indices)} 个异常值")
        
        return df