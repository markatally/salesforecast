"""
多产品规格数量融合模块

提供融合多产品规格数量的功能：
- 直接相加
- 自定义折算
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Union

from src.utils.config_utils import config
from src.utils.logger_utils import logger


class ProductFusion:
    """多品规数量融合器"""
    
    def __init__(self):
        self.verbose = config.get('project.verbose')
        self.prod_code_col = config.get('columns.prod_code')
        self.qty_col = config.get('columns.qty')
    
    def fuse(
        self,
        df: pd.DataFrame,
        method: Optional[str] = config.get('data_clean.fusion_method'),
        conversion_factors: Optional[Dict[str, float]] = config.get('data_clean.conversion_factors'),
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        融合多品规数量
        
        Args:
            df: 输入数据
            method: 融合方法
            conversion_factors: 折算系数
            group_cols: 分组列名（不为空时启用分组融合）
            
        Returns:
            pd.DataFrame: 融合后的数据
        """
        # 若未指定分组列名，则使用除产品列名和数量列名外的所有列
        group_cols = group_cols if group_cols is not None else [x for x in df.columns if x not in [self.prod_code_col, self.qty_col]]

        if self.verbose:
            logger.info(f"开始多品规数量融合，数据规模: {df.shape}")
        
        # 获取唯一品规列表
        prod_codes = df[self.prod_code_col].unique()
        if self.verbose:
            logger.info(f"总共 {len(prod_codes)} 个品规：{prod_codes}")
        
        # 根据融合方式处理
        if method == 'sum':
            df_fused = self._fuse_by_sum(df, group_cols)
        elif method == 'custom':
            df_fused = self._fuse_by_custom(df, group_cols, conversion_factors)
        else:
            raise ValueError(f"未知的融合方法: {method}")

        if self.verbose:
            logger.info(f"使用 {method} 方法融合后，数据规模: {df_fused.shape}")
        
        return df_fused

    def _fuse_by_sum(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """直接相加方式融合"""
        if self.prod_code_col in group_cols:
            group_cols.remove(self.prod_code_col)
        df = df.groupby(group_cols)[self.qty_col].sum().reset_index()

        return df
    
    def _fuse_by_custom(self, df: pd.DataFrame, group_cols: List[str], conversion_factors: Dict[str, float]) -> pd.DataFrame:
        """自定义折算方式融合"""
        # 创建副本避免修改原始数据框
        df = df.copy()

        if not conversion_factors:
            raise ValueError("使用自定义折算方式时，必须提供折算系数")
        
        # 应用折算系数
        df['converted_qty'] = df.apply(
            lambda row: row[self.qty_col] * conversion_factors.get(row[self.prod_code_col]),
            axis=1
        )
        
        # 聚合数量
        if self.prod_code_col in group_cols:
            group_cols.remove(self.prod_code_col)
        df = df.groupby(group_cols)['converted_qty'].sum().reset_index()
        df = df.rename(columns={'converted_qty': self.qty_col})
        
        return df