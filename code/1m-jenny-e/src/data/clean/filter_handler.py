"""
终端及时间范围筛选模块

提供流向数据的终端和时间范围筛选功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, AnyStr

from src.utils.config_utils import config
from src.utils.logger_utils import logger
from src.utils.data_utils import add_months


class FilterHandler:
    """数据筛选处理器"""

    def __init__(self):
        self.verbose = config.get('project.verbose')
        self.tar_ym = config.get('info.tar_ym')
        self.bizym_col = config.get('columns.bizym')
        self.term_code_col = config.get('columns.term_code')

    def filter_by_active_termitutions(
            self,
            df: pd.DataFrame,
            window_months: Optional[int] = config.get('data_clean.window_months')
        ) -> Tuple[pd.DataFrame, List[str]]:
        """
        筛选指定窗口期内的活跃终端（有进货记录）
        
        Args:
            df: 输入数据
            window_months: 窗口月数
            
        Returns:
            Tuple[筛选后的数据, 活跃终端列表]
        """
        # 计算窗口期起始年月，即目标年月向前N个月
        start_ym = add_months(self.tar_ym, -window_months)

        if self.verbose:
            logger.info(f"开始筛选活跃终端，数据规模: {df.shape}")

        active_term = df[
            (df[self.bizym_col] >= start_ym) & 
            (df[self.bizym_col] < self.tar_ym)
        ][self.term_code_col].unique()
        
        df_filtered = df[df[self.term_code_col].isin(active_term)].copy()
        
        if self.verbose:
            logger.info(f"活跃终端筛选完成：近{window_months}个月有{len(active_term)}个终端产生进货行为，选出{len(df_filtered)}条记录")
        
        return df_filtered, list(active_term)

    def filter_by_time_range(
        self,
        df: pd.DataFrame,
        start_ym: Optional[int] = config.get('data_clean.start_ym'),
        end_ym: Optional[int] = config.get('data_clean.end_ym'),
    ) -> pd.DataFrame:
        """
        根据时间范围筛选数据
        
        Args:
            df: 输入数据
            start_ym: 起始年月
            end_ym: 结束年月
            
        Returns:
            筛选后的数据
        """
        # 若未指定起始或结束年月，则使用数据中的最小或最大年月
        if start_ym is None:
            start_ym = df[self.bizym_col].min()
        if end_ym is None:
            end_ym = df[self.bizym_col].max()

        if self.verbose:
            logger.info(f"开始筛选时间范围，数据规模: {df.shape}")

        df_filtered = df[
            (df[self.bizym_col] >= start_ym) & 
            (df[self.bizym_col] <= end_ym)
        ].copy()
        
        if self.verbose:
            logger.info(f"时间范围 [{start_ym}, {end_ym}] 筛选完成: 选出{len(df_filtered)} 条记录")
        
        return df_filtered