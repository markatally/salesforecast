"""
退货记录处理模块

提供终端层面退货记录的处理功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from src.utils.config_utils import config
from src.utils.logger_utils import logger


class ReturnHandler:
    """退货记录处理器"""

    def __init__(self):
        self.verbose = config.get('project.verbose')
        self.date_col = config.get('columns.date')
        self.qty_col = config.get('columns.qty')
        self.term_code_col = config.get('columns.term_code')
    
    def handle(
        self, 
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        处理退货记录
        
        Args:
            df: 输入数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        if self.verbose:
            logger.info(f"开始处理退货记录")
        
        log_info = {
            'original_total': len(df),
            'original_returns': len(df[df[self.qty_col] < 0]),
            'phase1_removed': 0,
            'phase1_offset': 0,
            'phase2_removed': 0,
            'final_returns': 0
        }
        
        # Phase 1: 逐条向前抵消
        df_pcd, idx_list_full, idx_list_partial, idx_list_offset = self._process_return_phase1(df)
        df_pcd = df_pcd.drop(index=idx_list_full+idx_list_partial+idx_list_offset)
        
        if self.verbose:
            logger.info(f"Phase 1 完成：")
            logger.info(f"  - 移除 {len(idx_list_full)} 条完全抵消的退货记录")
            logger.info(f"  - 移除 {len(idx_list_partial)} 条部分抵消的退货记录")
            logger.info(f"  - 移除 {len(idx_list_offset)} 条抵消后为零的退货记录")
        log_info['phase1_removed'] = len(idx_list_full) + len(idx_list_partial)
        log_info['phase1_offset'] = len(idx_list_offset)
        
        # Phase 2: 去除因为处于起始位置而未能向前抵消的记录
        idx_list_phase2 = self._process_return_phase2(df_pcd)
        df_pcd = df_pcd.drop(index=idx_list_phase2)
        
        if self.verbose:
            logger.info(f"Phase 2 完成: 移除 {len(idx_list_phase2)} 条起始位置的退货记录")
        log_info['phase2_removed'] = len(idx_list_phase2)
        log_info['final_returns'] = len(df_pcd[df_pcd[self.qty_col] < 0])

        if self.verbose:
            logger.info(f"退货记录处理完成:")
            logger.info(f"  - 原始总条数: {log_info['original_total']}")
            logger.info(f"  - 原始退货条数: {log_info['original_returns']}")
            logger.info(f"  - Phase 1 移除退货条数: {log_info['phase1_removed']}")
            logger.info(f"  - Phase 1 抵消进货条数: {log_info['phase1_offset']}")
            logger.info(f"  - Phase 2 移除退货条数: {log_info['phase2_removed']}")
            logger.info(f"  - 最终记录条数: {len(df_pcd)}")
            logger.info(f"  - 剩余退货条数: {log_info['final_returns']}")
        return df_pcd
    
    def _process_return_phase1(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], List[int], List[int]]:
        """
        处理退货记录 Phase 1: 逐条向前抵消
        
        Args:
            df: 输入数据

        Returns:
            Tuple[处理后的数据, 完全抵消的索引列表, 部分抵消的索引列表, 抵消后为零的索引列表]
        """
        # 创建副本避免修改原始数据框
        df = df.copy()

        idx_list_full = []
        idx_list_partial = []
        idx_list_offset = []
        
        # 获取有退货的机构列表
        return_terms = df[df[self.qty_col] < 0][self.term_code_col].unique()
        if self.verbose:
            logger.info(f"Phase 1: 发现 {len(return_terms)} 家有退货的机构")
        
        # 遍历有退货的机构
        for term in return_terms:
            # 获取该机构的交易记录，按日期排序
            term_df = df[df[self.term_code_col] == term][
                [self.date_col, self.qty_col]
            ].sort_values(by=self.date_col).reset_index()
            
            # 获取退货记录
            return_records = term_df[term_df[self.qty_col] < 0]
            
            # 从前向后遍历退货记录
            for idx in return_records.index:
                return_date = term_df.loc[idx, self.date_col]
                return_qty = -term_df.loc[idx, self.qty_col]
                
                # 从该条退货记录开始，向前搜索可以抵消的记录
                pre_records = term_df[term_df[self.date_col] < return_date]
                
                if len(pre_records) > 0:
                    for i, row in pre_records[::-1].iterrows():
                        if row[self.qty_col] > 0:
                            # 当前记录足够抵消退货量
                            if row[self.qty_col] >= return_qty:
                                df.loc[row['index'], self.qty_col] -= return_qty
                                term_df.loc[i, self.qty_col] -= return_qty
                                return_qty = 0
                                # 被抵消掉的进货记录
                                if term_df.loc[i, self.qty_col] == 0:
                                    idx_list_offset.append(row['index'])
                                break
                            # 当前记录仅能抵消部分退货量
                            else:
                                df.loc[row['index'], self.qty_col] = 0
                                term_df.loc[i, self.qty_col] = 0
                                return_qty -= row[self.qty_col]
                                # 被抵消掉的进货记录
                                idx_list_offset.append(row['index'])
                
                # 待移除的退货记录索引
                if return_qty < -term_df.loc[idx, self.qty_col]: # 部分抵消
                    idx_list_partial.append(term_df.loc[idx, 'index'])
                elif return_qty == 0: # 完全抵消
                    idx_list_full.append(term_df.loc[idx, 'index'])
        
        return df, idx_list_full, idx_list_partial, idx_list_offset
    
    def _process_return_phase2(self, df: pd.DataFrame) -> List[int]:
        """
        处理退货记录 Phase 2: 去除因为处于起始位置而未能向前抵消的记录
        
        Args:
            df: 输入数据

        Returns:
            需要移除的索引列表
        """
        # 创建副本避免修改原始数据框
        df = df.copy()

        idx_list = []
        
        # 获取有退货的机构列表
        return_terms = df[df[self.qty_col] < 0][self.term_code_col].unique()
        if self.verbose:
            logger.info(f"Phase 2: 发现 {len(return_terms)} 家有退货的机构")

        for term in return_terms:
            # 获取该机构的交易记录，按日期排序
            term_df = df[df[self.term_code_col] == term][
                [self.date_col, self.qty_col]
            ].sort_values(by=self.date_col).reset_index()
            
            # 获取开头连续为负的记录
            negative_mask = term_df[self.qty_col] <= 0 # 覆盖退货记录之间夹杂零记录的情况，Ex. -2, 0, -1, ...
            if negative_mask.any():
                # 累计乘积：一旦遇到非负值，后续都为False
                continuous_negative = (negative_mask.cumprod() == 1)
                negative_at_start = term_df[continuous_negative]
                negative_at_start = negative_at_start[negative_at_start[self.qty_col] < 0] # 去除零记录
                idx_list.extend(negative_at_start['index'].tolist())
        
        return idx_list
    
    def analyze_returns(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        退货情况的基本统计分析
        
        Args:
            df: 输入数据

        Returns:
            Tuple[整体层面统计, 机构层面统计]
        """
        # 退货数据、进货数据
        return_records = df[df[self.qty_col] < 0]
        positive_records = df[df[self.qty_col] > 0]
        
        # 退货机构统计
        return_terms = return_records[self.term_code_col].unique()
        total_terms = df[self.term_code_col].unique()
        
        # 退货数量统计
        return_qty = return_records[self.qty_col].abs().sum()
        positive_qty = positive_records[self.qty_col].sum()
        
        # 按机构统计退货条数、退货数量
        term_return_stats = return_records.groupby(self.term_code_col).agg({
            self.qty_col: ['count', lambda x: x.abs().sum()]
        }).reset_index()
        term_return_stats.columns = [self.term_code_col, 'return_count', 'return_qty']
        
        return pd.DataFrame({
            '项目': ['退货条数', '总条数', '退货条数占比', 
                    '退货机构', '总机构', '退货机构占比', 
                    '退货数量', '进货数量', '退货数量占比'],
            '数值/百分比': [
                len(return_records),
                len(df),
                round(len(return_records) / len(df) * 100, 2) if len(df) > 0 else 0,
                len(return_terms),
                len(total_terms),
                round(len(return_terms) / len(total_terms)* 100, 2) if len(total_terms) > 0 else 0,
                round(return_qty, 2),
                round(positive_qty, 2),
                round(return_qty / positive_qty * 100, 2) if positive_qty > 0 else 0
            ]
        }), term_return_stats