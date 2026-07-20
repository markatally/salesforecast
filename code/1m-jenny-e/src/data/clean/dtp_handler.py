"""
DTP药房挂靠模块

提供将DTP药房的进货量挂靠至医疗机构的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List

from src.utils.config_utils import config
from src.utils.logger_utils import logger


class DTPHandler:
    """DTP药房挂靠处理器"""

    def __init__(self):
        self.verbose = config.get('project.verbose')
        self.bizym_col = config.get('columns.bizym')
        self.from_code_col = config.get('columns.from_code')
        self.term_code_col = config.get('columns.term_code')
        self.qty_col = config.get('columns.qty')

    def handle(
        self,
        df: pd.DataFrame,
        from_type_col: Optional[str] = config.get('data_clean.from_type_field'),
        term_type_col: Optional[str] = config.get('data_clean.term_type_field'),
        dtp_type_value: Optional[str] = config.get('data_clean.dtp_type_value'),
        hsp_type_value: Optional[str] = config.get('data_clean.hsp_type_value'),
    ) -> pd.DataFrame:
        """
        将DTP药房的进货量挂靠至医疗机构
        
        Args:
            df: 输入数据（经销商&零售 -> 医疗、经销商 -> 零售）
            from_type_col: 上游机构属性列名
            term_type_col: 下游机构属性列名
            dtp_type_value: DTP药房属性值（例如，零售）
            hsp_type_value: 医疗机构属性值（例如，医疗机构）

        Returns:
            pd.DataFrame: 处理后的数据
        """
        if self.verbose:
            logger.info(f"开始执行DTP挂靠逻辑，数据规模: {df.shape}")
        
        # Step 1: 提取需要挂靠的流向记录（零售 -> 医疗）
        df_dtp2hsp = df[df[from_type_col] == dtp_type_value].copy()
        
        # Step 2: By终端计算进货量
        df_left = df.groupby([self.bizym_col, self.term_code_col, term_type_col])[self.qty_col].sum().reset_index()
        
        # Step 3: ByDTP药房计算需要挂到医疗机构的数量
        df_dtp = df_dtp2hsp.groupby([self.bizym_col, self.from_code_col])[self.qty_col].sum().reset_index().rename({self.from_code_col: self.term_code_col, self.qty_col: 'dtp_qty'}, axis=1)
        df_dtp[term_type_col] = dtp_type_value

        # Step 4: By医疗机构计算来自DTP药房的数量
        df_hsp = df_dtp2hsp.groupby([self.bizym_col, self.term_code_col])[self.qty_col].sum().reset_index().rename({self.qty_col: 'af_qty'}, axis=1)
        df_hsp[term_type_col] = hsp_type_value

        # Step 5: 合并计算最终进货量
        df_right = pd.concat([df_dtp, df_hsp], ignore_index=True)
        df_result = df_left.merge(
            df_right, 
            on=[self.bizym_col, self.term_code_col, term_type_col],
            how='outer'  # 若某月存在DTP->医疗的关系，但无经销商->DTP的进货流向，此时添加该DTP药房的0进货流向
        )

        # 填充缺失值
        df_result[['qty', 'dtp_qty', 'af_qty']] = df_result[['qty', 'dtp_qty', 'af_qty']].fillna(0)
        
        # 保留原始数量
        df_result['raw_qty'] = df_result[self.qty_col]
        
        # 对于DTP药房，减去需要挂到医疗机构的数量
        mask = df_result['dtp_qty'] != 0
        df_result.loc[mask, self.qty_col] = df_result.loc[mask, self.qty_col] - df_result.loc[mask, 'dtp_qty']
        
        # 计算来自经销商的数量
        df_result['dist_qty'] = df_result['raw_qty'] - df_result['af_qty']
        
        # Step 6: 整理输出列
        output_cols = [self.bizym_col, self.term_code_col, term_type_col, 'raw_qty', 'dtp_qty', 'af_qty', 'dist_qty', self.qty_col]
        df_result = df_result[output_cols]
        
        if self.verbose:
            logger.info(f"DTP挂靠处理完成，数据规模: {df_result.shape}")
        
        return df_result