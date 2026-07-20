"""
CSV 文件导入模块

提供从 CSV 文件导入流向数据的功能
"""

import pandas as pd
import os
from typing import Optional

from src.utils.config_utils import config
from src.utils.logger_utils import logger


def load_from_csv(
    file_path: str,
    parse_dates: bool,
    usecols: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    从 CSV 文件导入流向数据

    Args:
        file_path: CSV 文件路径
        parse_dates: 是否解析日期列
        usecols: 指定读取的列，默认读取所有列
        verbose: 是否打印日志

    Returns:
        DataFrame: 原始流向数据

    Example:
        >>> df = load_from_csv('data/sales_data.csv')
        >>> df = load_from_csv('data/sales_data.csv', usecols=['bizym', 'tomdmcode', 'qty'])
    """
    # 验证文件路径
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if verbose:
        logger.info(f"正在读取文件: {file_path}")

    # 读取 CSV 文件，返回 DataFrame
    df = pd.read_csv(
        file_path,
        usecols=usecols
    )

    # 解析日期列
    if parse_dates and config.get('columns.date') in df.columns:
        df[config.get('columns.date')] = pd.to_datetime(df[config.get('columns.date')])

    if verbose:
        logger.info(f"数据加载完成: {len(df)} 条记录")

    return df