"""
数据导入模块

支持从不同数据源导入流向数据
"""

import pandas as pd

from src.data.retrieve.clickhouse_loader import load_from_clickhouse
from src.data.retrieve.csv_loader import load_from_csv
from src.utils.config_utils import config
from src.utils.logger_utils import logger


class DataLoader:
    """数据导入接口"""
    
    def __init__(self):
        self.source = config.get('data_retrieve.source')
        # ClickHouse 参数
        self.engine = config.get('data_retrieve.clickhouse_url')
        self.proj_code = config.get('data_retrieve.proj_code')
        self.prod_code = config.get('data_retrieve.prod_code')
        self.from_type = config.get('data_retrieve.from_type')
        self.to_type = config.get('data_retrieve.to_type')
        self.to_type_field = config.get('data_retrieve.to_type_field')
        self.proj_sales_table = config.get('data_retrieve.proj_sales_table')
        # CSV 参数
        self.file_path = config.get('data_retrieve.raw_data_path')
        self.parse_dates = config.get('data_retrieve.parse_dates')
        self.usecols = config.get('data_retrieve.usecols')
        # 通用参数
        self.verbose = config.get('project.verbose')

    def load_data(self) -> pd.DataFrame:
        """
        统一数据加载接口

        Returns:
            DataFrame: 原始流向数据
        """

        # 从 Clickhouse 导入流向数据
        if self.source.lower() == 'clickhouse':
            if self.verbose:
                logger.info("导入模式: ClickHouse")

            df = load_from_clickhouse(
                engine=self.engine,
                proj_code=self.proj_code,
                prod_code=self.prod_code,
                from_type=self.from_type,
                to_type=self.to_type,
                to_type_field=self.to_type_field,
                proj_sales_table=self.proj_sales_table,
                verbose=self.verbose,
            )

        # 从 CSV 导入流向数据
        elif self.source.lower() == 'csv':
            if self.verbose:
                logger.info("导入模式: CSV")

            df = load_from_csv(
                file_path=self.file_path,
                parse_dates=self.parse_dates,
                usecols=self.usecols,
                verbose=self.verbose,
            )
        
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

        if self.verbose:
            logger.info(f"导入数据规模: {df.shape}")

        return df