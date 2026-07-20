"""
ClickHouse 数据库导入模块

提供从 ClickHouse 数据库导入流向数据的功能
"""

import pandas as pd
from typing import List
from sqlalchemy import create_engine

from src.utils.logger_utils import logger


# ==========================================
# SQL 模板
# ==========================================
SQL_TEMPLATE = '''
SELECT
    bizym,
    SUBSTRING(transdate, 1, 10) AS transdate,
    prodmdmcode,
    tomdmcode,
    toedivndrnm,
    tomdphncode,
    tomdphnname,
    frommdmtype1,
    frommdmtype2,
    frommdmtype3,
    tomdmtype1,
    tomdmtype2,
    tomdmtype3,
    tomdmprovince,
    tomdmcity,
    tomdmcounty,
    tohospitallevel,
    cnvrtdqty AS qty
FROM dm.dws_dg_ph_md_fact_sales_2022_2023
WHERE projectcode IN %(proj_code)s
AND prodmdmcode IN %(prod_code)s
AND frommdmtype1 IN %(from_type)s
AND {to_type_field} IN %(to_type)s
UNION ALL
SELECT
    bizym,
    SUBSTRING(transdate, 1, 10) AS transdate,
    prodmdmcode,
    tomdmcode,
    toedivndrnm,
    tomdphncode,
    tomdphnname,
    frommdmtype1,
    frommdmtype2,
    frommdmtype3,
    tomdmtype1,
    tomdmtype2,
    tomdmtype3,
    tomdmprovince,
    tomdmcity,
    tomdmcounty,
    tohospitallevel,
    cnvrtdqty AS qty
FROM dm.dws_dg_ph_md_fact_sales
WHERE projectcode IN %(proj_code)s
AND prodmdmcode IN %(prod_code)s
AND frommdmtype1 IN %(from_type)s
AND {to_type_field} IN %(to_type)s
UNION ALL
SELECT
    bizym,
    toString(transdate) AS transdate,
    prodmdmcode,
    tomdmcode,
    toedivndrnm,
    tomdphncode,
    tomdphnname,
    frommdmtype1,
    frommdmtype2,
    frommdmtype3,
    tomdmtype1,
    tomdmtype2,
    tomdmtype3,
    tomdmprovince,
    tomdmcity,
    tomdmcounty,
    tohospitallevel,
    cnvrtdqty AS qty
FROM {proj_sales_table}
WHERE prodmdmcode IN %(prod_code)s
AND frommdmtype1 IN %(from_type)s
AND {to_type_field} IN %(to_type)s
'''

# ==========================================
#  ClickHouse 数据库加载接口
# ==========================================
def load_from_clickhouse(
    engine: str,
    proj_code: List[str],
    prod_code: List[str],
    from_type: List[str],
    to_type: List[str],
    to_type_field: str,
    proj_sales_table: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    从 ClickHouse 数据库加载流向数据

    Args:
        engine: 数据库连接引擎
        proj_code: 项目编码列表
        prod_code: 产品编码列表
        from_type: 上游机构属性列表
        to_type: 下游机构属性列表
        to_type_field: 下游机构属性筛选字段，可选 'tomdmtype1' 或 'tomdmtype2'
        proj_sales_table: 项目月交付数据表名
        verbose: 是否打印日志

    Returns:
        DataFrame: 原始流向数据

    Example:
        >>> df = load_from_clickhouse(
        ...     engine='clickhouse+http://username:password@server:port/database?socket_timeout=600000',
        ...     proj_code=['P001'],
        ...     prod_code=['PRD001', 'PRD002'],
        ...     from_type=['经销商'],
        ...     to_type=['医院'],
        ...     to_type_field='tomdmtype2',
        ...     proj_sales_table='sarx_ads.ads_sarx_sales'
        ... )
    """
    # 验证必要参数
    required_cols = [proj_code, prod_code, from_type, to_type, to_type_field, proj_sales_table]
    if any(arg is None for arg in required_cols):
        raise ValueError(
            f"使用 ClickHouse 数据源时，必须提供: {required_cols}"
        )
    # 验证 to_type_field 参数
    valid_fields = ['tomdmtype1', 'tomdmtype2']
    if to_type_field not in valid_fields:
        raise ValueError(f"to_type_field 必须为 {valid_fields} 之一")

    # 动态生成 SQL
    sql_query = SQL_TEMPLATE.format(
        to_type_field=to_type_field,
        proj_sales_table=proj_sales_table
    )

    # SQL 查询参数
    params = {
        'proj_code': proj_code,
        'prod_code': prod_code,
        'from_type': from_type,
        'to_type': to_type
    }

    if verbose:
        logger.info(f"正在加载数据...")
        logger.info(f"  - 筛选条件: projectcode IN {proj_code}")
        logger.info(f"  - 筛选条件: prodmdmcode IN {prod_code}")
        logger.info(f"  - 筛选条件: frommdmtype1 IN {from_type}")
        logger.info(f"  - 筛选条件: to_type_field IN {to_type}")
        logger.info(f"  - 项目月交付数据表: {proj_sales_table}")
    
    # 执行 SQL 查询，返回 DataFrame
    df = pd.read_sql_query(sql_query, con=create_engine(engine), params=params)

    if verbose:
        logger.info(f"数据加载完成: {len(df)} 条记录")
    
    return df