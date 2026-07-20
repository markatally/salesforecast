"""
数据处理工具
"""

from datetime import date, timedelta
import pandas as pd
from typing import List


def add_months(year_month: int, months: int) -> int:
    """
    根据年月与偏移月数，计算对应的年月

    Args:
        year_month: 整数类型的年月，格式 YYYYMM，如 202402 表示 2024年2月
        months: 偏移月数。正数表示向后（未来），负数表示向前（过去）

    Returns:
        添加偏移月数后的年月

    Examples:
        >>> add_months(202602, 3)
        202605
        >>> add_months(202602, -2)
        202512
    """
    year, month = year_month // 100, year_month % 100
    total_months = year * 12 + month - 1  # 转为从 0 起的月数
    new_total_months = total_months + months
    new_year = new_total_months // 12
    new_month = new_total_months % 12 + 1  # 恢复为 1–12 月
    return new_year * 100 + new_month

def get_first_day_of_month(year: int, month: int) -> date:
    """
    返回指定年月的第一天

    Args:
        year: 年份
        month: 月份

    Returns:
        date: 指定年月的第一天
    """
    return date(year, month, 1)

def get_last_day_of_month(year: int, month: int) -> date:
    """
    返回指定年月的最后一天

    Args:
        year: 年份
        month: 月份

    Returns:
        date: 指定年月的最后一天
    """
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return nxt - timedelta(days=1)

def get_month_diff(year_month_a: int, year_month_b: int) -> int:
    """
    计算两个年月之间相差的绝对月数

    Args:
        year_month_a: 年月a
        year_month_b: 年月b

    Returns:
        int: 年月a与年月b之间相差的绝对月数
    """
    year_a, month_a = int(year_month_a) // 100, int(year_month_a) % 100
    year_b, month_b = int(year_month_b) // 100, int(year_month_b) % 100
    return abs((year_b * 12 + month_b) - (year_a * 12 + month_a))

def generate_month_range(start_month: int, end_month: int) -> List[int]:
    """
    生成从开始年月到结束月的自然月列表

    Args:
        start_month: 开始年月
        end_month: 结束年月

    Returns:
        List[int]: 自然月列表
    """
    date_range = pd.date_range(
        start = pd.to_datetime(str(start_month), format="%Y%m"),
        end = pd.to_datetime(str(end_month), format="%Y%m"),
        freq = "MS",
    )
    month_range = [int(d.strftime("%Y%m")) for d in date_range]
    return month_range