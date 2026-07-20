"""
特征注册表模块

提供特征函数的元信息注册与查询功能，支持按分类、颗粒度、标签过滤
"""

import pandas as pd
from typing import Callable, Dict, List, Optional

from src.utils.logger_utils import logger


FEATURE_REGISTRY: Dict[str, dict] = {}


def register_feature(
    name: str,
    description: str,
    category: str,
    forecast_level: str,
    required_cols: List[str],
    output_cols: List[str],
    tags: List[str] = None,
):
    """
    特征注册装饰器

    Args:
        name: 特征唯一标识
        description: 特征含义描述
        category: 特征分类
        forecast_level: 特征颗粒度，可选: terminal | province | national
        required_cols: 该特征函数依赖的原始列
        output_cols: 该特征函数产出的列名列表
        tags: 自由标签列表

    Returns:
        Callable: 装饰后的函数（不修改原函数行为）
    """
    def decorator(func: Callable) -> Callable:
        FEATURE_REGISTRY[name] = {
            "func":           func,
            "description":    description,
            "category":       category,
            "forecast_level": forecast_level,
            "required_cols":  required_cols,
            "output_cols":    output_cols,
            "tags":           tags or [],
        }
        return func
    return decorator


def list_features(
    category: Optional[str] = None,
    forecast_level: Optional[str] = None,
    tags: Optional[List[str]] = None,
    keyword: Optional[str] = None,
) -> pd.DataFrame:
    """
    浏览特征目录，支持多条件过滤。

    Args:
        category: 按分类过滤
        forecast_level: 按颗粒度过滤
        tags: 按标签过滤（满足任意一个即命中）
        keyword: 按关键词模糊搜索名称或描述

    Returns:
        pd.DataFrame: 每行是一个特征的元信息摘要

    Examples:
        >>> list_features()                           # 查看全部特征
        >>> list_features(category="rfm")             # 仅看 RFM 类特征
        >>> list_features(forecast_level="terminal")  # 仅看终端颗粒度特征
        >>> list_features(tags=["rolling"])           # 含 rolling 标签的特征
        >>> list_features(keyword="增长")             # 关键词搜索
    """
    rows = []
    for name, meta in FEATURE_REGISTRY.items():
        if category and meta["category"] != category:
            continue
        if forecast_level and meta["forecast_level"] != forecast_level:
            continue
        if tags and not any(t in meta["tags"] for t in tags):
            continue
        if keyword and keyword not in name and keyword not in meta["description"]:
            continue
        rows.append({
            "name":           name,
            "description":    meta["description"],
            "category":       meta["category"],
            "forecast_level": meta["forecast_level"],
            "required_cols":  ", ".join(meta["required_cols"]),
            "output_cols":    ", ".join(meta["output_cols"]),
            "tags":           ", ".join(meta["tags"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("未找到符合条件的特征，请检查过滤条件。")
    return df


def get_feature_meta(name: str) -> dict:
    """获取单个特征的完整元信息"""
    if name not in FEATURE_REGISTRY:
        raise KeyError(f"特征 '{name}' 未注册，请通过 list_features() 查看可用特征。")
    return FEATURE_REGISTRY[name]