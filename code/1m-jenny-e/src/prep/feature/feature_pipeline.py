"""
特征流水线模块

提供基于注册表的特征流水线构建功能，支持按名称列表快速组装任意特征组合

Usage:
    from src.prep.feature.feature_pipeline import FeaturePipeline
    from src.prep.feature.feature_engineering import TerminalFeatureEngineering

    tfe = TerminalFeatureEngineering(df_data)

    pipeline = FeaturePipeline(
        fe=tfe,
        feature_names=["lag_features", "rfm_features"],
    )
    feature_df = pipeline.build(matrix)
    X = feature_df[pipeline.get_output_cols()]
"""

import inspect

import pandas as pd
from typing import List

from src.prep.feature.feature_registry import FEATURE_REGISTRY, list_features
from src.utils.logger_utils import logger


class FeaturePipeline:
    """特征流水线构建器"""

    def __init__(self, fe, feature_names: List[str]):
        self.fe = fe
        self.feature_names = feature_names
        self._validate()

    def build(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        按注册顺序依次执行特征计算，返回附加了所有特征列的 DataFrame。

        Args:
            matrix: 已经过 _align_ym() 对齐的基础 matrix

        Returns:
            pd.DataFrame: 包含原始列 + 所有选取特征列
        """
        matrix = matrix.copy()
        for name in self.feature_names:
            meta = FEATURE_REGISTRY[name]
            func = meta["func"]
            # 过滤当前函数接受的额外参数
            # sig_params = set(inspect.signature(func).parameters.keys()) - {"self", "matrix"}
            # extra = {k: v for k, v in kwargs.items() if k in sig_params}
            # matrix = func(self.fe, matrix, **extra)
            matrix = func(self.fe, matrix)
            logger.info(f"[Pipeline] {name} 完成 → 产出列: {meta['output_cols']}")
        return matrix

    def get_output_cols(self) -> List[str]:
        """返回该 pipeline 所有特征产出的列名"""
        cols, seen = [], set()
        for name in self.feature_names:
            for col in FEATURE_REGISTRY[name]["output_cols"]:
                if col not in seen:
                    cols.append(col)
                    seen.add(col)
        return cols

    def summary(self) -> pd.DataFrame:
        """输出当前 pipeline 所含特征的元信息摘要"""
        return list_features()[
            list_features()["name"].isin(self.feature_names)
        ].reset_index(drop=True)

    def _validate(self):
        """校验特征名是否均已在注册表中"""
        missing = [n for n in self.feature_names if n not in FEATURE_REGISTRY]
        if missing:
            raise ValueError(
                f"以下特征未在注册表中找到：{missing}\n"
                f"请调用 list_features() 查看全部可用特征。"
            )