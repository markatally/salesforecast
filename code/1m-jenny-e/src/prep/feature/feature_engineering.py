"""
特征工程模块

提供终端级别进货预测所需的全部特征计算功能，每个特征方法均通过
@register_feature 注册元信息，支持被 FeaturePipeline 检索、组装和复用
"""

from datetime import date, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import chinese_calendar as calendar

from src.prep.feature.feature_registry import register_feature
from src.utils.config_utils import config
from src.utils.logger_utils import logger
from src.utils.data_utils import add_months, get_first_day_of_month, get_last_day_of_month, get_month_diff, generate_month_range


class TerminalFeatureEngineering:
    """终端级别进货预测特征工程"""

    def __init__(
        self,
        df_data: pd.DataFrame,
        start_ym: Optional[int] = None,
        end_ym: Optional[int] = None,
    ):
        """
        Args:
            df_data: 输入数据
            start_ym: 开始年月
            end_ym: 结束年月
        """
        # 字段名
        self.bizym_col = config.get('columns.bizym')
        self.transdate_col = config.get('columns.date')
        self.term_code_col = config.get('columns.term_code')
        self.qty_col = config.get('columns.qty')

        # 天级别数据（原始/清洗后数据）
        self.df_data = df_data
        # 月汇总数据（存在间隔月份）
        self.df_ym   = self.df_data.groupby([self.term_code_col, self.bizym_col])[[self.qty_col]].sum().reset_index()

        # 时间序列
        self.start_ym = start_ym if start_ym else self.df_data[self.bizym_col].min()
        self.end_ym = end_ym if end_ym else config.get('info.tar_ym')
        self.tar_mtd = config.get('info.tar_mtd')
        self.full_ym_list = generate_month_range(self.start_ym, self.end_ym)

        # 月补齐数据（补齐间隔月份、已按终端+年月排序）
        self.matrix = self._fill_terminal_month_gaps()

        # 业务输入
        self.jump_ym = config.get('prep.feature.jump_ym')
        self.drop_ym = config.get('prep.feature.drop_ym')
        
    # def get_prob_qty_ft(
    #     self,
    #     jump_ym,
    #     drop_ym,
    #     dur_prob: int = 12,
    #     dur_qty:  int = 6,
    # ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     按顺序计算所有时序特征，返回概率模型和数量模型的特征表。
    #     如需自定义特征组合，请改用 FeaturePipeline。

    #     Args:
    #         jump_ym: 需标记的异常高峰年月列表
    #         drop_ym: 需标记的异常低谷年月列表
    #         dur_prob: 概率模型时间窗口月数，默认 12
    #         dur_qty: 数量模型时间窗口月数，默认 6

    #     Returns:
    #         Tuple[pd.DataFrame, pd.DataFrame]: (df_prob, df_qty)
    #     """
    #     matrix = self._align_ym()
    #     logger.info("年月对齐完成")

    #     steps = [
    #         ("时间特征",   lambda m: self.calc_time_features(m, jump_ym, drop_ym)),
    #         ("滞后特征",   self.calc_lag_features),
    #         ("RFM特征",    self.calc_rfm_features),
    #         ("趋势特征",   self.calc_trend_features),
    #         ("进货月数",   self.calc_sellin_month_features),
    #         ("增长率",     self.calc_growth_features),
    #         ("增跌幅特征", lambda m: self.calc_yoy_growth_features(m, jump_ym, drop_ym)),
    #         ("MTD特征",    self.calc_mtd_features),
    #         ("进货间隔",   self.calc_gap_features),
    #         ("年月距离",   self.calc_ym_dist_features),
    #         ("距离月数",   self.calc_recency_month_features),
    #         ("正常消耗",   self.calc_base_consumption_features),
    #         ("库存压力",   self.calc_stock_ratio_features),
    #         ("库存代理",   self.calc_delta_stock_features),
    #     ]
    #     for name, fn in steps:
    #         matrix = fn(matrix)
    #         logger.info(f"{name}完成")

    #     matrix = matrix.rename({self.qty_col: "ttl_qty"}, axis=1)
    #     matrix["is_sellin"] = (matrix["ttl_qty"] > 0).astype(int)

    #     prob_cut_ym = add_months(self.tar_ym, -dur_prob)
    #     qty_cut_ym  = add_months(self.tar_ym, -dur_qty)
    #     logger.info(f"概率模型时间窗口：过去 {dur_prob} 个月（≥{prob_cut_ym}）")
    #     logger.info(f"数量模型时间窗口：过去 {dur_qty}  个月（≥{qty_cut_ym}）")

    #     prob_cols = [
    #         self.term_code_col, self.bizym_col,
    #         "month", "quarter",
    #         "is_jump_ym", "is_drop_ym",
    #         "ym_id", "num_sellin_ym", "num_wd",
    #         "freq6m", "freq3m", "freq6m_per_month", "freq3m_per_month", "recency",
    #         "has_trans_6m", "has_trans_3m",
    #         "DayDiffMean3m", "DayDiffMean6m", "DayDiffMean12m",
    #         "is_exceed_diff12m", "is_exceed_diff6m", "is_exceed_diff3m",
    #         "recency_months",
    #         "stock_ratio_3m", "stock_ratio_6m", "weighted_stock_ratio", "last_spike", "delta_stock_proxy",
    #         "mtd_qty", "is_sellin",
    #     ]
    #     qty_cols = [
    #         self.term_code_col, self.bizym_col,
    #         "month", "quarter",
    #         "is_jump_ym", "is_drop_ym",
    #         "ym_id", "num_sellin_ym", "num_wd",
    #         "qty_lag_1", "qty_lag_3", "qty_lag_6", "qty_lag_12",
    #         "mnt6m", "mnt3m", "mnt6m_per_month", "mnt3m_per_month",
    #         "freq6m", "freq3m", "freq6m_per_month", "freq3m_per_month",
    #         "mnt6m_per_trans", "mnt3m_per_trans", "mnt3m_div_6m", "recency",
    #         "delta_qty_6m", "delta_qty_3m", "has_trans_6m", "has_trans_3m",
    #         "growth_lag_1", "growth_lag_2", "growth_lag_3", "growth_lag_6", "growth_lag_12",
    #         "growth_jump", "growth_drop",
    #         "stock_ratio_3m", "stock_ratio_6m", "weighted_stock_ratio", "last_spike", "delta_stock_proxy",
    #         "mtd_qty", "ttl_qty", "bc_med",
    #     ]

    #     df_prob = matrix[matrix[self.bizym_col] >= prob_cut_ym][prob_cols]
    #     df_qty  = matrix[matrix[self.bizym_col] >= qty_cut_ym][qty_cols]
    #     return df_prob, df_qty

    @register_feature(
        name="time_features",
        description="基础时间特征",
        category="time",
        forecast_level="terminal",
        required_cols=["bizym_col"],
        output_cols=["year", "month", "quarter", "ym_id", "is_jump_ym", "is_drop_ym"],
        tags=["time", "flag"],
    )
    def calc_time_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """基础时间特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        # 年份、月份、季度
        matrix["year"]        = matrix[self.bizym_col].map(lambda x: int(str(x)[:4])).astype(np.int8)
        matrix["month"]       = matrix[self.bizym_col].map(lambda x: int(str(x)[4:])).astype(np.int8)
        matrix["quarter"]     = ((matrix["month"] - 1) // 3 + 1).astype(np.int8)
        # 年月顺序编号
        ym_idx_map = dict(zip(self.full_ym_list, range(len(self.full_ym_list))))
        matrix["ym_id"]       = matrix[self.bizym_col].map(ym_idx_map).astype(np.int8)
        # 是否显著涨幅/降幅月
        matrix["is_jump_ym"]  = matrix[self.bizym_col].isin(self.jump_ym).astype(np.int8)
        matrix["is_drop_ym"]  = matrix[self.bizym_col].isin(self.drop_ym).astype(np.int8)
        return matrix
    
    @register_feature(
        name="term_time_features",
        description="终端年月编号",
        category="time",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col"],
        output_cols=["term_ym_id"],
        tags=["time"],
    )
    def calc_term_time_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """终端年月编号"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix["term_ym_id"] = matrix.groupby(self.term_code_col)[self.bizym_col].transform(
            lambda x: list(range(1, len(x) + 1))
        )
        return matrix

    @register_feature(
        name="lag_features",
        description="历史滞后进货量特征",
        category="lag",
        forecast_level="terminal",
        required_cols=["term_code_col", "qty_col", "ym_id"],
        output_cols=["qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_6", "qty_lag_12"],
        tags=["lag"],
    )
    def calc_lag_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """历史滞后进货量特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix = self._calc_multi_lags(matrix, [1, 2, 3, 6, 12], self.qty_col)
        return matrix

    @register_feature(
        name="rfm_features",
        description="RFM 特征：上次进货距离天数、近 3/6 月总计/平均交易笔数、近 3/6 月总计/平均进货量",
        category="rfm",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "transdate_col", "qty_col"],
        output_cols=[
            "recency_days",
            "order_count_6m", "order_count_3m", 
            "avg_monthly_order_count_3m", "avg_monthly_order_count_3m",
            "qty_6m", "qty_3m", 
            "avg_monthly_qty_6m", "avg_monthly_qty_3m",
            "avg_order_qty_6m", "avg_order_qty_3m",
        ],
        tags=["rfm", "recency", "frequency", "rolling"],
    )
    def calc_rfm_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """RFM 特征：上次进货距离天数、近 3/6 月总计/平均交易笔数、近 3/6 月总计/平均进货量"""
        # 创建副本避免修改原始数据框
        df_rfm = self.df_data.copy()
        matrix = matrix.copy()

        # ===========================================================   
        # 上次进货距离天数、近 3/6 月总计交易笔数、近 3/6 月总计进货量
        # ================================================================
        # 当月最后一次进货时间、当月交易笔数
        df_rfm = (
            df_rfm.groupby([self.term_code_col, self.bizym_col])
            .agg({self.transdate_col: "last", self.qty_col: "count"})
            .reset_index()
            .rename({self.transdate_col: "last_sellin_date", self.qty_col: "order_count"}, axis=1)
        )
        matrix = matrix.merge(df_rfm, how="left", on=[self.term_code_col, self.bizym_col])

        # 填充缺失值：当月最后一次进货时间前向填充（若当月无进货，则使用上月最后一次进货时间）
        matrix["last_sellin_date"] = matrix.groupby(self.term_code_col)["last_sellin_date"].transform(
            lambda x: x.ffill()
        )
        # 填充缺失值：当月交易笔数填充0
        matrix["order_count"] = matrix["order_count"].fillna(0)

        # 计算上次进货距离天数、近 3/6 月总计交易笔数、近 3/6 月总计进货量
        matrix = matrix.groupby(self.term_code_col).apply(self._calc_rfm_by_group).reset_index(drop=True)
        matrix["current_date"] = matrix[self.bizym_col].apply(
            lambda x: get_last_day_of_month(int(str(x)[:4]), int(str(x)[4:]))
        )
        matrix["recency_days"] = (
            pd.to_datetime(matrix["current_date"]) - matrix["last_sellin_date"]
        ).dt.days

        # =======================================
        # 近 3/6 月平均交易笔数、近 3/6 月平均进货量
        # =======================================
        # 平均月交易笔数、平均月进货量、平均单笔进货量
        matrix["avg_monthly_order_count_6m"] = round(matrix["order_count_6m"] / 6, 2)
        matrix["avg_monthly_order_count_3m"] = round(matrix["order_count_3m"] / 3, 2)
        matrix["avg_monthly_qty_6m"] = round(matrix["qty_6m"] / 6, 2)
        matrix["avg_monthly_qty_3m"] = round(matrix["qty_3m"] / 3, 2)
        matrix["avg_order_qty_6m"] = round(matrix["qty_6m"] / matrix["order_count_6m"], 2)
        matrix["avg_order_qty_3m"] = round(matrix["qty_3m"] / matrix["order_count_3m"], 2)
        
        # 处理除0的情形
        matrix.loc[matrix["order_count_6m"] == 0, "avg_order_qty_6m"] = 0
        matrix.loc[matrix["order_count_3m"] == 0, "avg_order_qty_3m"] = 0

        # 移除中间列
        drop_cols = [self.qty_col, "order_count", "last_sellin_date", "current_date"]
        matrix = matrix.drop(columns=drop_cols)
        return matrix

    @register_feature(
        name="recency_months_features",
        description="RFM 特征：上次进货距离月数",
        category="rfm",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "qty_col"],
        output_cols=["recency_months"],
        tags=["rfm", "recency"],
    )
    def calc_recency_months(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """RFM 特征：上次进货距离月数"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix["is_sellin"]    = matrix[self.qty_col] > 0
        matrix["sellin_ym_lag"] = (
            matrix.groupby(self.term_code_col)[self.bizym_col]
            .shift(1)
            .where(matrix.groupby(self.term_code_col)["is_sellin"].shift(1))
        )
        matrix["last_sellin_ym"] = matrix.groupby(self.term_code_col)["sellin_ym_lag"].ffill()
        matrix["recency_months"] = matrix.apply(
            lambda x: get_month_diff(x["last_sellin_ym"], x[self.bizym_col])
            if not np.isnan(x["last_sellin_ym"]) else np.nan,
            axis=1,
        )
        return matrix

    @register_feature(
        name="active_months_features",
        description="RFM 特征：近 3/6 月产生交易月数",
        category="rfm",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "qty_col"],
        output_cols=["active_months_3m", "active_months_6m"],
        tags=["rfm", "frequency", "rolling"],
    )
    def calc_active_months(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """RFM 特征：近 3/6 月产生交易月数"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix["active_months_3m"] = (
            (matrix[self.qty_col] != 0).astype(int)
            .groupby(matrix[self.term_code_col])
            .rolling(window=3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        matrix["active_months_3m"] = matrix.groupby(self.term_code_col)["active_months_3m"].transform(lambda x: x.shift(1))

        matrix["active_months_6m"] = (
            (matrix[self.qty_col] != 0).astype(int)
            .groupby(matrix[self.term_code_col])
            .rolling(window=6, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        matrix["active_months_6m"] = matrix.groupby(self.term_code_col)["active_months_6m"].transform(lambda x: x.shift(1))
        return matrix
    
    @register_feature(
        name="interval_features",
        description="进货间隔特征：近 3/6/12 月平均进货间隔天数、上次进货距离天数是否超过平均水平",
        category="interval",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "transdate_col", "qty_col", "recency_days"],
        output_cols=[
            "avg_interval_3m", "avg_interval_6m", "avg_interval_12m",
            "is_exceed_interval_3m", "is_exceed_interval_6m", "is_exceed_interval_12m",
        ],
        tags=["interval", "rolling", "flag"],
    )
    def calc_interval_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """进货间隔特征"""
        # 创建副本避免修改原始数据框
        df_diff = self.df_data.copy()
        matrix = matrix.copy()

        # 与上笔交易间隔天数
        df_diff["last_sellin_date"] = df_diff.groupby(self.term_code_col)[self.transdate_col].shift(1)
        df_diff["interval_days"] = (df_diff[self.transdate_col] - df_diff["last_sellin_date"]).dt.days
        df_diff = df_diff.dropna(subset=["interval_days"])

        # 近 3/6/12 月平均进货间隔天数
        for window, col_name in [(3, "avg_interval_3m"), (6, "avg_interval_6m"), (12, "avg_interval_12m")]:
            rows = []
            for i in range(window, len(self.full_ym_list)):
                ym_start  = self.full_ym_list[i - window]
                ym_end    = self.full_ym_list[i - 1]
                ym_target = self.full_ym_list[i]
                df_slice  = df_diff[(df_diff[self.bizym_col] >= ym_start) & (df_diff[self.bizym_col] <= ym_end)]
                df_term   = df_slice.groupby(self.term_code_col).agg({"interval_days": "mean"}).reset_index()
                df_term.columns = [self.term_code_col, col_name]
                df_term[self.bizym_col] = ym_target
                rows.append(df_term)
            matrix = matrix.merge(pd.concat(rows), how="left", on=[self.term_code_col, self.bizym_col])

        # 上次进货距离天数是否超过平均水平
        matrix["is_exceed_interval_12m"] = (matrix["recency_days"] > matrix["avg_interval_12m"]).astype(int)
        matrix["is_exceed_interval_6m"]  = (matrix["recency_days"] > matrix["avg_interval_6m"]).astype(int)
        matrix["is_exceed_interval_3m"]  = (matrix["recency_days"] > matrix["avg_interval_3m"]).astype(int)
        return matrix

    @register_feature(
        name="avg_trend_features",
        description="趋势特征：上月相对近 3/6 月平均水平的变化率、近 3 月相对近 6 月平均水平的变化率",
        category="trend",
        forecast_level="terminal",
        required_cols=["qty_lag_1", "avg_monthly_qty_6m", "avg_monthly_qty_3m"],
        output_cols=["chg_rate_last_vs_avg3m", "chg_rate_last_vs_avg6m", "chg_rate_avg3m_vs_avg6m"],
        tags=["trend", "rolling"],
    )
    def calc_avg_trend_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """趋势特征：上月相对近 3/6 月平均水平的变化率、近 3 月相对近 6 月平均水平的变化率"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix = self._calc_change_rate_between_fts(matrix, "qty_lag_1", "avg_monthly_qty_6m", "chg_rate_last_vs_avg3m")
        matrix = self._calc_change_rate_between_fts(matrix, "qty_lag_1", "avg_monthly_qty_3m", "chg_rate_last_vs_avg6m")
        matrix = self._calc_change_rate_between_fts(matrix, "avg_monthly_qty_3m", "avg_monthly_qty_6m", "chg_rate_avg3m_vs_avg6m")
        return matrix

    @register_feature(
        name="lag_trend_features",
        description="趋势特征：相对历史滞后进货量的变化率",
        category="trend",
        forecast_level="terminal",
        required_cols=["term_code_col", "qty_col", "qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_6", "qty_lag_12"],
        output_cols=["chg_rate_lag_1", "chg_rate_lag_2", "chg_rate_lag_3", "chg_rate_lag_6", "chg_rate_lag_12"],
        tags=["trend", "lag"],
    )
    def calc_lag_trend_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """趋势特征：相对历史滞后进货量的变化率"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        for lag in [1, 2, 3, 6, 12]:
            matrix = self._calc_single_lag_trend(matrix, lag)
        return matrix

    @register_feature(
        name="mom_trend_features",
        description="趋势特征：显著涨跌月份去年同期的环比变化率",
        category="trend",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "qty_col"],
        output_cols=["mom_jump_lag12", "mom_drop_lag12"],
        tags=["trend", "mom"],
    )
    def calc_mom_trend_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """趋势特征：显著涨跌月份去年同期的环比变化率"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        df_jump = self._calc_single_mom_trend(matrix, self.jump_ym)
        df_drop = self._calc_single_mom_trend(matrix, self.drop_ym)

        matrix = pd.merge(matrix, df_jump, how="left", left_on=[self.term_code_col, self.bizym_col], right_on=[self.term_code_col, "tar_ym"])
        matrix["mom_jump_lag12"] = matrix["mom_lag12"].fillna(0)
        matrix = matrix.drop(columns=["tar_ym", "mom_lag12"])

        matrix = pd.merge(matrix, df_drop, how="left", left_on=[self.term_code_col, self.bizym_col], right_on=[self.term_code_col, "tar_ym"])
        matrix["mom_drop_lag12"] = matrix["mom_lag12"].fillna(0)
        matrix = matrix.drop(columns=["tar_ym", "mom_lag12"])
        return matrix

    @register_feature(
        name="mtd_features",
        description="MTD 特征：截至第 k 个工作日的当月累计进货量、当月工作日数量",
        category="mtd",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "transdate_col", "qty_col"],
        output_cols=["mtd_qty", "wd_count"],
        tags=["mtd", "workday"],
    )
    def calc_mtd_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """MTD 特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        # 生成日期范围
        df_list = []
        for bizym in self.full_ym_list:
            y, m = int(str(bizym)[:4]), int(str(bizym)[4:])
            date_range = pd.date_range(
                start=get_first_day_of_month(y, m),
                end=get_last_day_of_month(y, m),
            ).tolist()
            df_list.append(pd.DataFrame({self.bizym_col: [bizym] * len(date_range), self.transdate_col: date_range}))

        # 标记工作日
        df_days = pd.concat(df_list)
        df_days["is_workday"] = df_days[self.transdate_col].apply(lambda x: 1 if calendar.is_workday(x) else 0)
        df_days["wd_id"] = (df_days.groupby(self.bizym_col)["is_workday"].cumsum().values) * df_days["is_workday"]

        # 每月工作日数量、每月第 k 个工作日日期
        df_ym_wd   = df_days.groupby(self.bizym_col)["is_workday"].sum().reset_index().rename({"is_workday": "wd_count"}, axis=1)
        df_wd_date = df_days[df_days["wd_id"] == self.tar_mtd].drop(columns=["is_workday", "wd_id"]).rename({self.transdate_col: "wd_date"}, axis=1)
        df_ym_wd["wd_date"] = df_wd_date["wd_date"].values

        # 每月MTD进货量
        df_mtd = pd.merge(self.df_data[[self.term_code_col, self.bizym_col, self.transdate_col, self.qty_col]], df_ym_wd, how="left", on=self.bizym_col)
        df_mtd = df_mtd[df_mtd[self.transdate_col] <= df_mtd["wd_date"]]
        df_mtd = (
            df_mtd.groupby([self.term_code_col, self.bizym_col])[self.qty_col]
            .sum()
            .reset_index()
            .rename({self.qty_col: "mtd_qty"}, axis=1)
        )

        # 合并MTD进货量、每月工作日数量
        matrix = matrix.merge(df_mtd, how="left", on=[self.term_code_col, self.bizym_col]) 
        matrix["mtd_qty"] = matrix["mtd_qty"].fillna(0)
        matrix = matrix.merge(df_ym_wd[[self.bizym_col, "wd_count"]], how="left", on=self.bizym_col)
        return matrix

    @register_feature(
        name="estimated_consumption_features",
        description="估计消耗量特征：近 12 个月中位数进货量、平均进货量",
        category="stock",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "qty_col"],
        output_cols=["ec_med", "ec_mean"],
        tags=["stock", "rolling"],
    )
    def calc_estimated_consumption_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """估计消耗量特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        # 按终端分组计算滚动 12 月进货量的中位数和均值
        df_ec = (
            matrix.groupby(self.term_code_col)[[self.term_code_col, self.bizym_col, self.qty_col]]
            .apply(self._calc_estimated_consumption_by_group)
            .reset_index(drop=True)
        )
        matrix = matrix.merge(df_ec[[self.term_code_col, self.bizym_col, "ec_med"]], how="left", on=[self.term_code_col, self.bizym_col])
        return matrix

    @register_feature(
        name="stock_stress_features",
        description="库存压力特征：近 1/3/6 月进货量相对估计消耗量的比率、近 3 月加权比率",
        category="stock",
        forecast_level="terminal",
        required_cols=["qty_3m", "qty_6m", "qty_lag_1", "qty_lag_2", "qty_lag_3", "ec_med"],
        output_cols=["ratio_qty3m_vs_ec", "ratio_qty6m_vs_ec", "ratio_last_vs_ec", "weighted_ratio_qty_vs_ec"],
        tags=["stock", "ratio"],
    )
    def calc_stock_stress_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """库存压力特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix["ratio_qty3m_vs_ec"]  = matrix["qty_3m"]    / (matrix["ec_med"] * 3 + 1)
        matrix["ratio_qty6m_vs_ec"]  = matrix["qty_6m"]    / (matrix["ec_med"] * 6 + 1)
        matrix["ratio_last_vs_ec"]   = matrix["qty_lag_1"] / (matrix["ec_med"] + 1)
        matrix["weighted_ratio_qty_vs_ec"] = (
            0.5 * matrix["qty_lag_1"] + 0.3 * matrix["qty_lag_2"] + 0.2 * matrix["qty_lag_3"]
        ) / (matrix["ec_med"] + 1)
        return matrix

    @register_feature(
        name="stock_proxy_features",
        description="库存代理特征：模拟当月期初库存（上月期初库存+上月进货量-估计消耗量）",
        category="stock",
        forecast_level="terminal",
        required_cols=["term_code_col", "bizym_col", "qty_col", "ec_med"],
        output_cols=["stock_proxy"],
        tags=["stock"],
    )
    def calc_stock_proxy_features(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """库存代理特征"""
        # 创建副本避免修改原始数据框
        matrix = matrix.copy()

        matrix["stock_proxy"] = np.nan
        for _, group in matrix.groupby(self.term_code_col):
            group = group[(group["ec_med"].notna())]
            if len(group) > 0:
                group["stock_proxy"] = 0.0
                group.iloc[0, -1] = group.iloc[0][self.qty_col] - group.iloc[0]["ec_med"]
                for i in range(1, len(group)):
                    group.iloc[i, -1] = (
                        group.iloc[i - 1]["stock_proxy"]
                        + group.iloc[i][self.qty_col]
                        - group.iloc[i]["ec_med"]
                    )
                matrix.loc[group.index, "stock_proxy"] = group["stock_proxy"]

        matrix["stock_proxy"] = matrix.groupby(self.term_code_col)["stock_proxy"].transform(lambda x: x.shift(1))
        return matrix

    # ==================================================
    # 辅助方法
    # ==================================================
    def _fill_terminal_month_gaps(self) -> pd.DataFrame:
        """按终端分组补齐间隔月份"""
        # 获取每个终端的首个年月
        term_first_ym = self.df_ym.groupby(self.term_code_col)[self.bizym_col].first().reset_index()
        # 按终端补齐年月序列，间隔月数量补0
        trans_ym_list, term_code_list = [], []
        for i, c in enumerate(term_first_ym[self.term_code_col].tolist()):
            ym_range = generate_month_range(term_first_ym.loc[i, self.bizym_col], self.end_ym)
            trans_ym_list.extend(ym_range)
            term_code_list.extend([c] * len(ym_range))
        matrix = pd.DataFrame({self.term_code_col: term_code_list, self.bizym_col: trans_ym_list})
        matrix = pd.merge(matrix, self.df_ym[[self.term_code_col, self.bizym_col, self.qty_col]], on=[self.term_code_col, self.bizym_col], how="left")
        matrix[self.qty_col] = matrix[self.qty_col].fillna(0)
        # 按终端、年月排序，便于后续计算
        matrix = matrix.sort_values([self.term_code_col, self.bizym_col])
        return matrix

    def _calc_multi_lags(self, matrix: pd.DataFrame, lags: List[int], col: str) -> pd.DataFrame:
        """计算多个滞后特征列"""
        tmp = matrix[["ym_id", self.term_code_col, col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["ym_id", self.term_code_col, f"{col}_lag_{i}"]
            shifted["ym_id"] += i
            matrix = pd.merge(matrix, shifted, on=["ym_id", self.term_code_col], how="left")
        return matrix

    def _calc_rfm_by_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """按终端分组计算 RFM 基础量"""
        group["last_sellin_date"] = group["last_sellin_date"].shift(1)
        group["order_count_6m"] = group["order_count"].rolling(window=6, min_periods=1).sum().shift(1)
        group["order_count_3m"] = group["order_count"].rolling(window=3, min_periods=1).sum().shift(1)
        group["qty_6m"] = group[self.qty_col].rolling(window=6, min_periods=1).sum().shift(1)
        group["qty_3m"] = group[self.qty_col].rolling(window=3, min_periods=1).sum().shift(1)
        return group
    
    def _calc_change_rate_between_fts(self, matrix: pd.DataFrame, numer: str, denom: str, ft_name: str) -> pd.DataFrame:
        """计算两个特征列之间的变化率"""
        conditions = [
            (matrix[denom] == 0) & (matrix[numer] > 0),   # 特殊情况1：分母为0、分子大于0，变化率视为100%
            (matrix[denom] == 0) & (matrix[numer] == 0),  # 特殊情况2：分母为0、分子为0，变化率视为0
            (matrix[denom] > 0),
        ]
        choices = [1, 0, (matrix[numer] - matrix[denom]) / matrix[denom]]
        matrix[ft_name] = np.select(conditions, choices, default=np.nan)
        return matrix

    def _calc_single_lag_trend(self, matrix: pd.DataFrame, lag_num: int) -> pd.DataFrame:
        """计算相对单个 lag 的变化率"""
        lag_col = f"qty_lag_{lag_num}"
        out_col = f"chg_rate_lag_{lag_num}"
        conditions = [
            (matrix[lag_col] == 0) & (matrix[self.qty_col] > 0),  # 特殊情况1：分母为0、分子大于0，变化率视为100%
            (matrix[lag_col] == 0) & (matrix[self.qty_col] == 0), # 特殊情况2：分母为0、分子为0，变化率视为0
            (matrix[lag_col] > 0),
        ]
        choices = [1, 0, (matrix[self.qty_col] - matrix[lag_col]) / matrix[lag_col]]
        matrix[out_col] = np.select(conditions, choices, default=np.nan)
        matrix[out_col] = matrix.groupby(self.term_code_col)[out_col].transform(lambda x: x.shift(1))
        return matrix
    
    def _calc_single_mom_trend(self, matrix: pd.DataFrame, tar_ym_list: List[int]) -> pd.DataFrame:
        """计算指定年月去年同期的环比变化率"""
        pivot_df = matrix.pivot_table(index=self.term_code_col, columns=self.bizym_col, values=self.qty_col, aggfunc="sum").fillna(0)
        df_list = []
        for tar_ym in tar_ym_list:
            ym_lag_12 = add_months(tar_ym, -12)
            ym_lag_13 = add_months(tar_ym, -13)
            if ym_lag_12 in pivot_df.columns and ym_lag_13 in pivot_df.columns:
                df_tmp = pd.DataFrame({
                    self.term_code_col: pivot_df.index,
                    "tar_ym": tar_ym,
                    "mom_lag12": ((pivot_df[ym_lag_12] - pivot_df[ym_lag_13]) / pivot_df[ym_lag_13] * 100).values, # 去年同期的环比变化率
                }).replace([np.nan, np.inf, -np.inf], 0)
                df_list.append(df_tmp)
        return pd.concat(df_list)

    def _calc_estimated_consumption_by_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """按终端分组计算滚动 12 月进货量的中位数和均值"""
        group = group.sort_values(self.bizym_col)
        group["ec_med"]  = group[self.qty_col].rolling(window=12, min_periods=3).median().shift(1)
        group["ec_mean"] = group[self.qty_col].rolling(window=12, min_periods=3).mean().shift(1)
        return group