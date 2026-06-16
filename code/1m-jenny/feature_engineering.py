import pandas as pd
import numpy as np
import chinese_calendar as calendar
from datetime import timedelta, date
import logging

class FeatureEngineering:
    def __init__(self, df_data, TAR_YM, MTD, cutoff_ym=202301):
        self.df_data = df_data
        self.TAR_YM = TAR_YM
        self.MTD = MTD

        self.logger = self._setup_logger()

        self.START_YM = cutoff_ym 
        # self.START_YM = self._get_previous_month(self.TAR_YM, 24) # 历史2年的数据
        self.df_ym = self.df_data.groupby(['tomdmcode','bizym'])[['qty']].sum().reset_index()
        self.full_ym_list = self._generate_business_months(self.START_YM, self.TAR_YM)

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('FE')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s | %(message)s'
            )
            # 输出到控制台
            console_handler = logging.StreamHandler() 
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler )
            # 输出到文件
            # file_handler = logging.FileHandler('run.log', mode="a")
            # file_handler.setFormatter(formatter)
            # logger.addHandler(file_handler)
        
        return logger
    
    def get_hos_attr(self, df_region_path='../data/地域属性.xlsx') -> pd.DataFrame:
        """获取医院属性"""
        # 1、获取所有医院清单
        df_hospitals = self.df_data[['tomdmcode','tomdmprovince','tomdmcity','tomdmcounty','tohospitallevel']].drop_duplicates(subset=['tomdmcode'], keep='last')
        # 2、读取地域属性表
        df_region = pd.read_excel(df_region_path)
        # 3、两表拼接
        df_hos_ft = pd.merge(
            df_hospitals, 
            df_region, 
            how='left', 
            left_on=['tomdmprovince','tomdmcity','tomdmcounty'],
            right_on=['province','city','county']).drop(columns=['province','city','county','countytype']
        )
        # 4、LabelEncoding
        prov_map = dict(zip(df_hos_ft['tomdmprovince'].sort_values().unique(), list(range(1, 32))))
        hos_level_map = dict(zip(['未核实到','未评级','一级','二级','三级'], [0,1,2,3,4]))
        city_type_map = dict(zip(['县域市场', '城市市场'], [0,1]))
        city_level_map = dict(zip(['T5','T4','T3','T2','NewT1','T1'], [1,2,3,4,5,6]))
        df_hos_ft['hos_id'] = list(range(1,len(df_hos_ft)+1))
        df_hos_ft['prov_id'] = df_hos_ft['tomdmprovince'].apply(lambda x: prov_map[x])
        df_hos_ft["hos_level"] = df_hos_ft['tohospitallevel'].map(hos_level_map)
        df_hos_ft["city_type"] = df_hos_ft['city_type'].map(city_type_map)
        df_hos_ft["city_level"] = df_hos_ft['city_level'].map(city_level_map)
        return df_hos_ft
    
    def get_prob_qty_ft(
            self, 
            jump_ym, #TODO
            drop_ym, #TODO
            dur_prob=12, 
            dur_qty=6
    ) -> pd.DataFrame:
        """构造特征"""
        # 特征工程
        matrix = self._align_ym()
        self.logger.info("年月对齐完成")
        matrix = self._get_time_ft(matrix, jump_ym, drop_ym)
        self.logger.info("时间特征完成")
        matrix = self._get_lag_ft(matrix)
        self.logger.info("滞后特征完成")
        matrix = self._get_rfm_ft(matrix)
        self.logger.info("RFM特征完成")
        matrix = self._get_trend_ft(matrix)
        self.logger.info("趋势特征完成")
        matrix = self._get_sellin_ym(matrix)
        self.logger.info("进货月数完成")
        matrix = self._get_growth_ft(matrix)
        self.logger.info("增长率完成")
        matrix = self._calculate_growth_yoy(matrix, jump_ym, drop_ym)
        self.logger.info("增/跌幅特征完成")
        matrix = self._get_mtd_qty(matrix)
        self.logger.info("MTD特征完成")
        matrix = self._get_gap_ft(matrix)
        self.logger.info("进货间隔特征完成")
        matrix = self._get_ym_dist(matrix)
        self.logger.info("年月距离完成")
        matrix = self._get_recency_months(matrix)
        self.logger.info("距离月数完成")
        matrix = self._get_base_consumption(matrix)
        self.logger.info("正常消耗完成")
        matrix = self._get_stock_ratios(matrix)
        self.logger.info("库存压力完成")
        matrix = self._get_delta_stock(matrix)
        self.logger.info("库存代理完成")
        # Label
        matrix = matrix.rename({'qty':'ttl_qty'}, axis=1)
        matrix['is_sellin'] = matrix['ttl_qty'].apply(lambda x: 1 if x>0 else 0)
        # 训练数据时间窗口
        prob_cut_ym = self._get_previous_month(self.TAR_YM, dur_prob)
        qty_cut_ym = self._get_previous_month(self.TAR_YM, dur_qty)
        self.logger.info(f"概率模型时间窗口：过去{dur_prob}个月")
        self.logger.info(f"数量模型时间窗口：过去{dur_qty}个月")
        # 进货概率模型
        df_prob = matrix = matrix[matrix['bizym']>=prob_cut_ym]
        df_prob = df_prob[[
            'tomdmcode', 'bizym',
            'month', 'quarter', 
            #'is_qtr_begin', 'is_qtr_end', 'is_year_end', 
            'is_jump_ym', 'is_drop_ym',
            'ym_num', 'num_sellin_ym', 'num_wd',
            'freq6m', 'freq3m', 'freq6m_per_month', 'freq3m_per_month', 'recency',
            'has_trans_6m', 'has_trans_3m',
            'DayDiffMean3m', 'DayDiffMean6m', 'DayDiffMean12m',
            'is_exceed_diff12m', 'is_exceed_diff6m', 'is_exceed_diff3m',
            # New
            'recency_months',
            'stock_ratio_3m', 'stock_ratio_6m', 'weighted_stock_ratio', 'last_spike', 'delta_stock_proxy',
            'mtd_qty', 'is_sellin'
        ]]
        # 进货数量模型
        df_qty = matrix[matrix['bizym']>=qty_cut_ym]
        df_qty = df_qty[[
            'tomdmcode', 'bizym',
            'month', 'quarter', 
            #'is_qtr_begin', 'is_qtr_end', 'is_year_end', 
            'is_jump_ym', 'is_drop_ym',
            'ym_num', 'num_sellin_ym', 'num_wd',
            'qty_lag_1','qty_lag_3','qty_lag_6','qty_lag_12',
            'mnt6m', 'mnt3m', 'mnt6m_per_month', 'mnt3m_per_month',
            'freq6m', 'freq3m', 'freq6m_per_month', 'freq3m_per_month',
            'mnt6m_per_trans', 'mnt3m_per_trans', 'mnt3m_div_6m', 'recency',
            'delta_qty_6m', 'delta_qty_3m', 'has_trans_6m', 'has_trans_3m',
            'growth_lag_1', 'growth_lag_2', 'growth_lag_3', 'growth_lag_6', 'growth_lag_12',
            'growth_jump', 'growth_drop',
            # New
            'stock_ratio_3m', 'stock_ratio_6m', 'weighted_stock_ratio', 'last_spike', 'delta_stock_proxy',
            'mtd_qty', 'ttl_qty', 'bc_med'
        ]]
        return df_prob, df_qty
    
    # ==================== 计算方法 ====================
    def _align_ym(self) -> pd.DataFrame:
        """补齐首月到目标月之间的所有年月"""
        df_gb = self.df_ym.groupby(['tomdmcode'])['bizym'].first().reset_index()
        trans_ym_list = []
        code_list = []
        # 1、By机构补齐年月：机构首个年月~目标年月
        for i, c in enumerate(df_gb['tomdmcode'].tolist()):
            ym = self._generate_business_months(df_gb.loc[i, 'bizym'], self.TAR_YM)
            trans_ym_list.extend(ym)
            code_list.extend([c] * len(ym))
        matrix = pd.DataFrame({
            'tomdmcode': code_list,
            'bizym': trans_ym_list
        })
        # 2、关联每月对应数量，补齐月以0填充
        matrix = pd.merge(matrix, self.df_ym[['tomdmcode','bizym','qty']], on=['tomdmcode','bizym'], how='left')
        matrix['qty'] = matrix['qty'].fillna(0)
        return matrix
    
    def _get_time_ft(self, matrix, jump_ym, drop_ym) -> pd.DataFrame:
        """时间特征"""
        # 年份
        matrix['year'] = matrix['bizym'].map(lambda x: int(str(x)[:4])).astype(np.int32)
        # 月份
        matrix['month'] = matrix['bizym'].map(lambda x: int(str(x)[4:])).astype(np.int8)
        # 季度
        matrix['quarter'] = ((matrix['month'] - 1) // 3 + 1).astype(np.int8)
        # 是否季度初
        # quarter_bgs = [1, 4, 7, 10]
        # matrix['is_qtr_begin'] = matrix['month'].isin(quarter_bgs).astype(np.int8)
        # 是否季度末
        # quarter_ends = [3, 6, 9, 12]
        # matrix['is_qtr_end'] = matrix['month'].isin(quarter_ends).astype(np.int8)
        # 是否年末
        # matrix['is_year_end'] = (matrix['month'] == 12).astype(np.int8)
        matrix['is_jump_ym'] = matrix['bizym'].isin(jump_ym).astype(np.int8)
        matrix['is_drop_ym'] = matrix['bizym'].isin(drop_ym).astype(np.int8)
        return matrix
    
    def _get_lag_ft(self, matrix) -> pd.DataFrame:
        """滞后特征"""
        ym_map = dict(zip(self.full_ym_list, range(len(self.full_ym_list))))
        matrix['ym_num'] = (matrix['bizym'].map(ym_map)).astype(np.int8)
        # 过去1,2,3,6,12个月进货量
        matrix = self._lag_feature(matrix, [1,2,3,6,12], 'qty')
        return matrix
    
    def _get_rfm_base(self, group) -> pd.DataFrame:
        """RFM特征(1/2)"""
        group = group.sort_values('bizym')
        # 过去6个月总进货量
        group['mnt6m'] = group['qty'].rolling(
            window=6, min_periods=1
        ).sum().shift(1)
        # 过去3个月总进货量
        group['mnt3m'] = group['qty'].rolling(
            window=3, min_periods=1
        ).sum().shift(1)
        # 过去6个月总进货次数
        group['freq6m'] = group['ym_freq'].rolling(
            window=6, min_periods=1
        ).sum().shift(1)
        # 过去3个月总进货次数
        group['freq3m'] = group['ym_freq'].rolling(
            window=3, min_periods=1
        ).sum().shift(1)
        # 当月之前最近一次进货日期
        group['last_sellin_date'] = group['last_sellin_date'].shift(1)
        return group
    
    def _get_rfm_ft(self, matrix) -> pd.DataFrame:
        """RFM特征(2/2)"""
        # 当月最后一次进货时间、当月进货次数
        df_rfm = self.df_data[['tomdmcode','bizym','transdate','qty']].copy()
        df_rfm_rf = df_rfm.groupby(['tomdmcode','bizym']).agg({'transdate':'last', 'qty':'count'}).reset_index().rename({'transdate':'last_sellin_date', 'qty':'ym_freq'}, axis=1)
        
        # 无进货月：当月最后一次进货时间用最近进货月填充、当月进货次数用0填充
        matrix_rfm = matrix[['tomdmcode','bizym','qty']].copy()
        matrix_rfm = matrix_rfm.merge(df_rfm_rf, how='left', on=['tomdmcode','bizym'])
        matrix_rfm['last_sellin_date'] = matrix_rfm.groupby('tomdmcode')['last_sellin_date'].transform(lambda x: x.fillna(method='ffill'))
        matrix_rfm['ym_freq'] = matrix_rfm['ym_freq'].fillna(0)
        
        # 过去3&6个月总进货量、过去3&6个月总进货次数、当月之前最近一次进货日期
        matrix_rfm = matrix_rfm.groupby('tomdmcode').apply(self._get_rfm_base).reset_index(drop=True)
        
        # 过去6个月平均月进货量
        matrix_rfm['mnt6m_per_month'] = round(matrix_rfm['mnt6m'] / 6, 2)
        # 过去3个月平均月进货量
        matrix_rfm['mnt3m_per_month'] = round(matrix_rfm['mnt3m'] / 3, 2)
        # 过去6个月平均月进货次数
        matrix_rfm['freq6m_per_month'] = round(matrix_rfm['freq6m'] / 6, 2)
        # 过去3个月平均月进货次数
        matrix_rfm['freq3m_per_month'] = round(matrix_rfm['freq3m'] / 3, 2)
        # 过去6个月平均单笔进货量
        matrix_rfm['mnt6m_per_trans'] = round(matrix_rfm['mnt6m'] / matrix_rfm['freq6m'], 2)
        # 过去3个月平均单笔进货量
        matrix_rfm['mnt3m_per_trans'] = round(matrix_rfm['mnt3m'] / matrix_rfm['freq3m'], 2)
        # 处理除0的情形
        matrix_rfm.loc[(matrix_rfm['freq3m']==0), 'mnt3m_per_trans'] = 0
        matrix_rfm.loc[(matrix_rfm['freq6m']==0), 'mnt6m_per_trans'] = 0
        # Recency：“当月之前最近一次进货日期”距离“当月最后一天”的天数
        matrix_rfm['current_date'] = matrix_rfm['bizym'].apply(lambda x: self._get_last_day_of_month(int(str(x)[:4]), int(str(x)[4:])))
        matrix_rfm['recency'] = (pd.to_datetime(matrix_rfm['current_date']) - matrix_rfm['last_sellin_date']).dt.days
        matrix = pd.merge(matrix, matrix_rfm.drop(columns=['qty','ym_freq','last_sellin_date','current_date']), how='left', on=['tomdmcode','bizym'])
        return matrix
    
    def _get_trend_ft(self, matrix) -> pd.DataFrame:
        """趋势特征"""
        # 上月对比近6个月进货量变化趋势
        matrix = self._get_growth_between_fts(matrix,'qty_lag_1','mnt6m_per_month','delta_qty_6m')
        # 上月对比近3个月进货量变化趋势
        matrix = self._get_growth_between_fts(matrix,'qty_lag_1','mnt3m_per_month','delta_qty_3m')
        # 过去3个月对比过去6个月的倍数
        matrix = self._get_growth_between_fts(matrix,'mnt3m_per_month','mnt6m_per_month','mnt3m_div_6m')
        return matrix
    
    def _get_sellin_ym(self, matrix) -> pd.DataFrame:
        """进货月数"""
        # 过去3个月进货月数
        matrix['has_trans_3m'] = (matrix['qty'] != 0).astype(int).groupby(matrix['tomdmcode']).rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
        matrix['has_trans_3m'] = matrix.groupby('tomdmcode')['has_trans_3m'].transform(lambda x: x.shift(1))
        # 过去6个月进货月数
        matrix['has_trans_6m'] = (matrix['qty'] != 0).astype(int).groupby(matrix['tomdmcode']).rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
        matrix['has_trans_6m'] = matrix.groupby('tomdmcode')['has_trans_6m'].transform(lambda x: x.shift(1))
        return matrix

    def _get_single_growth(self, matrix, lag_num) -> pd.DataFrame:
        """增长率特征(1/2)"""
        conditions = [
            (matrix['qty_lag_{}'.format(lag_num)] == 0) & (matrix['qty'] > 0), # 从零启动
            (matrix['qty_lag_{}'.format(lag_num)] == 0) & (matrix['qty'] == 0), # 持续为零
            (matrix['qty_lag_{}'.format(lag_num)] > 0) # 正常情况
        ]
        choices = [
            1, # 视为100%增长
            0, # 或-1
            (matrix['qty'] - matrix['qty_lag_{}'.format(lag_num)]) / matrix['qty_lag_{}'.format(lag_num)] # 正常增长率
        ]
        matrix['growth_lag_{}'.format(lag_num)] = np.select(conditions, choices, default=np.nan)
        matrix['growth_lag_{}'.format(lag_num)] = matrix.groupby('tomdmcode')['growth_lag_{}'.format(lag_num)].transform(lambda x: x.shift(1))
        return matrix
    
    def _get_growth_ft(self, matrix) -> pd.DataFrame:
        """增长率特征(2/2)"""
        matrix = self._get_single_growth(matrix, 1)
        matrix = self._get_single_growth(matrix, 2)
        matrix = self._get_single_growth(matrix, 3)
        matrix = self._get_single_growth(matrix, 6)
        matrix = self._get_single_growth(matrix, 12)
        return matrix
    
    def _calculate_growth_from_pivot(self, matrix, special_ym):
        """从透视表计算增长率(1/2)"""
        df = matrix[['tomdmcode','bizym','qty']]
        pivot_df = df.pivot_table(
            index='tomdmcode',
            columns='bizym',
            values='qty',
            aggfunc='sum'
        )
        pivot_df = pivot_df.replace(np.nan, 0)
        
        result_list = []
        for tar_ym in special_ym:
            same_month = self._get_previous_month(tar_ym, 12) # 去年同期
            last_month = self._get_previous_month(tar_ym, 13) # 去年同期上月
            
            if same_month in pivot_df.columns and last_month in pivot_df.columns:
                growth = (pivot_df[same_month] - pivot_df[last_month]) / pivot_df[last_month] * 100      
                result = pd.DataFrame({
                    'tomdmcode': pivot_df.index,
                    'growth_yoy': growth.values,
                    'target_ym': tar_ym,
                })
                result = result.replace([np.nan, np.inf , -np.inf ], 0)
                result_list.append(result)
        return pd.concat(result_list)
    
    def _calculate_growth_yoy(self, matrix, jump_ym, drop_ym):
        df_jump = self._calculate_growth_from_pivot(matrix, jump_ym)
        df_drop = self._calculate_growth_from_pivot(matrix, drop_ym)
        matrix = pd.merge(matrix, df_jump, how='left', left_on=['tomdmcode','bizym'], right_on=['tomdmcode','target_ym'])
        matrix['growth_jump'] = matrix['growth_yoy'].replace(np.nan, 0)
        matrix = matrix.drop(columns=['target_ym','growth_yoy'])
        matrix = pd.merge(matrix, df_drop, how='left', left_on=['tomdmcode','bizym'], right_on=['tomdmcode','target_ym'])
        matrix['growth_drop'] = matrix['growth_yoy'].replace(np.nan, 0)
        matrix = matrix.drop(columns=['target_ym','growth_yoy'])
        return matrix

    def _get_mtd_qty(self, matrix) -> pd.DataFrame:
        """MTD特征"""
        # 列出起始到目标月的所有日期 
        # e.g. 202301~202510 -> 2023/1/1，2023/1/2，...，2025/10/31
        df_list = []
        for bizym in self.full_ym_list:
            date_start = self._get_first_day_of_month(int(str(bizym)[:4]), int(str(bizym)[4:]))
            date_end = self._get_last_day_of_month(int(str(bizym)[:4]), int(str(bizym)[4:]))
            date_range = pd.date_range(start=date_start, end=date_end).tolist()
            df_list.append(pd.DataFrame({'bizym':[bizym]*len(date_range),'transdate':date_range}))
        df_days = pd.concat(df_list)
        # 标记工作日
        df_days['is_workday'] = df_days['transdate'].apply(lambda x: 1 if calendar.is_workday(x) else 0)
        # 工作日编号
        df_days['wd'] = (df_days.groupby(['bizym'])['is_workday'].cumsum().values) * df_days['is_workday']
        # 每月工作日天数
        df_ym_wd = df_days.groupby(['bizym'])['is_workday'].sum().reset_index().rename({'is_workday':'num_wd'}, axis=1)
        # 每月第k个工作日
        df_wd_date = df_days[df_days['wd']==self.MTD].drop(columns=['is_workday','wd']).rename({'transdate':'wd_date'}, axis=1)
        df_ym_wd['wd_date'] = df_wd_date['wd_date'].values
        # MTD进货量
        df_mtd = pd.merge(self.df_data[['tomdmcode','bizym','transdate','qty']], df_ym_wd, how='left', on=['bizym'])
        df_mtd = df_mtd[df_mtd['transdate']<=df_mtd['wd_date']]
        df_mtd = df_mtd.groupby(['tomdmcode','bizym'])['qty'].sum().reset_index().rename({'bizym':'mtd_bizym','qty':'mtd_qty'}, axis=1)
        matrix = pd.merge(matrix, df_mtd, how='left', left_on=['tomdmcode','bizym'], right_on=['tomdmcode','mtd_bizym']).drop(columns=['mtd_bizym'])
        matrix = matrix.merge(df_ym_wd[['bizym','num_wd']], how='left', on=['bizym'])
        matrix['mtd_qty'] = matrix['mtd_qty'].fillna(0)
        return matrix
    
    def _get_gap_ft(self, matrix) -> pd.DataFrame:
        """进货间隔特征"""
        # 进货间隔
        df_diff = self.df_data[['tomdmcode','bizym','transdate','qty']].copy().sort_values(by=['tomdmcode','bizym'])
        df_diff['PrevPurchaseDate'] = df_diff.groupby('tomdmcode')['transdate'].shift(1)
        df_diff['DayDiff'] = (df_diff['transdate'] - df_diff['PrevPurchaseDate']).dt.days
        df_diff = df_diff.dropna(subset=['DayDiff'])
        # 过去3个月平均进货间隔
        diff_ft_3m_list=[]
        for i in range(3, len(self.full_ym_list)):
            ym_start, ym_end, ym_target = self.full_ym_list[i-3], self.full_ym_list[i-1], self.full_ym_list[i]
            df_slice = df_diff[(df_diff['bizym']>=ym_start) & (df_diff['bizym']<=ym_end)][['tomdmcode','bizym','transdate','DayDiff']]
            df_hos = df_slice.groupby('tomdmcode').agg({'DayDiff': 'mean'}).reset_index()
            df_hos.columns = ['tomdmcode', 'DayDiffMean3m']
            df_hos['bizym'] = ym_target
            diff_ft_3m_list.append(df_hos)
        df_diff_ft_3m = pd.concat(diff_ft_3m_list)
        # 过去6个月平均进货间隔
        diff_ft_6m_list=[]
        for i in range(6, len(self.full_ym_list)):
            ym_start, ym_end, ym_target = self.full_ym_list[i-6], self.full_ym_list[i-1], self.full_ym_list[i]
            df_slice = df_diff[(df_diff['bizym']>=ym_start) & (df_diff['bizym']<=ym_end)][['tomdmcode','bizym','transdate','DayDiff']]
            df_hos = df_slice.groupby('tomdmcode').agg({'DayDiff': 'mean'}).reset_index()
            df_hos.columns = ['tomdmcode', 'DayDiffMean6m']
            df_hos['bizym'] = ym_target
            diff_ft_6m_list.append(df_hos)
        df_diff_ft_6m = pd.concat(diff_ft_6m_list)
        # 过去12个月平均进货间隔
        diff_ft_12m_list=[]
        for i in range(12, len(self.full_ym_list)):
            ym_start, ym_end, ym_target = self.full_ym_list[i-12], self.full_ym_list[i-1], self.full_ym_list[i]
            df_slice = df_diff[(df_diff['bizym']>=ym_start) & (df_diff['bizym']<=ym_end)][['tomdmcode','bizym','transdate','DayDiff']]
            df_hos = df_slice.groupby('tomdmcode').agg({'DayDiff': 'mean'}).reset_index()
            df_hos.columns = ['tomdmcode', 'DayDiffMean12m']
            df_hos['bizym'] = ym_target
            diff_ft_12m_list.append(df_hos)
        df_diff_ft_12m = pd.concat(diff_ft_12m_list)

        matrix = matrix.merge(
            df_diff_ft_3m, how='left', on=['tomdmcode','bizym']
        )
        matrix = matrix.merge(
            df_diff_ft_6m, how='left', on=['tomdmcode','bizym']
        )
        matrix = matrix.merge(
            df_diff_ft_12m, how='left', on=['tomdmcode','bizym']
        )
        # 最近一次进货是否超过平均间隔
        matrix['is_exceed_diff12m'] = (matrix['recency'] > matrix['DayDiffMean12m']).astype(int)
        matrix['is_exceed_diff6m'] = (matrix['recency'] > matrix['DayDiffMean6m']).astype(int)
        matrix['is_exceed_diff3m'] = (matrix['recency'] > matrix['DayDiffMean3m']).astype(int)
    
        return matrix

    def _get_ym_dist(self, matrix):
        """距离首次进货月数"""
        matrix['num_sellin_ym'] = matrix.groupby(['tomdmcode'])['bizym'].transform(
            lambda x: list(range(1,len(x)+1))
        )
        return matrix
    
    def _get_recency_months(self, matrix):
        matrix = matrix.sort_values(['tomdmcode', 'bizym'])
        matrix['is_purchase'] = matrix['qty'] > 0
        matrix['purchase_ym_lag'] = (
            matrix.groupby('tomdmcode')['bizym']
            .shift(1)
            .where(matrix.groupby('tomdmcode')['is_purchase'].shift(1))
        )
        # 最近一次历史进货月
        matrix['last_purchase_ym'] = (
            matrix.groupby('tomdmcode')['purchase_ym_lag']
            .ffill()
        )
        matrix['recency_months'] = matrix.apply(lambda x: self._months_between(x['last_purchase_ym'],x['bizym']) if not np.isnan(x['last_purchase_ym']) else np.nan, axis=1)       
        return matrix
    
    def _get_base_consumption_gb(self, group):
        group = group.sort_values('bizym')
        # 过去12个月总进货量
        group['bc_med'] = group['qty'].rolling(
            window=12, min_periods=3
        ).median().shift(1)
        group['bc_mean'] = group['qty'].rolling(
            window=12, min_periods=3
        ).mean().shift(1)
        return group

    def _get_base_consumption(self, matrix):
        df_bc = matrix.groupby('tomdmcode')[['tomdmcode','bizym','qty']].apply(self._get_base_consumption_gb).reset_index(drop=True)
        matrix = matrix.merge(df_bc[['tomdmcode','bizym','bc_med']], how='left', on=['tomdmcode','bizym'])
        return matrix
    
    def _get_stock_ratios(self, matrix):
        matrix['stock_ratio_3m'] = matrix['mnt3m'] / (matrix['bc_med']*3+1)
        matrix['stock_ratio_6m'] = matrix['mnt6m'] / (matrix['bc_med']*6+1)
        matrix['last_spike'] = matrix['qty_lag_1'] / (matrix['bc_med']+1)
        matrix['weighted_stock_ratio'] = (0.5*matrix['qty_lag_1']+0.3*matrix['qty_lag_2']+0.2*matrix['qty_lag_3']) / (matrix['bc_med']+1)
        return matrix

    def _get_delta_stock(self, matrix):
        df = matrix[['tomdmcode', 'bizym', 'bc_med', 'qty']].copy()
        df['delta_stock_proxy'] = np.nan  # 初始化库存余额列

        # 按照机构和时间排序
        df = df.sort_values(by=['tomdmcode', 'bizym'])
        
        # 计算库存余额
        for inst_code, group in df.groupby('tomdmcode'):
            group = group[(group['bizym']>=202401) & (group['bc_med'].notna())]
            if len(group) > 0:
                # 初始化每个医院的库存余额
                group['delta_stock_proxy'] = 0
                group.iloc[0, -1] = 0 + group.iloc[0]['qty'] - group.iloc[0]['bc_med']
                for i in range(1, len(group)):
                    # 上个月的库存余额 + 本月的进货量 - 本月消耗量（这里假设消耗量等于进货量）
                    group.iloc[i, -1] = \
                        group.iloc[i-1]['delta_stock_proxy'] + group.iloc[i]['qty'] - group.iloc[i]['bc_med']
                
                # 将计算结果返回到原数据框中
                df.loc[group.index, 'delta_stock_proxy'] = group['delta_stock_proxy']
        df['delta_stock_proxy'] = df.groupby('tomdmcode')['delta_stock_proxy'].transform(lambda x: x.shift(1))
        matrix = matrix.merge(df[['tomdmcode','bizym','delta_stock_proxy']], how='left', on=['tomdmcode','bizym'])
        return matrix

    # ==================== 辅助方法 ====================
    def _get_previous_month(self, year_month:int, months_back:int) -> int:
        """获取指定年月向前N个月对应的年月"""
        year = year_month // 100
        month = year_month % 100
        total_months = year * 12 + month - 1  # -1 因为月份从1开始
        new_total_months = total_months - months_back
        new_year = new_total_months // 12
        new_month = new_total_months % 12 + 1  # +1 恢复月份从1开始
        return new_year * 100 + new_month

    def _generate_business_months(self, start_month: int, end_month: int) -> list:
        # 转换为日期对象
        start_date = pd.to_datetime(str(start_month), format='%Y%m')
        end_date = pd.to_datetime(str(end_month), format='%Y%m')
        # 生成月份范围
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq='MS'  # 每月第一天
        )
        # 转换为业务月格式
        business_months = [d.strftime('%Y%m') for d in date_range]
        return [int(month) for month in business_months]
    
    def _lag_feature(self, df, lags, col):
        tmp = df[['ym_num','tomdmcode',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['ym_num','tomdmcode',col+'_lag_'+str(i)]
            shifted['ym_num'] += i
            df = pd.merge(df, shifted, on=['ym_num','tomdmcode'], how='left')
        return df
    
    def _get_growth_between_fts(self, matrix, numerator, denominator, ft_name):
        conditions = [
            (matrix[denominator] == 0) & (matrix[numerator] > 0), # 分母为0、分子大于0
            (matrix[denominator] == 0) & (matrix[numerator] == 0), # 分母为0、分子为0
            (matrix[denominator] > 0) # 正常情况
        ]
        choices = [
            1, # 视为100%增长
            0, # 或-1
            (matrix[numerator] - matrix[denominator]) / matrix[denominator] # 正常增长率
        ]
        matrix[ft_name] = np.select(conditions, choices, default=np.nan)
        return matrix
    
    def _get_first_day_of_month(self, year, month):
        first_day = date(year, month , 1)
        return first_day #.strftime("%Y-%m-%d")

    def _get_last_day_of_month(self, year, month):
        if month == 12:
            next_month_first_day = date(year + 1, 1, 1)
        else:
            next_month_first_day = date(year, month + 1, 1)
        last_day = next_month_first_day - timedelta(days=1)
        return last_day #.strftime("%Y-%m-%d")
    
    def _months_between(self, ym1, ym2):
        ''' 计算第2个减第1个的间隔月数'''
        # 提取年份和月份
        y1 = ym1 // 100
        m1 = ym1 % 100
        y2 = ym2 // 100
        m2 = ym2 % 100
        # 计算总月数差
        ttl1 = y1 * 12 + m1
        ttl2 = y2 * 12 + m2
        # 返回绝对差值
        return ttl2 - ttl1