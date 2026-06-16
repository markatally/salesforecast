import pandas as pd
import logging

class DataPreprocessing:
    def __init__(self, df, TAR_YM, cutoff_ym=202301, is_vaild=True):
        '''
        Docstring for __init__
    
        :df: Raw Data
        :TAR_YM: 预测年月
        :cutoff_ym: 特征计算起始年月
        :is_vaild: 是否统计当月进货终端数量
        '''
        self.df_ori = df
        self.TAR_YM = TAR_YM
        self.is_vaild = is_vaild
        self.logger = self._setup_logger()
        self.logger.info(f"预测月份：{TAR_YM}")
        
        self.df_train = self.df_ori[self.df_ori['bizym']<self.TAR_YM]
        # 测试情形
        if self.is_vaild:
            self.df_test = self.df_ori[self.df_ori['bizym']==self.TAR_YM]
        
        self.START_YM = cutoff_ym 
        # self.START_YM = self._get_previous_month(self.TAR_YM, 24) # 历史2年的数据
        self.FULL_YM = self._get_previous_month(self.TAR_YM, 12)

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DP')
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
    
    def read_data(self, input_path:str) -> pd.DataFrame:
        """
        读取原始数据
        
        df_ori.columns:
            - transdate: 流向交易日期
            - tomdmcode: SNF机构编码
            - toedivndrnm: SNF机构名称
            - bizym: 业务月
            - tomdmprovince: 机构注册省份
            - tomdmcity: 机构注册地市
            - tomdmcounty: 机构注册区县
            - tohospitallevel: 医院等级
            - qty: 流向交易数量
        """
        df_ori = pd.read_csv(input_path)
        df_ori['transdate'] = pd.to_datetime(df_ori['transdate'])
        df_ori = df_ori.groupby([
            'tomdmcode','toedivndrnm',
            'bizym','transdate',
            'tomdmprovince','tomdmcity','tomdmcounty',
            'tohospitallevel'])[['qty']].sum().reset_index()
        df_ori = df_ori[df_ori['qty']!=0]
        return df_ori

    def process_return_records(self, df:pd.DataFrame) -> pd.DataFrame:
        """处理退货记录"""
        df_pcd, idx_list_phase1 = self._process_return_records_phase1(df)
        df_pcd = df_pcd.drop(index=idx_list_phase1)
        idx_list_phase2 = self._process_return_records_phase2(df_pcd)
        df_pcd = df_pcd.drop(index=idx_list_phase2)
        self.logger.info(f"退货记录处理完成：剩余{len(df_pcd[df_pcd['qty']<0])}条退货记录")
        df_pcd = df_pcd[df_pcd['qty']>0]
        return df_pcd

    def get_target_inst(self, df:pd.DataFrame) -> pd.DataFrame:
        """仅保留近一年有进货医院"""
        log_num_list = []
        inst_has_trans = df[(df['bizym']>=self.FULL_YM) & (df['bizym']<self.TAR_YM)]['tomdmcode'].unique()
        df_data = df[(df['bizym']>=self.START_YM) & (df['bizym']<=self.TAR_YM) & (df['tomdmcode'].isin(inst_has_trans))]
        self.logger.info(f"近一年内有进货医院：{len(inst_has_trans)}家")
        log_num_list.append(len(inst_has_trans))
        # 测试情形
        if self.is_vaild:
            df_tar = self.df_test[self.df_test['qty']>0]
            self.logger.info(f"当月产生流向医院：{df_tar['tomdmcode'].nunique()}家")
            log_num_list.append(df_tar['tomdmcode'].nunique())
            self.logger.info(f"属于近一年内有进货医院{df_tar[df_tar['tomdmcode'].isin(inst_has_trans)]['tomdmcode'].nunique()}家（{round(df_tar[df_tar['tomdmcode'].isin(inst_has_trans)]['qty'].sum(),1)}盒）")
            log_num_list.append(df_tar[df_tar['tomdmcode'].isin(inst_has_trans)]['tomdmcode'].nunique())
            log_num_list.append(round(df_tar[df_tar['tomdmcode'].isin(inst_has_trans)]['qty'].sum(), 1))
            self.logger.info(f"首次/一年前进货医院：{df_tar[~df_tar['tomdmcode'].isin(inst_has_trans)]['tomdmcode'].nunique()}家（{round(df_tar[~df_tar['tomdmcode'].isin(inst_has_trans)]['qty'].sum(),1)}盒）")
            log_num_list.append(df_tar[~df_tar['tomdmcode'].isin(inst_has_trans)]['tomdmcode'].nunique())
            log_num_list.append(round(df_tar[~df_tar['tomdmcode'].isin(inst_has_trans)]['qty'].sum(), 1))
            df_test = self.df_test[self.df_test['tomdmcode'].isin(inst_has_trans)]
            df_data = pd.concat([df_data, df_test]).reset_index(drop=True)
   
        if self.is_vaild:
            description_list = ["近一年内有进货医院","当月产生流向医院","属于近一年内有进货医院","属于近一年内有进货医院贡献量","首次/一年前进货医院","首次/一年前进货医院贡献量"]
        else:
            description_list = ["近一年内有进货医院"]
        df_log = pd.DataFrame({
            'Description': description_list,
            'Value': log_num_list
        })
        return df_data, df_log
    
    # ==================== 计算方法 ====================
    def _process_return_records_phase1(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        处理退货记录(1/2)：逐条向前抵消
        
        df_ori.columns:
            - transdate: 流向交易日期
            - tomdmcode: 机构编码
            - qty: 流向交易数量
        """
        idx_list_fw = []
        df_ast = df.copy()
        # 遍历有退货医院
        for hos in df_ast[df_ast['qty']<0]['tomdmcode'].unique():
            df_hos = df_ast[df_ast['tomdmcode']==hos][['transdate', 'qty']].sort_values(by='transdate').reset_index()
            # 从前向后遍历退货记录
            for idx in df_hos[df_hos['qty']<0].index: # .index[::-1]:
                # 退货日期&量
                return_date = df_hos.loc[idx, 'transdate']
                return_qty = - df_hos.loc[idx, 'qty']
                # 从后向前搜索
                df_search = df_hos[df_hos['transdate']<return_date]
                if len(df_search) > 0:
                    for i, row in df_search[::-1].iterrows():
                        if row['qty'] > 0:
                            # 当前记录足够抵消退货量
                            if row['qty'] >= return_qty:
                                df_ast.loc[row['index'], 'qty'] -= return_qty
                                df_hos.loc[i, 'qty'] -= return_qty
                                return_qty=0
                                break
                            # 当前记录仅能抵消部分退货量
                            else:
                                df_ast.loc[row['index'], 'qty'] = 0
                                df_hos.loc[i, 'qty'] = 0
                                return_qty -= row['qty']
                if return_qty < -df_hos.loc[idx, 'qty']:
                    idx_list_fw.append(df_hos.loc[idx, 'index'])
        return df_ast, idx_list_fw
    
    def _process_return_records_phase2(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        处理退货记录(2/2)：去除开头为负，故未能抵消的记录
        
        df.columns:
            - transdate: 流向交易日期
            - tomdmcode: 机构编码
            - qty: 流向交易数量
        """
        idx_list_fw = []
        df_ast = df.copy()
        df_ast = df_ast[df_ast['qty']!=0] # 开头可能存在连续为0的记录
        for hos in df_ast[df_ast['qty']<0]['tomdmcode'].unique():
            df_hos = df_ast[df_ast['tomdmcode']==hos][['transdate', 'qty']].sort_values(by='transdate').reset_index()
            # 记录开头连续为负的索引
            all_negative_mask = df_hos['qty'] < 0
            continuous_negative_mask = (all_negative_mask.cumprod() == 1)
            df_neg = df_hos[continuous_negative_mask]
            idx_list_fw.extend(df_neg['index'].tolist())
        return idx_list_fw
    
    # ==================== 辅助方法 ====================
    def _get_previous_month(self, year_month:int, months_back:int) -> int:
        """
        获取指定年月向前N个月对应的年月
        
        :year_month: 指定年月
        :months_back: 向前N个月

        示例：
            Inputs:
                year_month = 202508
                months_back = 6
            Outputs:
                202502
        """
        year = year_month // 100
        month = year_month % 100
        total_months = year * 12 + month - 1  # -1 因为月份从1开始
        new_total_months = total_months - months_back
        new_year = new_total_months // 12
        new_month = new_total_months % 12 + 1  # +1 恢复月份从1开始
        return new_year * 100 + new_month
