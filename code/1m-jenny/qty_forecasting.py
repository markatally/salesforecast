import pandas as pd
import numpy as np
import math
import logging

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans 

from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV, cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rc("font",family='Microsoft YaHei')
sns.set_style("whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc("axes",
        labelweight="bold", # 标签粗细
        labelsize="large", # 标签字体
        titleweight="bold", # 标题粗细
        titlesize=16, # 标题字体
        titlepad=10,
      )
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

class QtyForecasting:
    def __init__(self, df_ym_ft, TAR_YM, MTD, is_valid=True, is_mtd=True):
        self.df_ym_ft = df_ym_ft
        self.TAR_YM = TAR_YM
        self.MTD = MTD
        self.is_valid = is_valid
        self.is_mtd = is_mtd

        self.df_train = self.df_ym_ft[self.df_ym_ft['bizym']<self.TAR_YM].reset_index(drop=True)
        self.df_test = self.df_ym_ft[self.df_ym_ft['bizym']==self.TAR_YM].reset_index(drop=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('QF')
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
    
    def apply_baseline(self):
        """XGBoost"""
        # 数据集
        df = self.df_train.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty'])
        y = self.df_train["ttl_qty"]
        # 验证集分数
        rgr = XGBRegressor(random_state=66)
        ps = self._predefined_split(self.df_train)
        n_scores = cross_val_score(rgr, df, y, cv=ps, scoring='neg_root_mean_squared_error')

        # 训练模型（基于全量数据，包括验证集）
        rgr = XGBRegressor(random_state=66)
        rgr.fit(df, y)

        # 验证集误差率
        df_valid = self.df_train[self.df_train['bizym']==self._get_previous_month(self.TAR_YM,1)].copy()
        val_x = df_valid.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty'])
        val_y_hat = rgr.predict(val_x)
        df_valid['ttl_qty_pos'] = df_valid['ttl_qty']
        df_valid.loc[df_valid['ttl_qty']<0, 'ttl_qty_pos'] = 0
        
        # 输出预测进货数量
        test_x = self.df_test.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty'])
        y_fcst = rgr.predict(test_x)
        df_fcst = self.df_test[['tomdmcode','bizym','mtd_qty']]
        df_fcst['fcst_qty'] = y_fcst

        '''Testing'''
        if self.is_valid:
            test_y = self.df_test['ttl_qty']
            df_fcst['ttl_qty'] = test_y
            # mae_test_xgb = mean_absolute_error(test_y, y_fcst)
            # self.logger.info('Test MAE: {:.2f}'.format(mae_test_xgb))
            # rmse_test_xgb = root_mean_squared_error(test_y, y_fcst)
            # self.logger.info('Test RMSE: {:.2f}'.format(rmse_test_xgb))

        importances = list(rgr.feature_importances_)
        feature_importances = [(feature, round(float(importance), 2)) for feature, importance in zip(df.columns.tolist(), importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        df_imp = pd.DataFrame({
            'Features': [pair[0] for pair in feature_importances],
            'Importances': [pair[1] for pair in feature_importances]
        })

        self.logger.info('Baseline Valid Error: {:.2f}%'.format(round((val_y_hat.sum() / df_valid['ttl_qty_pos'].sum() - 1) * 100, 2)))
        self.logger.info("Baseline Valid RMSE: {:.2f}".format(round(-np.mean(n_scores), 2)))
        self.logger.info(f'直接训练 + 验证完毕')

        return df_fcst, df_imp

    # ================================================================================
    # KMeans
    # ================================================================================
    def _elbow_method(self, df):
        inertia = []
        K = range(1, 17)
        for k in K:
            KMeans_Model = KMeans(n_clusters=k, random_state=2025)
            KMeans_Model.fit(df)
            inertia.append(KMeans_Model.inertia_)
        fig_elbow = plt.figure(figsize = (8,5))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel("Num of Clusters")
        plt.ylabel("SSE: Sum of Squared Errors")
        plt.title("The Elbow Method")

        return inertia

    def _select_num_group(self, inertia, base_k, sse_thr):
        num_cluster = base_k
        # self.logger.info("SSE: {}".format([round(sse, 0) for sse in inertia]))
        diff = [round(inertia[i] / inertia[0] * 100, 2) for i in range(1, len(inertia))]
        # self.logger.info("SSE%: {}".format(diff))
        for i in range(len(diff)-1):
            if diff[i] <= sse_thr:
                num_cluster = i+2
                break
        return num_cluster

    def apply_clustering(self, cls_ft_list, key_ft, base_k=4, sse_thr=5, k=0):
        df_cluster = self.df_test[['tomdmcode']+cls_ft_list].copy()
        # 1、归一化消除尺度影响
        m = MinMaxScaler()
        arr_ft = m.fit_transform(df_cluster.drop(columns=['tomdmcode']))
        inertia = self._elbow_method(arr_ft)
        
        # 2、选取聚类数量
        if k == 0:
            k = self._select_num_group(inertia, base_k, sse_thr)

        # 3、获取聚类结果
        KMeans_model = KMeans(
            n_clusters=k,  
            init='k-means++',  
            max_iter=100,  
            random_state=2025
        )
        KMeans_model.fit(arr_ft)
        labels = KMeans_model.labels_
        df_cluster["group"] = labels

        # 4、各组编号按指定维度降序排列
        df_group = df_cluster.groupby(['group']).agg({key_ft:'mean'}).reset_index().sort_values(by=[key_ft], ascending=False)
        cluster_map = dict(zip(df_group['group'], range(1, k+1)))
        df_cluster["group"] = df_cluster["group"].apply(lambda x: cluster_map[x])
        
        # 5、输出组别信息
        cls_ft_list.remove(key_ft)
        agg_dic = {
            'tomdmcode':'count',
            key_ft:['mean','min','max']
        }
        agg_dic.update({k:'mean' for k in cls_ft_list})
        df_info = df_cluster.groupby(['group']).agg(agg_dic)
        col_l0, col_l1 = df_info.columns.get_level_values(0), df_info.columns.get_level_values(1)
        df_info.columns = col_l0 + '_' + col_l1
        df_info = df_info.reset_index()
        return df_cluster, df_info
    
    # ================================================================================
    # 分层建模
    # ================================================================================
    def apply_rfm(self, df_cluster, df_info):
        # 挂上组别ID
        df_train = self.df_train.merge(df_cluster[['tomdmcode','group']], how='left', on=['tomdmcode'])
        df_test = self.df_test.merge(df_cluster[['tomdmcode','group']], how='left', on=['tomdmcode'])
        df_valid = df_train[df_train['bizym']==self._get_previous_month(self.TAR_YM, 1)]
        # 分层建模
        df_valid_list = []
        df_rfm_list = []
        group_cv_scores = []
        dict_score = {
            'Group':[],
            'Valid RMSE':[]
        }
        # if self.is_valid:
        #     dict_score = {
        #         'Group':[],
        #         'Val RMSE':[],
        #         'Test MAE':[],
        #         'Test RMSE':[],
        #         'Test Group Error%':[]
        #     }

        df_group_ft_list = []
        for CAT in list(range(1, df_cluster['group'].nunique()+1)):
            self.logger.info('Group {}'.format(CAT))
            dict_score['Group'].append(CAT)
            '''Training'''
            # 数据集
            df_train_cat = df_train[df_train['group']==CAT].reset_index(drop=True)
            trn_x = df_train_cat.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty','group'])
            trn_y = df_train_cat['ttl_qty']
            # CV
            rgr = XGBRegressor(random_state=66)
            ps = self._predefined_split(df_train_cat)
            scores = cross_val_score(rgr, trn_x, trn_y, cv=ps, scoring='neg_root_mean_squared_error')
            group_cv_scores.append(-np.mean(scores))
            self.logger.info('Valid RMSE: {:.2f}'.format(-np.mean(scores)))
            dict_score['Valid RMSE'].append(round(-np.mean(scores), 2))

            '''Forecasting'''
            # 训练模型（基于全量数据，包括验证集）
            rgr = XGBRegressor(random_state=66)
            rgr.fit(trn_x, trn_y)

            # 验证集结果
            df_val_cat = df_valid[df_valid['group']==CAT]
            val_x = df_val_cat.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty','group'])
            df_rfm_val = df_val_cat[['tomdmcode','bizym','ttl_qty']]
            df_rfm_val['fcst_qty'] = rgr.predict(val_x)
            df_rfm_val['ttl_qty_pos'] = df_rfm_val['ttl_qty']
            df_rfm_val.loc[df_rfm_val['ttl_qty']<0, 'ttl_qty_pos'] = 0
            self.logger.info('Valid Error%: {:.2f}%'.format(round((df_rfm_val['fcst_qty'].sum() / df_rfm_val['ttl_qty_pos'].sum() - 1) * 100, 2)))
            df_valid_list.append(df_rfm_val)

            # 输出预测进货数量
            df_test_cat = df_test[df_test['group']==CAT]
            test_x = df_test_cat.drop(columns=['bizym','tomdmcode','mtd_qty','ttl_qty','group'])
            y_fcst = rgr.predict(test_x)
            df_rfm_test = df_test_cat[['tomdmcode','bizym','mtd_qty']]
            df_rfm_test['fcst_qty'] = y_fcst
            
            '''Testing'''
            if self.is_valid:
                test_y = df_test_cat['ttl_qty']
                df_rfm_test['ttl_qty'] = test_y
                # mae_test_xgb = mean_absolute_error(test_y, y_fcst)
                # self.logger.info('Test MAE: {:.2f}'.format(mae_test_xgb))
                # rmse_test_xgb = root_mean_squared_error(test_y, y_fcst)
                # self.logger.info('Test RMSE: {:.2f}'.format(rmse_test_xgb))
                # self.logger.info('Test Group Error%: {:.2f}%'.format((df_rfm_test['fcst_qty'].sum() / df_rfm_test['ttl_qty'].sum() - 1) * 100))
                # dict_score['Test MAE'].append(round(mae_test_xgb, 2))
                # dict_score['Test RMSE'].append(round(rmse_test_xgb, 2))
                # dict_score['Test Group Error%'].append(round((df_rfm_test['fcst_qty'].sum() / df_rfm_test['ttl_qty'].sum() - 1) * 100, 2))
            
            df_rfm_list.append(df_rfm_test)
            self.logger.info(f'='*30)
            
            importances = list(rgr.feature_importances_)
            feature_importances = [(feature, round(float(importance), 2)) for feature, importance in zip(trn_x.columns.tolist(), importances)]
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            df_group_ft = pd.DataFrame({
                'Group': [CAT]*len(feature_importances),
                'Features': [pair[0] for pair in feature_importances],
                'Importances': [pair[1] for pair in feature_importances]
            })
            df_group_ft_list.append(df_group_ft)

        ttl_rmse = self._cal_group_rmse(df_info, group_cv_scores)
        df_valid = pd.concat(df_valid_list)
        df_fcst = pd.concat(df_rfm_list)
        df_imp = pd.concat(df_group_ft_list)

        self.logger.info('RFM Valid Error: {:.2f}%'.format(round((df_valid['fcst_qty'].sum() / df_valid['ttl_qty_pos'].sum() - 1) * 100, 2)))
        self.logger.info("RFM Valid RMSE: {:.2f}".format(round(ttl_rmse, 2)))
        self.logger.info(f'RFM分层训练 + 验证完毕')
        
        return df_fcst, df_imp
    
    def process_fcst_value(self, df_prob, df_fcst):
        df_fcst = df_fcst.merge(df_prob[['tomdmcode','bizym','fcst_label']], how='left', on=['tomdmcode','bizym'])
        # 预测负数，抹0
        df_fcst.loc[df_fcst['fcst_qty']<0, 'fcst_qty'] = 0
        #! MTD: 总量少于MTD，使用MTD数量
        if self.is_mtd:
            df_fcst.loc[df_fcst['fcst_qty']<df_fcst['mtd_qty'], 'fcst_qty'] = df_fcst[df_fcst['fcst_qty']<df_fcst['mtd_qty']]['mtd_qty']
        # 根据进货概率模型输出，将预测不进货的医院设为0
        df_fcst.loc[df_fcst['fcst_label']==0, 'fcst_qty'] = 0
        # 真实数量为负，抹0
        if self.is_valid:
            df_fcst['ttl_qty_pos'] = df_fcst['ttl_qty']
            df_fcst.loc[df_fcst['ttl_qty']<0, 'ttl_qty_pos'] = 0
        return df_fcst

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
    
    def _predefined_split(self, df):
        ''' Sklearn PredefinedSplit '''
        valid_ym = self._get_previous_month(self.TAR_YM, 1)
        test_fold = np.array([-1] * len(df))
        test_fold[df[df['bizym']==valid_ym].index.values] = 0
        ps = PredefinedSplit(test_fold)
        return ps
    
    def _cal_group_rmse(self, df_info, group_cv_scores):
        ttl_rmse = 0
        for num, score in zip(df_info['tomdmcode_count'].tolist(), group_cv_scores):
            ttl_rmse += num*score**2
        ttl_rmse = np.sqrt(ttl_rmse / df_info['tomdmcode_count'].sum())
        return ttl_rmse
            