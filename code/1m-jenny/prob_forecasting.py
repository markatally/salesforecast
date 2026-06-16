import pandas as pd
import numpy as np
import math
import logging
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.feature_selection import RFECV

from prophet import Prophet

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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

class HosNumForecasting:
    def __init__(self, df_data, TAR_YM, is_valid=True):
        self.df_data = df_data
        self.TAR_YM = TAR_YM
        self.is_valid = is_valid
    
    def apply_prophet(self, yearly_num=10):
        df_sellin = self._make_input_data()
        model = Prophet(
            growth='linear', 
            seasonality_mode='additive', 
            interval_width=0.95, 
            daily_seasonality=False, 
            weekly_seasonality=False, 
            yearly_seasonality=yearly_num
        )
        model.fit(df_sellin[df_sellin['bizym']<self.TAR_YM])
        future = model.make_future_dataframe(periods=1, freq='MS')
        fcst = model.predict(future)
        num_prophet = fcst['yhat'].values[-1]
        num_prophet = int(round(num_prophet,0))
        
        if self.is_valid:
            x_axis = df_sellin['bizym'].astype(str)
        else:
            x_axis = df_sellin['bizym'].astype(str).tolist() + [str(self.TAR_YM)]
        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(
            x = df_sellin['bizym'].astype(str), 
            y = df_sellin['y'], 
            mode = 'lines+markers', 
            line = dict(color='purple', width=2),
            name = 'Actual Sellin Insts'
        ))
        fig_prophet.add_trace(go.Scatter(
            x = x_axis, 
            y = fcst['yhat'], 
            mode = 'lines+markers', 
            line = dict(color='green', width=2),
            name = 'Fcst Sellin Insts'
        ))
        fig_prophet.update_xaxes(title='bizym')
        fig_prophet.update_yaxes(title='#Insts')
        fig_prophet.update_layout(
            template='simple_white',
            title=f'Monthly Sellin Insts',
            width=800, 
            height=400
        )
        return num_prophet, fig_prophet
    
    def _make_input_data(self):
        # 筛选进货终端
        df_term_ym = self.df_data.groupby(['tomdmcode','bizym'])[['qty']].sum().reset_index()
        df_sellin = df_term_ym[(df_term_ym['qty']>0)]
        df_sellin = df_sellin.groupby(['bizym'])['tomdmcode'].nunique().reset_index()

        # 准备Prophet输入数据
        start_date = str(df_sellin.iloc[0,0])
        df_sellin['ds'] = pd.date_range(start=start_date[:4] + '-' + start_date[4:], periods=len(df_sellin), freq='MS')
        df_sellin = df_sellin.rename({'tomdmcode': 'y'}, axis=1)
        return df_sellin

class ProbForecasting:
    def __init__(self, df_ym_ft, TAR_YM, MTD, is_valid=True, is_mtd=True):
        # self.df_no_returns = df_no_returns
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
        logger = logging.getLogger('PF')
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
        df = self.df_train.drop(columns=['bizym','tomdmcode','is_sellin','mtd_qty'])
        y = self.df_train["is_sellin"]
        
        # CV
        clf = XGBClassifier(random_state=66)
        ps = self._predefined_split()
        n_scores = cross_val_score(clf, df, y, cv=ps, scoring='accuracy')
        
        self.logger.info("Baseline Valid Accuracy: {:.2f}".format(round(np.mean(n_scores)*100, 2)))
        self.logger.info('直接训练 + 验证完毕')

        return df.columns

    def apply_rfe(self):
        """XGBoost + RFE特征筛选"""
        # 数据集
        df = self.df_train.drop(columns=['bizym','tomdmcode','is_sellin','mtd_qty'])
        y = self.df_train["is_sellin"]
        
        # RFE选取特征
        ps = self._predefined_split()
        rfecv=RFECV(  
            estimator=XGBClassifier(random_state=66),  
            step=1,  
            cv=ps,
            scoring='accuracy',  
            min_features_to_select=8,
            verbose=0 # 1 
        )  
        rfecv.fit(df, y) 
        sel_fts = list(df.columns[rfecv.support_]) 
        
        # CV
        clf = XGBClassifier(random_state=66)
        n_scores = cross_val_score(clf, df[sel_fts], y, cv=ps, scoring='accuracy')
         
        # RFE图像
        fig_rfe = plt.figure(figsize=(8, 5))
        x1 = rfecv.cv_results_['n_features']  
        y1 = rfecv.cv_results_['mean_test_score']  
        ax = sns.lineplot(x=x1, y=y1)  
        plt.xlabel("Number of Features Selected")  
        plt.ylabel("Mean CV Accuracy")  
        plt.title("RFECV: Number of Features vs. CV Accuracy")  
        ax.axhline(y=np.max(y1), color='r', linestyle=':', label='Max CV Accuracy')  
        plt.xticks(x1)  
        plt.yticks(np.arange(0.6,0.9,0.02))  
        plt.grid(True)  
        plt.legend()

        self.logger.info("RFE Valid Accuracy: {:.2f}".format(round(np.mean(n_scores)*100, 2)))
        self.logger.info('RFE特征筛选 + 训练 + 验证完毕') 

        return sel_fts

    def forecasting(self, sel_fts):
        """预测进货概率"""
        # 数据集
        trn_x = self.df_train[sel_fts]
        trn_y = self.df_train['is_sellin']
        test_x = self.df_test[sel_fts]
        
        # 训练模型（基于全量数据，包括验证集）
        clf = XGBClassifier(random_state=66)
        clf.fit(trn_x, trn_y)
        y_fcst_test = clf.predict_proba(test_x)[:,1]
        
        # 特征重要性
        importances = list(clf.feature_importances_)
        feature_importances = [(feature, round(float(importance), 2)) for feature, importance in zip(sel_fts, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        df_imp = pd.DataFrame({
            'Features':[pair[0] for pair in feature_importances],
            'Importances':[pair[1] for pair in feature_importances]
        })
        
        # 输出预测进货概率
        df_prob = self.df_test[['tomdmcode','bizym','mtd_qty']]
        df_prob['fcst_prob'] = y_fcst_test

        if self.is_valid:
            df_prob['is_sellin'] = self.df_test['is_sellin']

        #TODO
        self.logger.info('进货概率预测完毕')
        return df_prob, df_imp
    
    def get_prob_thr(self, df_prob, num_prophet=-1, is_auto=False):
        """进货概率划分阈值"""
        if is_auto:
        #     p1, p2, p3 = self._get_last3m_sellin_prop()
        #     avg_prop = round(np.mean([p1,p2,p3]), 2)
        #     thr_prob = sorted(df_prob['fcst_prob'].tolist(), reverse=True)[math.ceil(len(df_prob)*avg_prop)]
            thr_prob = sorted(df_prob['fcst_prob'].tolist(), reverse=True)[num_prophet]
        else:
            thr_prob = 0.5
        return thr_prob
    
    def get_is_sellin_label(self, df_prob, cut_thr):
        """获取是否进货标签 0,1"""
        df_prob['fcst_label'] = 0
        df_prob.loc[df_prob['fcst_prob']>=cut_thr, 'fcst_label'] = 1
        #! MTD: 预测为0但MTD有量，则改为1
        if self.is_mtd:
            df_prob.loc[(df_prob['mtd_qty']>0) & (df_prob['fcst_label']==0), 'fcst_label'] = 1
        if self.is_valid:
            self.logger.info("Test Accuracy: {:.2f}".format(round(accuracy_score(df_prob['is_sellin'].values, df_prob['fcst_label'].values)*100, 2)))
            fig_cfs_matrix = ConfusionMatrixDisplay.from_predictions(df_prob['is_sellin'].values, df_prob['fcst_label'].values)
        self.logger.info('是否进货打标完毕')
        return df_prob

    # ==================== 计算方法 ====================
    # def _get_single_sellin_prop(self, months_back):
    #     """单月进货终端比例"""
    #     ym_last = self._get_previous_month(self.TAR_YM, months_back)
    #     ym_last_12m = self._get_previous_month(ym_last, 12)
    #     df_tar = self.df_no_returns[self.df_no_returns['bizym']==ym_last]
    #     inst_appear = self.df_no_returns[(self.df_no_returns['bizym']>=ym_last_12m) & (self.df_no_returns['bizym']<ym_last)]['tomdmcode'].unique()
    #     num_hos_ttl = len(inst_appear)
    #     num_hos_appear = df_tar[df_tar['tomdmcode'].isin(inst_appear)]['tomdmcode'].nunique()
    #     return num_hos_appear, num_hos_ttl, num_hos_appear / num_hos_ttl

    # def _get_last3m_sellin_prop(self):
    #     """过往N个月进货终端比例"""
    #     appear_l1, ttl_l1, prop_l1 = self._get_single_sellin_prop(months_back=1)
    #     appear_l2, ttl_l2, prop_l2 = self._get_single_sellin_prop(months_back=2)
    #     appear_l3, ttl_l3, prop_l3 = self._get_single_sellin_prop(months_back=3)
    #     dict_num = dict(zip(["T-1月进货医院数量","T-1月近一年有进货医院数量","T-1月进货医院占比","T-2月进货医院数量","T-2月近一年有进货医院数量","T-2月进货医院占比","T-3月进货医院数量","T-3月近一年有进货医院数量","T-3月进货医院占比"],[appear_l1, ttl_l1, round(prop_l1,3), appear_l2, ttl_l2, round(prop_l2,3), appear_l3, ttl_l3, round(prop_l3,3)]))
    #     return prop_l1, prop_l2, prop_l3
    
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
    
    def _predefined_split(self):
        ''' Sklearn PredefinedSplit '''
        valid_ym = self._get_previous_month(self.TAR_YM, 1)
        test_fold = np.array([-1] * len(self.df_train))
        test_fold[self.df_train[self.df_train['bizym']==valid_ym].index.values] = 0
        ps = PredefinedSplit(test_fold)
        return ps
        