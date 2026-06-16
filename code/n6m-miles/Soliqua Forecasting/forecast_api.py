"""Sales forecast API"""

import pandas as pd
import numpy as np
import pickle
import base64
import zlib
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy import stats

from feature_builder import FeatureBuilder, create_holistic_sales_samples_v3_171
class SalesPredictor:
    def __init__(self):
        self._models_abs = None
        self._models_mom = None
        self._scaler = None
        self._feature_stats = None
        self._optimal_weights = None
        self._is_initialized = False
    
    def initialize(self):
        self._models_abs = []
        for i in range(6):
            model = Ridge(alpha=5.0, random_state=42)
            self._models_abs.append(model)
        
        self._models_mom = []
        for i in range(6):
            model = Ridge(alpha=5.0, random_state=42)
            self._models_mom.append(model)
        
        self._scaler = StandardScaler()
        self._optimal_weights = {
            'w_mom': 0.7,
            'w_abs': 0.3
        }
        
        self._feature_stats = {}
        
        self._is_initialized = True
    
    def load_pretrained_models(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self._models_abs = model_data['models_abs']
        self._models_mom = model_data['models_mom']
        self._scaler = model_data['scaler']
        self._feature_stats = model_data['feature_stats']
        # 固定与notebook一致的权重
        self._optimal_weights = {
            'w_mom': 0.5,
            'w_abs': 0.5
        }
        
        self._is_initialized = True
    
    def predict(self, features: Dict, base_sale: float, future_months: List[pd.Timestamp]) -> pd.DataFrame:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
        
        pred_X = pd.DataFrame([features])
        pred_X_filled = self._fill_missing_values(pred_X)
        pred_X_scaled = self._scaler.transform(pred_X_filled)
        
        pred_abs = np.array([model.predict(pred_X_scaled)[0] for model in self._models_abs])
        pred_mom_rates = np.array([model.predict(pred_X_scaled)[0] for model in self._models_mom])
        
        pred_from_mom = []
        current_sale = base_sale
        for mom_rate in pred_mom_rates:
            next_sale = current_sale * (1 + mom_rate / 100)
            pred_from_mom.append(next_sale)
            current_sale = next_sale
        pred_from_mom = np.array(pred_from_mom)
        
        w_mom = self._optimal_weights['w_mom']
        w_abs = self._optimal_weights['w_abs']
        pred_ensemble = w_mom * pred_from_mom + w_abs * pred_abs
        
        mom_pcts = []
        prev_sale = base_sale
        for sale in pred_ensemble:
            mom_pct = (sale - prev_sale) / prev_sale * 100
            mom_pcts.append(mom_pct)
            prev_sale = sale
        
        result = pd.DataFrame({
            'month': [m.strftime('%Y-%m') for m in future_months],
            'pred_abs': pred_abs,
            'pred_mom': pred_from_mom,
            'pred': pred_ensemble,
            'mom_rate': mom_pcts
        })
        
        return result
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isna().all():
                df_filled[col] = 0
            elif 'mom' in col:
                if col in self._feature_stats:
                    df_filled[col] = df_filled[col].fillna(self._feature_stats[col]['median'])
                else:
                    df_filled[col] = df_filled[col].fillna(0)
            elif 'slope' in col or 'r2' in col:
                df_filled[col] = df_filled[col].fillna(0)
            else:
                if col in self._feature_stats:
                    df_filled[col] = df_filled[col].fillna(self._feature_stats[col]['mean'])
                else:
                    df_filled[col] = df_filled[col].fillna(0)
        
        return df_filled
    
    def get_model_info(self) -> Dict:
        return {
            'type': 'Ridge',
            'method': 'Hybrid',
            'weights': self._optimal_weights,
            'n_features': len(self._feature_stats) if self._feature_stats else 0,
            'status': 'ready' if self._is_initialized else 'not_ready'
        }

class SalesForecastAPI:
    def __init__(self):
        self.feature_builder = FeatureBuilder()
        self.predictor = SalesPredictor()
        self.predictor_171 = SalesPredictor()
        self._is_ready = False
        self._is_ready_171 = False
    
    def setup(self, model_path_170: str = None, model_path_171: str = None):
        if model_path_170:
            self.predictor.load_pretrained_models(model_path_170)
        else:
            self.predictor.initialize()
        self._is_ready = True

        if model_path_171:
            self.predictor_171.load_pretrained_models(model_path_171)
            self._is_ready_171 = True
    
    def forecast(self, df_sales: pd.DataFrame) -> pd.DataFrame:
        if not self._is_ready:
            raise RuntimeError("Model not ready")
        
        df_monthly = self._prepare_data(df_sales)
        target_date = pd.Timestamp('2025-10-01')
        features, future_months = self.feature_builder.build_features(df_monthly, target_date)
        
        base_sale = df_monthly[df_monthly['date_ym'] == target_date]['sales'].values
        if len(base_sale) == 0:
            raise ValueError(f"Data not found for {target_date.strftime('%Y-%m')}")
        base_sale = base_sale[0]
        result = self.predictor.predict(features, base_sale, future_months)
        result = result[['month', 'pred']]
        
        return result
    
    def forecast_from_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df['prodmdmcode'] = df['prodmdmcode'].astype(str)
        df = df[df['prodmdmcode'].str.contains('170', na=False)].copy()
        df_sales = pd.DataFrame()
        df_sales['date_ym'] = pd.to_datetime(df['bizym'], format='%Y%m')
        df_sales['qty'] = pd.to_numeric(df['cnvrtdqty'], errors='coerce')
        return self.forecast(df_sales=df_sales)

    def forecast_171_from_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if not self._is_ready_171:
            raise RuntimeError("Spec 171 model not ready")

        df = df_raw.copy()
        df['prodmdmcode'] = df['prodmdmcode'].astype(str)
        df = df[df['prodmdmcode'].str.contains('171', na=False)].copy()
        if df.empty:
            raise ValueError("No records found for spec 171 in raw data")
        if 'bizym' not in df.columns or 'cnvrtdqty' not in df.columns:
            raise ValueError("Raw data must contain 'bizym' and 'cnvrtdqty' columns for spec 171")

        df['date_ym'] = pd.to_datetime(df['bizym'], format='%Y%m')
        df['qty'] = pd.to_numeric(df['cnvrtdqty'], errors='coerce')

        if 'tophncode' not in df.columns:
            df['tophncode'] = 'UNKNOWN'

        df_qty = (
            df.groupby('date_ym', as_index=False)
            .agg(sales=('qty', 'sum'))
            .sort_values('date_ym')
        )
        df_active = (
            df[df['qty'] > 0]
            .groupby('date_ym')['tophncode']
            .nunique()
            .reset_index()
            .rename(columns={'tophncode': 'active_hosp'})
        )
        df_monthly = df_qty.merge(df_active, on='date_ym', how='left')
        df_monthly = df_monthly[df_monthly['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
        df_monthly = df_monthly.sort_values('date_ym').reset_index(drop=True)
        df_monthly['month'] = df_monthly['date_ym'].dt.month
        df_monthly['year'] = df_monthly['date_ym'].dt.year

        hist_yoy: List[float] = []
        for i in range(1, len(df_monthly)):
            curr = df_monthly.iloc[i]
            prev = df_monthly[
                (df_monthly['year'] == curr['year'] - 1)
                & (df_monthly['month'] == curr['month'])
            ]
            if len(prev) > 0 and prev.iloc[0]['sales'] > 0:
                yoy = (curr['sales'] - prev.iloc[0]['sales']) / prev.iloc[0]['sales'] * 100
                if curr['year'] < 2026:
                    hist_yoy.append(yoy)

        target_yoy_mean = float(np.mean(hist_yoy)) if hist_yoy else 10.0

        pred_date = pd.Timestamp('2025-10-01')
        features_pred, _, _, future_months = create_holistic_sales_samples_v3_171(
            df_monthly, pred_date, target_yoy_mean
        )

        base_sale_arr = df_monthly[df_monthly['date_ym'] == pred_date]['sales'].values
        if len(base_sale_arr) == 0:
            raise ValueError("Base sale for 2025-10 not found for spec 171")
        base_sale = float(base_sale_arr[0])

        pred_X = pd.DataFrame([features_pred])
        pred_X_filled = self.predictor_171._fill_missing_values(pred_X)
        pred_X_scaled = self.predictor_171._scaler.transform(pred_X_filled)

        pred_abs = np.array([model.predict(pred_X_scaled)[0] for model in self.predictor_171._models_abs])
        pred_mom_rates = np.array([model.predict(pred_X_scaled)[0] for model in self.predictor_171._models_mom])

        max_mom = 6.0
        spring_festival_months = {2024: 2, 2025: 1, 2026: 2}
        for i in range(len(pred_mom_rates)):
            if pred_mom_rates[i] > max_mom:
                pred_mom_rates[i] = max_mom

        for i, month in enumerate(future_months):
            year = month.year
            month_num = month.month
            if spring_festival_months.get(year) == month_num:
                pred_mom_rates[i] = -16.0

        pred_from_mom = []
        current_sale = base_sale
        for mom_rate in pred_mom_rates:
            next_sale = current_sale * (1 + mom_rate / 100)
            pred_from_mom.append(next_sale)
            current_sale = next_sale
        pred_from_mom = np.array(pred_from_mom)

        pred_yoy_constrained = []
        for month in future_months:
            last_year = month - pd.DateOffset(years=1)
            last_year_sale = df_monthly[
                (df_monthly['year'] == last_year.year)
                & (df_monthly['month'] == last_year.month)
            ]['sales'].values
            if len(last_year_sale) > 0 and last_year_sale[0] > 0:
                target_sale_yoy = last_year_sale[0] * (1 + target_yoy_mean / 100)
                pred_yoy_constrained.append(target_sale_yoy)
            else:
                pred_yoy_constrained.append(np.nan)
        pred_yoy_constrained = np.array(pred_yoy_constrained)

        w_mom = 0.2
        w_abs = 0.1
        w_yoy = 0.7

        pred_ensemble = np.zeros(6)
        for i in range(6):
            if not np.isnan(pred_yoy_constrained[i]):
                pred_ensemble[i] = (
                    w_mom * pred_from_mom[i]
                    + w_abs * pred_abs[i]
                    + w_yoy * pred_yoy_constrained[i]
                )
            else:
                pred_ensemble[i] = (w_mom / (w_mom + w_abs)) * pred_from_mom[i] + (
                    w_abs / (w_mom + w_abs)
                ) * pred_abs[i]

        recent_3m = df_monthly[
            (df_monthly['date_ym'] >= pd.Timestamp('2025-08-01'))
            & (df_monthly['date_ym'] <= pd.Timestamp('2025-10-01'))
        ]['sales'].values
        if len(recent_3m) > 0:
            recent_avg = float(np.mean(recent_3m))
            pred_avg = float(np.mean(pred_ensemble))
            if pred_avg > recent_avg * 1.10:
                factor = (recent_avg * 1.05) / pred_avg
                pred_ensemble = pred_ensemble * factor

        idx_nov = None
        idx_dec = None
        for i, m in enumerate(future_months):
            if m.year == 2025 and m.month == 11:
                idx_nov = i
            elif m.year == 2025 and m.month == 12:
                idx_dec = i
        if idx_nov is not None and idx_dec is not None:
            if pred_ensemble[idx_dec] >= pred_ensemble[idx_nov]:
                mid = (pred_ensemble[idx_nov] + pred_ensemble[idx_dec]) / 2.0
                pred_ensemble[idx_nov] = mid * 1.03
                pred_ensemble[idx_dec] = mid / 1.03

        result = pd.DataFrame({
            'month': [m.strftime('%Y-%m') for m in future_months],
            'pred': pred_ensemble,
        })
        return result
    
    def _prepare_data(self, df_sales: pd.DataFrame) -> pd.DataFrame:
        df = df_sales.copy()
        if 'date_ym' not in df.columns or 'qty' not in df.columns:
            raise ValueError("Input data must contain 'date_ym' and 'qty' columns")
        
        df['date_ym'] = pd.to_datetime(df['date_ym'])
        
        df_monthly = (
            df.groupby('date_ym', as_index=False)
            .agg(sales=('qty', 'sum'))
            .sort_values('date_ym')
            .reset_index(drop=True)
        )
        df_monthly = df_monthly[df_monthly['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
        df_monthly['mom_rate'] = df_monthly['sales'].pct_change() * 100
        df_monthly['month'] = df_monthly['date_ym'].dt.month
        
        return df_monthly
    
    def get_info(self) -> dict:
        return {
            'version': '1.0',
            'product': 'Spec_170',
            'forecast_period': '6_months',
            'model_info': self.predictor.get_model_info() if self._is_ready else 'not_ready'
        }
