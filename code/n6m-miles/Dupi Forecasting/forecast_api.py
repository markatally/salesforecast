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

from feature_builder import process_price_elasticity_162, process_price_elasticity_169, create_holistic_samples_162_recent
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
       
        self._feature_stats = model_data.get('train_stats') or model_data.get('feature_stats', {})
        
       
        if 'optimal_w_mom' in model_data:
            self._optimal_weights = {
                'w_mom': model_data['optimal_w_mom'],
                'w_abs': model_data['optimal_w_abs']
            }
        else:
            self._optimal_weights = model_data.get('optimal_weights', {
                'w_mom': 0.5,
                'w_abs': 0.5
            })
        
        self._is_initialized = True
    
    def predict(self, features: Dict, base_sale: float, future_months: List[pd.Timestamp]) -> pd.DataFrame:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
        
        pred_X = pd.DataFrame([features])
        pred_X_filled = self._fill_missing_values(pred_X)
        
        # Handle both StandardScaler object and dict format
        if isinstance(self._scaler, dict):
            # New format: dict with mean and scale
            pred_X_scaled = (pred_X_filled - self._scaler['mean']) / self._scaler['scale']
        else:
            # Old format: StandardScaler object
            pred_X_scaled = self._scaler.transform(pred_X_filled)
        
        # Handle both model object list and model params dict list
        if isinstance(self._models_abs[0], dict):
            # New format: list of dicts with coef and intercept
            pred_abs = []
            for model_param in self._models_abs:
                pred_val = np.dot(pred_X_scaled.values[0], model_param['coef']) + model_param['intercept']
                pred_abs.append(pred_val)
            pred_abs = np.array(pred_abs)
            
            pred_mom_rates = []
            for model_param in self._models_mom:
                pred_val = np.dot(pred_X_scaled.values[0], model_param['coef']) + model_param['intercept']
                pred_mom_rates.append(pred_val)
            pred_mom_rates = np.array(pred_mom_rates)
        else:
            # Old format: list of trained model objects
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
    """
    Sales Forecast API for Dupixent (达必妥)
    Supports:
    - Spec 162 (300mg): Ridge regression model
    - Spec 169 (200mg): Rule-based method
    """
    def __init__(self):
        # Dupixent models
        self.predictor_162 = SalesPredictor()
        self._rule_params_169 = None  
        self._is_ready_162 = False
        self._is_ready_169 = False
    
    def setup(self, model_path_162: str = None, model_path_169: str = None):
        """
        Setup forecast models
        
        Args:
            model_path_162: Path to Dupixent 300mg model file
            model_path_169: Path to Dupixent 200mg rule parameters file
        """
        # Dupixent 162 (300mg)
        if model_path_162:
            self.predictor_162.load_pretrained_models(model_path_162)
            self._is_ready_162 = True
        
        # Dupixent 169 (200mg)
        if model_path_169:
            with open(model_path_169, 'rb') as f:
                self._rule_params_169 = pickle.load(f)
            self._is_ready_169 = True
    
    def forecast_162_from_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast for spec 162 - Use pretrained model if available
        """
        if not self._is_ready_162:
            raise RuntimeError("Spec 162 model not ready")
        
        df = df_raw.copy()
        
        # Filter spec 162 data (support prodmdmcode and spec columns)
        if 'prodmdmcode' in df.columns:
            df['prodmdmcode'] = df['prodmdmcode'].astype(str)
            df = df[df['prodmdmcode'].str.contains('162', na=False)].copy()
            if df.empty:
                raise ValueError("No records found for spec 162 in raw data (prodmdmcode)")
        elif 'spec' in df.columns:
            df = df[df['spec'] == '162'].copy()
            if df.empty:
                raise ValueError("No records found for spec 162 in raw data (spec)")
        elif '规格' in df.columns:
            df = df[df['规格'] == '162'].copy()
            if df.empty:
                raise ValueError("No records found for spec 162 in raw data (spec)")
        else:
            raise ValueError("Raw data must contain 'prodmdmcode', 'spec' or '规格' column")
        
        if 'bizym' not in df.columns:
            raise ValueError("Raw data must contain 'bizym' column")
        
        qty_col = 'cnvrtdqty' if 'cnvrtdqty' in df.columns else 'qty'
        if qty_col not in df.columns:
            raise ValueError(f"Raw data must contain '{qty_col}' column for spec 162")
        
        df['date_ym'] = pd.to_datetime(df['bizym'], format='%Y%m')
        df['qty'] = pd.to_numeric(df[qty_col], errors='coerce')
        
        if 'tophncode' not in df.columns:
            df['tophncode'] = 'UNKNOWN'
        
        df_qty = (
            df.groupby('date_ym', as_index=False)
            .agg(sales=('qty', 'sum'))
            .sort_values('date_ym')
        )
        df_active = (
            df.groupby('date_ym')['tophncode']
            .nunique()
            .reset_index()
            .rename(columns={'tophncode': 'active_hosp'})
        )
        df_monthly = df_qty.merge(df_active, on='date_ym', how='left')
                
        # Price elasticity processing
        df_nat, interval_multipliers, _, _ = process_price_elasticity_162(df_monthly)
                
        # Use English column names and filter to training period
        df_nat = df_nat[df_nat['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
        df_nat = df_nat.sort_values('date_ym')
                
        # Prepare df_monthly with English columns
        df_monthly_en = df_nat[['date_ym', 'baseline_sales', 'price_interval_id']].copy()
        df_monthly_en = df_monthly_en.rename(columns={'baseline_sales': 'sales'})
        df_monthly_en['date_ym'] = pd.to_datetime(df_monthly_en['date_ym'])
                
        # Calculate mom_rate and month
        df_monthly_en['mom_rate'] = df_monthly_en['sales'].pct_change() * 100
        df_monthly_en['month'] = df_monthly_en['date_ym'].dt.month
        
        # Generate prediction features
        pred_date = pd.Timestamp('2025-10-01')
        features_pred, _, _, future_months = create_holistic_samples_162_recent(df_monthly_en, pred_date)
        
        base_sale = df_monthly_en[df_monthly_en['date_ym'] == pred_date]['sales'].values[0]
        
        # Use pretrained model from predictor_162
        result = self.predictor_162.predict(features_pred, base_sale, future_months)
        
        # Restore actual sales using interval_multipliers
        current_price_mult = interval_multipliers[max(interval_multipliers.keys())]
        result['pred'] = result['pred'] * current_price_mult
        
        result = result[['month', 'pred']]
        return result
    
    def forecast_169_from_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast for spec 169 - Use rule-based method with pretrained parameters
        """
        if not self._is_ready_169:
            raise RuntimeError("Spec 169 model not ready")
        
        df = df_raw.copy()
        
        if 'prodmdmcode' in df.columns:
            df['prodmdmcode'] = df['prodmdmcode'].astype(str)
            df = df[df['prodmdmcode'].str.contains('169', na=False)].copy()
            if df.empty:
                raise ValueError("No records found for spec 169 in raw data")
        
        if 'bizym' not in df.columns:
            raise ValueError("Raw data must contain 'bizym' column")
        
        qty_col = 'cnvrtdqty' if 'cnvrtdqty' in df.columns else 'qty'
        if qty_col not in df.columns:
            raise ValueError(f"Raw data must contain '{qty_col}' column for spec 169")
        
        df['date_ym'] = pd.to_datetime(df['bizym'], format='%Y%m')
        df['qty'] = pd.to_numeric(df[qty_col], errors='coerce')
        
        if 'tophncode' not in df.columns:
            df['tophncode'] = 'UNKNOWN'
        
        df_qty = (
            df.groupby('date_ym', as_index=False)
            .agg(sales=('qty', 'sum'))
            .sort_values('date_ym')
        )
        df_active = (
            df.groupby('date_ym')['tophncode']
            .nunique()
            .reset_index()
            .rename(columns={'tophncode': 'active_hosp'})
        )
        df_monthly = df_qty.merge(df_active, on='date_ym', how='left')
        
        # Price elasticity processing
        df_nat, interval_multipliers, _, _ = process_price_elasticity_169(df_monthly)
        
        # Get current price multiplier
        current_price_mult = interval_multipliers[max(interval_multipliers.keys())]
        
        # ============================================================
        # Use pretrained rule parameters
        # ============================================================
        params = self._rule_params_169
        
        pred_date = pd.Timestamp('2025-10-01')
        future_months = pd.date_range(pred_date + pd.DateOffset(months=1), periods=6, freq='MS')
        
        # Get base sale from current data (2025-10)
        base_sale_oct = df_nat[df_nat['date_ym'] == pred_date]['baseline_sales'].values
        base_sale_oct = base_sale_oct[0] if len(base_sale_oct) > 0 else params['base_sale_oct']
        
        # ============================================================
        # Generate 2025-11, 2025-12 using historical ratios
        # ============================================================
        pred_25_base = np.zeros(2)
        pred_25_base[0] = base_sale_oct * params['ratio_24_11_vs_10']
        pred_25_base[1] = pred_25_base[0] * params['ratio_24_12_vs_11']
        
        # ============================================================
        # Generate 2026-01~04 using rules + intercept constraint
        # ============================================================
        pred_26_base = np.zeros(4)
        pred_26_base[0] = pred_25_base[1] * params['rule_26_01_mult']
        pred_26_base[1] = pred_26_base[0] * params['rule_26_02_mult']
        pred_26_base[2] = pred_26_base[1] * params['rule_26_03_mult']
        pred_26_base[3] = pred_26_base[2] * params['avg_apr_vs_mar']
        
        # ============================================================

        full_6months_actual = np.concatenate([
            pred_25_base * current_price_mult,
            pred_26_base * current_price_mult
        ])
        
        temp_seq = np.arange(6)
        slope_temp, intercept_temp, _, _, _ = stats.linregress(temp_seq, full_6months_actual)
        
        adjustment = params['target_intercept_26'] - intercept_temp
        pred_26_actual_adjusted = pred_26_base * current_price_mult + adjustment
        
        pred_26_base_final = pred_26_actual_adjusted / current_price_mult
        
        # ============================================================
        # Merge final result
        # ============================================================
        pred_ensemble_base_169_final = np.concatenate([pred_25_base, pred_26_base_final])
        pred_actual = pred_ensemble_base_169_final * current_price_mult
        
        result = pd.DataFrame({
            'month': [m.strftime('%Y-%m') for m in future_months],
            'pred': pred_actual,
        })
        return result
    
    def get_info(self) -> dict:
        return {
            'version': '2.0',
            'forecast_period': '6_months',
            'models': {
                'Spec_162': self.predictor_162.get_model_info() if self._is_ready_162 else 'not_ready',
                'Spec_169': {'type': 'rule_based', 'status': 'ready' if self._is_ready_169 else 'not_ready'}
            }
        }
