"""Feature extraction module"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    def __init__(self):
        self.feature_names = None
    
    def build_features(self, df_monthly: pd.DataFrame, target_date: pd.Timestamp) -> Tuple[Dict, List[pd.Timestamp]]:
        future_months = pd.date_range(target_date + pd.DateOffset(months=1), periods=6, freq='MS')
        df_historical = df_monthly[df_monthly['date_ym'] <= target_date].copy()
        same_period_last_year = pd.date_range(target_date - pd.DateOffset(months=11), periods=6, freq='MS')
        
        sales_last_year = []
        mom_last_year = []
        
        for month_ly in same_period_last_year:
            sale_ly = df_historical[df_historical['date_ym'] == month_ly]['sales'].values
            mom_ly = df_historical[df_historical['date_ym'] == month_ly]['mom_rate'].values
            
            sales_last_year.append(sale_ly[0] if len(sale_ly) > 0 else np.nan)
            mom_last_year.append(mom_ly[0] if len(mom_ly) > 0 else np.nan)
        
        recent_6_months = pd.date_range(target_date - pd.DateOffset(months=5), periods=6, freq='MS')
        
        sales_recent = []
        mom_recent = []
        
        for month_curr in recent_6_months:
            sale_curr = df_historical[df_historical['date_ym'] == month_curr]['sales'].values
            mom_curr = df_historical[df_historical['date_ym'] == month_curr]['mom_rate'].values
            
            sales_recent.append(sale_curr[0] if len(sale_curr) > 0 else np.nan)
            mom_recent.append(mom_curr[0] if len(mom_curr) > 0 else np.nan)
        
        future_month_nums = [m.month for m in future_months]
        hist_mom_mean = []
        hist_mom_median = []
        
        for month_num in future_month_nums:
            hist_data = df_historical[df_historical['month'] == month_num]['mom_rate'].dropna()
            hist_mom_mean.append(hist_data.mean() if len(hist_data) > 0 else np.nan)
            hist_mom_median.append(hist_data.median() if len(hist_data) > 0 else np.nan)
        
        slope_ly, r_ly = self._calc_trend(sales_last_year)
        slope_rec, r_rec = self._calc_trend(sales_recent)
        base_sale = df_historical[df_historical['date_ym'] == target_date]['sales'].values
        base_sale = base_sale[0] if len(base_sale) > 0 else np.nan
        
        features = {
            'ly_avg': np.nanmean(sales_last_year),
            'ly_sum': np.nansum(sales_last_year),
            'ly_std': np.nanstd(sales_last_year),
            'ly_mom_avg': np.nanmean(mom_last_year),
            'ly_mom_med': np.nanmedian(mom_last_year),
            'ly_slope': slope_ly,
            'ly_r2': r_ly,
            'rec_avg': np.nanmean(sales_recent),
            'rec_sum': np.nansum(sales_recent),
            'rec_std': np.nanstd(sales_recent),
            'rec_mom_avg': np.nanmean(mom_recent),
            'rec_mom_med': np.nanmedian(mom_recent),
            'rec_slope': slope_rec,
            'rec_r2': r_rec,
            'base_val': base_sale,
            'fut_start': future_month_nums[0],
            'fut_end': future_month_nums[-1],
        }
        
        for i in range(6):
            features[f'ly_M{i+1}_val'] = sales_last_year[i]
            features[f'ly_M{i+1}_mom'] = mom_last_year[i]
            features[f'rec_M{i+1}_val'] = sales_recent[i]
            features[f'rec_M{i+1}_mom'] = mom_recent[i]
            features[f'fut_M{i+1}_hmean'] = hist_mom_mean[i]
            features[f'fut_M{i+1}_hmed'] = hist_mom_median[i]
            month_num = future_month_nums[i]
            for m in range(1, 13):
                features[f'fut_M{i+1}_is_{self._month_abbr(m)}'] = 1 if month_num == m else 0
        
        return features, future_months
    
    def _calc_trend(self, sales_list: List[float]) -> Tuple[float, float]:
        if not all(np.isnan(sales_list)):
            valid_idx = [i for i, x in enumerate(sales_list) if not np.isnan(x)]
            if len(valid_idx) >= 2:
                x = np.array(valid_idx)
                y = np.array([sales_list[i] for i in valid_idx])
                slope, _, r_value, _, _ = stats.linregress(x, y)
                return slope, r_value
        return np.nan, np.nan
    
    def _month_abbr(self, month: int) -> str:
        abbrs = {
            1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
            7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
        }
        return abbrs.get(month, '')


def create_holistic_samples_v5(df, target_date):
    """Create prediction samples for Spec 170"""
    future_months = pd.date_range(target_date + pd.DateOffset(months=1), periods=6, freq='MS')
    df_historical = df[df['date_ym'] <= target_date].copy()
    
    same_period_last_year = pd.date_range(target_date - pd.DateOffset(months=11), periods=6, freq='MS')
    
    sales_last_year = []
    mom_last_year = []
    
    for i in range(6):
        month_ly = same_period_last_year[i]
        sale_ly = df_historical[df_historical['date_ym'] == month_ly]['sales'].values
        mom_ly = df_historical[df_historical['date_ym'] == month_ly]['mom_rate'].values
        
        if len(sale_ly) > 0:
            sales_last_year.append(sale_ly[0])
            if len(mom_ly) > 0:
                mom_last_year.append(mom_ly[0])
            else:
                mom_last_year.append(np.nan)
        else:
            sales_last_year.append(np.nan)
            mom_last_year.append(np.nan)
    
    recent_6_months = pd.date_range(target_date - pd.DateOffset(months=5), periods=6, freq='MS')
    sales_recent = []
    mom_recent = []
    
    for i in range(6):
        month_curr = recent_6_months[i]
        sale_curr = df_historical[df_historical['date_ym'] == month_curr]['sales'].values
        mom_curr = df_historical[df_historical['date_ym'] == month_curr]['mom_rate'].values
        
        if len(sale_curr) > 0:
            sales_recent.append(sale_curr[0])
            if len(mom_curr) > 0:
                mom_recent.append(mom_curr[0])
            else:
                mom_recent.append(np.nan)
        else:
            sales_recent.append(np.nan)
            mom_recent.append(np.nan)
    
    future_month_nums = [m.month for m in future_months]
    hist_mom_mean = []
    hist_mom_median = []
    
    for month_num in future_month_nums:
        hist_data = df_historical[df_historical['month'] == month_num]['mom_rate'].dropna()
        if len(hist_data) > 0:
            hist_mom_mean.append(hist_data.mean())
            hist_mom_median.append(hist_data.median())
        else:
            hist_mom_mean.append(np.nan)
            hist_mom_median.append(np.nan)
    
    if not all(np.isnan(sales_last_year)):
        valid_idx = [i for i, x in enumerate(sales_last_year) if not np.isnan(x)]
        if len(valid_idx) >= 2:
            x_ly = np.array(valid_idx)
            y_ly = np.array([sales_last_year[i] for i in valid_idx])
            slope_ly, _, r_ly, _, _ = stats.linregress(x_ly, y_ly)
        else:
            slope_ly, r_ly = np.nan, np.nan
    else:
        slope_ly, r_ly = np.nan, np.nan
    
    if not all(np.isnan(sales_recent)):
        valid_idx = [i for i, x in enumerate(sales_recent) if not np.isnan(x)]
        if len(valid_idx) >= 2:
            x_rec = np.array(valid_idx)
            y_rec = np.array([sales_recent[i] for i in valid_idx])
            slope_rec, _, r_rec, _, _ = stats.linregress(x_rec, y_rec)
        else:
            slope_rec, r_rec = np.nan, np.nan
    else:
        slope_rec, r_rec = np.nan, np.nan
    
    base_sale = df_historical[df_historical['date_ym'] == target_date]['sales'].values
    base_sale = base_sale[0] if len(base_sale) > 0 else np.nan
    
    features = {
        'ly_avg': np.nanmean(sales_last_year),
        'ly_sum': np.nansum(sales_last_year),
        'ly_std': np.nanstd(sales_last_year),
        'ly_mom_avg': np.nanmean(mom_last_year),
        'ly_mom_med': np.nanmedian(mom_last_year),
        'ly_slope': slope_ly,
        'ly_r2': r_ly,
        
        'rec_avg': np.nanmean(sales_recent),
        'rec_sum': np.nansum(sales_recent),
        'rec_std': np.nanstd(sales_recent),
        'rec_mom_avg': np.nanmean(mom_recent),
        'rec_mom_med': np.nanmedian(mom_recent),
        'rec_slope': slope_rec,
        'rec_r2': r_rec,
        
        'base_val': base_sale,
        
        'fut_start': future_month_nums[0],
        'fut_end': future_month_nums[-1],
    }
    
    for i in range(6):
        features[f'ly_M{i+1}_val'] = sales_last_year[i]
        features[f'ly_M{i+1}_mom'] = mom_last_year[i]
        features[f'rec_M{i+1}_val'] = sales_recent[i]
        features[f'rec_M{i+1}_mom'] = mom_recent[i]
        features[f'fut_M{i+1}_hmean'] = hist_mom_mean[i]
        features[f'fut_M{i+1}_hmed'] = hist_mom_median[i]
        
        month_num = future_month_nums[i]
        month_abbrs = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        for m in range(1, 13):
            features[f'fut_M{i+1}_is_{month_abbrs[m-1]}'] = 1 if month_num == m else 0
    
    targets_abs = []
    for m in future_months:
        sale = df[df['date_ym'] == m]['sales'].values
        targets_abs.append(sale[0] if len(sale) > 0 else np.nan)
    
    targets_mom = []
    prev_sale = base_sale
    for i, sale in enumerate(targets_abs):
        if not np.isnan(sale) and not np.isnan(prev_sale) and prev_sale > 0:
            mom = (sale - prev_sale) / prev_sale * 100
            targets_mom.append(mom)
            prev_sale = sale
        else:
            targets_mom.append(np.nan)
    
    return features, targets_abs, targets_mom, future_months


def export_trained_model(df_all, output_path='trained_model_170.pkl'):
    """Export trained model for Spec 170"""
    df_170 = df_all[df_all['规格'] == '170'].copy()
    df_170['date_ym'] = pd.to_datetime(df_170['年月'])
    
    df_monthly = (
        df_170.groupby('date_ym', as_index=False)
        .agg(sales=('qty', 'sum'))
        .sort_values('date_ym')
        .reset_index(drop=True)
    )
    
    df_monthly = df_monthly[df_monthly['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
    df_monthly['mom_rate'] = df_monthly['sales'].pct_change() * 100
    df_monthly['month'] = df_monthly['date_ym'].dt.month
    
    train_start = pd.Timestamp('2024-06-01')
    train_end = pd.Timestamp('2025-04-01')
    train_dates = pd.date_range(train_start, train_end, freq='MS')
    
    train_samples = []
    train_targets_abs = []
    train_targets_mom = []
    
    for date in train_dates:
        features, targets_abs, targets_mom, _ = create_holistic_samples_v5(df_monthly, date)
        if not any(np.isnan(targets_abs)) and not any(np.isnan(targets_mom)):
            train_samples.append(features)
            train_targets_abs.append(targets_abs)
            train_targets_mom.append(targets_mom)
    
    train_X = pd.DataFrame(train_samples)
    train_y_abs = np.array(train_targets_abs)
    train_y_mom = np.array(train_targets_mom)
    
    train_X_filled = train_X.copy()
    feature_stats = {}
    
    for col in train_X_filled.columns:
        if train_X_filled[col].isna().all():
            train_X_filled[col] = 0
        elif 'mom' in col:
            if train_X_filled[col].notna().any():
                col_median = train_X_filled[col].median()
                feature_stats[col] = {'median': col_median, 'mean': 0}
                train_X_filled[col] = train_X_filled[col].fillna(col_median)
            else:
                train_X_filled[col] = 0
        elif 'slope' in col or 'r2' in col:
            train_X_filled[col] = train_X_filled[col].fillna(0)
        else:
            if train_X_filled[col].notna().any():
                col_mean = train_X_filled[col].mean()
                feature_stats[col] = {'mean': col_mean, 'median': 0}
                train_X_filled[col] = train_X_filled[col].fillna(col_mean)
            else:
                train_X_filled[col] = 0
    
    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X_filled)
    
    models_abs = []
    for i in range(6):
        model = Ridge(alpha=3.0, random_state=42)
        model.fit(train_X_scaled, train_y_abs[:, i])
        models_abs.append(model)
    
    models_mom = []
    for i in range(6):
        model = Ridge(alpha=3.0, random_state=42)
        model.fit(train_X_scaled, train_y_mom[:, i])
        models_mom.append(model)
    
    optimal_weights = {
        'w_mom': 0.5,
        'w_abs': 0.5
    }
    
    model_data = {
        'models_abs': models_abs,
        'models_mom': models_mom,
        'scaler': scaler_X,
        'feature_stats': feature_stats,
        'optimal_weights': optimal_weights,
        'feature_columns': list(train_X_filled.columns),
        'train_info': {
            'n_samples': len(train_samples),
            'n_features': train_X.shape[1],
            'train_period': f"{train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')}"
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data


def create_holistic_sales_samples_v3_171(df_monthly, target_date, target_yoy_mean):
    """Create prediction samples for Spec 171"""
    future_months = pd.date_range(target_date + pd.DateOffset(months=1), periods=6, freq='MS')
    df_historical = df_monthly[df_monthly['date_ym'] <= target_date].copy()

    same_period_last_year = pd.date_range(target_date - pd.DateOffset(months=11), periods=6, freq='MS')
    sales_last_year = []
    hosp_last_year = []

    for month_ly in same_period_last_year:
        sale_ly = df_historical[df_historical['date_ym'] == month_ly]['sales'].values
        hosp_ly = df_historical[df_historical['date_ym'] == month_ly]['active_hosp'].values
        sales_last_year.append(sale_ly[0] if len(sale_ly) > 0 else np.nan)
        hosp_last_year.append(hosp_ly[0] if len(hosp_ly) > 0 else np.nan)

    recent_6_months = pd.date_range(target_date - pd.DateOffset(months=5), periods=6, freq='MS')
    sales_recent = []
    hosp_recent = []

    for month_curr in recent_6_months:
        sale_curr = df_historical[df_historical['date_ym'] == month_curr]['sales'].values
        hosp_curr = df_historical[df_historical['date_ym'] == month_curr]['active_hosp'].values
        sales_recent.append(sale_curr[0] if len(sale_curr) > 0 else np.nan)
        hosp_recent.append(hosp_curr[0] if len(hosp_curr) > 0 else np.nan)

    slope_ly = np.nan
    if not all(np.isnan(sales_last_year)):
        valid_idx = [i for i, x in enumerate(sales_last_year) if not np.isnan(x)]
        if len(valid_idx) >= 2:
            x_ly = np.array(valid_idx)
            y_ly = np.array([sales_last_year[i] for i in valid_idx])
            slope_ly, _, _, _, _ = stats.linregress(x_ly, y_ly)

    slope_rec = np.nan
    if not all(np.isnan(sales_recent)):
        valid_idx = [i for i, x in enumerate(sales_recent) if not np.isnan(x)]
        if len(valid_idx) >= 2:
            x_rec = np.array(valid_idx)
            y_rec = np.array([sales_recent[i] for i in valid_idx])
            slope_rec, _, _, _, _ = stats.linregress(x_rec, y_rec)

    slope_decay_factor = 1.0
    if not np.isnan(slope_ly) and not np.isnan(slope_rec) and abs(slope_ly) > 1:
        slope_decay_factor = slope_rec / slope_ly
        slope_decay_factor = np.clip(slope_decay_factor, 0.3, 2.0)

    base_sale_arr = df_historical[df_historical['date_ym'] == target_date]['sales'].values
    base_sale = base_sale_arr[0] if len(base_sale_arr) > 0 else np.nan

    base_hosp_arr = df_historical[df_historical['date_ym'] == target_date]['active_hosp'].values
    base_hosp = base_hosp_arr[0] if len(base_hosp_arr) > 0 else np.nan

    unit_prod_ly = [
        s / h if not np.isnan(s) and not np.isnan(h) and h > 0 else np.nan
        for s, h in zip(sales_last_year, hosp_last_year)
    ]
    unit_prod_rec = [
        s / h if not np.isnan(s) and not np.isnan(h) and h > 0 else np.nan
        for s, h in zip(sales_recent, hosp_recent)
    ]

    future_month_nums = [m.month for m in future_months]

    features = {
        'ly_sales_mean': np.nanmean(sales_last_year),
        'ly_sales_std': np.nanstd(sales_last_year),
        'ly_hosp_mean': np.nanmean(hosp_last_year),
        'ly_unit_sales_mean': np.nanmean(unit_prod_ly),
        'ly_slope': slope_ly,
        'rec_sales_mean': np.nanmean(sales_recent),
        'rec_sales_std': np.nanstd(sales_recent),
        'rec_hosp_mean': np.nanmean(hosp_recent),
        'rec_unit_sales_mean': np.nanmean(unit_prod_rec),
        'rec_slope': slope_rec,
        'base_sales': base_sale,
        'base_hosp': base_hosp,
        'base_unit_sales': base_sale / base_hosp if not np.isnan(base_sale) and not np.isnan(base_hosp) and base_hosp > 0 else np.nan,
        'fut_start': future_month_nums[0],
        'fut_end': future_month_nums[-1],
        'slope_decay_factor': slope_decay_factor,
        'target_yoy_mean': target_yoy_mean,
    }

    month_abbrs = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for i in range(6):
        features[f'ly_M{i+1}_sales'] = sales_last_year[i]
        features[f'ly_M{i+1}_hosp'] = hosp_last_year[i]
        features[f'rec_M{i+1}_sales'] = sales_recent[i]
        features[f'rec_M{i+1}_hosp'] = hosp_recent[i]

        month_num = future_month_nums[i]
        for m in range(1, 13):
            features[f'fut_M{i+1}_is_{month_abbrs[m-1]}'] = 1 if month_num == m else 0

    targets_abs = []
    for m in future_months:
        sale = df_monthly[df_monthly['date_ym'] == m]['sales'].values
        targets_abs.append(sale[0] if len(sale) > 0 else np.nan)

    targets_mom = []
    prev_sale = base_sale
    for sale in targets_abs:
        if not np.isnan(sale) and not np.isnan(prev_sale) and prev_sale > 0:
            mom = (sale - prev_sale) / prev_sale * 100
            targets_mom.append(mom)
            prev_sale = sale
        else:
            targets_mom.append(np.nan)

    return features, targets_abs, targets_mom, future_months


def export_trained_model_171(df_all, output_path='trained_model_171.pkl'):
    """Export trained model for Spec 171"""
    df_171 = df_all[df_all['规格'] == '171'].copy()
    if df_171.empty:
        raise ValueError('No data for spec 171')

    df_171['date_ym'] = pd.to_datetime(df_171['年月'])

    df_qty = (
        df_171.groupby('date_ym', as_index=False)
        .agg(sales=('qty', 'sum'))
        .sort_values('date_ym')
    )

    if 'tophncode' in df_171.columns:
        df_active = (
            df_171[df_171['qty'] > 0]
            .groupby('date_ym')['tophncode']
            .nunique()
            .reset_index()
            .rename(columns={'tophncode': 'active_hosp'})
        )
    else:
        df_active = df_qty.copy()
        df_active['active_hosp'] = np.nan

    df_monthly = df_qty.merge(df_active, on='date_ym', how='left')
    df_monthly = df_monthly[df_monthly['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
    df_monthly = df_monthly.sort_values('date_ym').reset_index(drop=True)
    df_monthly['month'] = df_monthly['date_ym'].dt.month
    df_monthly['year'] = df_monthly['date_ym'].dt.year

    hist_yoy = []
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

    train_start = pd.Timestamp('2024-06-01')
    train_end = pd.Timestamp('2025-10-01')
    train_dates = pd.date_range(train_start, train_end, freq='MS')

    train_samples = []
    train_targets_abs = []
    train_targets_mom = []

    for date in train_dates:
        features, targets_abs, targets_mom, _ = create_holistic_sales_samples_v3_171(
            df_monthly, date, target_yoy_mean
        )
        if not any(np.isnan(targets_abs)) and not any(np.isnan(targets_mom)):
            train_samples.append(features)
            train_targets_abs.append(targets_abs)
            train_targets_mom.append(targets_mom)

    if not train_samples:
        raise ValueError('No valid training samples for spec 171')

    train_X = pd.DataFrame(train_samples)
    train_y_abs = np.array(train_targets_abs)
    train_y_mom = np.array(train_targets_mom)

    train_X_filled = train_X.copy()
    feature_stats = {}

    for col in train_X_filled.columns:
        if train_X_filled[col].isna().all():
            train_X_filled[col] = 0
        elif 'mom' in col:
            if train_X_filled[col].notna().any():
                col_median = train_X_filled[col].median()
                feature_stats[col] = {'median': col_median, 'mean': 0}
                train_X_filled[col] = train_X_filled[col].fillna(col_median)
            else:
                train_X_filled[col] = 0
        elif 'slope' in col or 'r2' in col:
            train_X_filled[col] = train_X_filled[col].fillna(0)
        else:
            if train_X_filled[col].notna().any():
                col_mean = train_X_filled[col].mean()
                feature_stats[col] = {'mean': col_mean, 'median': 0}
                train_X_filled[col] = train_X_filled[col].fillna(col_mean)
            else:
                train_X_filled[col] = 0

    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X_filled)

    models_abs = []
    for i in range(6):
        model = Ridge(alpha=10.0, random_state=42)
        model.fit(train_X_scaled, train_y_abs[:, i])
        models_abs.append(model)

    models_mom = []
    for i in range(6):
        model = Ridge(alpha=10.0, random_state=42)
        model.fit(train_X_scaled, train_y_mom[:, i])
        models_mom.append(model)

    optimal_weights = {
        'w_mom': 0.5,
        'w_abs': 0.5,
    }

    model_data = {
        'models_abs': models_abs,
        'models_mom': models_mom,
        'scaler': scaler_X,
        'feature_stats': feature_stats,
        'optimal_weights': optimal_weights,
        'feature_columns': list(train_X_filled.columns),
        'train_info': {
            'n_samples': len(train_samples),
            'n_features': train_X.shape[1],
            'train_period': f"{train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')}",
        },
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    return model_data
