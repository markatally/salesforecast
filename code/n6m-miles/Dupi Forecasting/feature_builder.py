import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def get_price_162(date: pd.Timestamp) -> float:
    if date < pd.Timestamp('2024-01-01'):
        return 3160.0
    elif date < pd.Timestamp('2025-01-01'):
        return 2780.0
    else:
        return 1508.0


def get_price_169(date: pd.Timestamp) -> float:
    if date < pd.Timestamp('2024-01-01'):
        return 2980.0
    elif date < pd.Timestamp('2025-01-01'):
        return 2680.0
    else:
        return 1408.0


def process_price_elasticity_162(df_monthly: pd.DataFrame, annual_growth_rate: float = None) -> tuple:
    df_nat = df_monthly.copy()
    df_nat = df_nat.sort_values('date_ym').reset_index(drop=True)
    df_nat['price'] = df_nat['date_ym'].apply(get_price_162)
    df_nat['is_price_drop_month'] = 0
    df_nat.loc[df_nat['date_ym'] == pd.Timestamp('2024-01-01'), 'is_price_drop_month'] = 1
    df_nat.loc[df_nat['date_ym'] == pd.Timestamp('2025-01-01'), 'is_price_drop_month'] = 1
    df_nat['price_change'] = df_nat['price'].diff()
    df_nat['is_price_down'] = (df_nat['price_change'] < -100).astype(int)
    
    price_interval_list = []
    current_interval_id = 0
    for idx, row in df_nat.iterrows():
        if row['is_price_down'] == 1:
            current_interval_id += 1
        price_interval_list.append(current_interval_id)
    df_nat['price_interval_id'] = price_interval_list
    
    interval_0_data = df_nat[df_nat['price_interval_id'] == 0]
    base_sales = interval_0_data['sales'].mean()
    base_price = interval_0_data['price'].mean()
    base_hosp = interval_0_data['active_hosp'].mean()
    base_year = interval_0_data['date_ym'].min().year
    
    if annual_growth_rate is None:
        df_baseline = df_nat[df_nat['price_interval_id'] == 0].copy()
        if len(df_baseline) > 0:
            df_baseline['year'] = df_baseline['date_ym'].dt.year
            yearly_stats = df_baseline.groupby('year').agg({
                'sales': 'mean',
                'active_hosp': 'mean'
            }).reset_index()
            yearly_stats['per_hosp_sales'] = yearly_stats['sales'] / yearly_stats['active_hosp']
            yearly_stats['yoy_growth'] = yearly_stats['per_hosp_sales'].pct_change()
            growth_rates = yearly_stats['yoy_growth'].dropna()
            if len(growth_rates) > 0:
                annual_growth_rate = growth_rates.median()
            else:
                annual_growth_rate = 0.20
        else:
            annual_growth_rate = 0.20
    
    interval_multipliers = {}
    interval_prices = {}
    interval_natural_growth = {}
    
    for interval_id in sorted(df_nat['price_interval_id'].unique()):
        interval_data = df_nat[df_nat['price_interval_id'] == interval_id]
        interval_sales = interval_data['sales'].mean()
        interval_price = interval_data['price'].mean()
        interval_hosp = interval_data['active_hosp'].mean()
        interval_start_year = interval_data['date_ym'].min().year
        
        if interval_id == 0:
            natural_growth_coef = 1.0
            multiplier = 1.0
        else:
            years_elapsed = interval_start_year - base_year
            hosp_growth_mult = interval_hosp / base_hosp
            per_hosp_growth_coef = (1 + annual_growth_rate) ** years_elapsed
            natural_growth_coef = hosp_growth_mult * per_hosp_growth_coef
            actual_growth_mult = interval_sales / base_sales
            multiplier = actual_growth_mult / natural_growth_coef
        
        interval_multipliers[interval_id] = multiplier
        interval_prices[interval_id] = interval_price
        interval_natural_growth[interval_id] = natural_growth_coef
    
    df_nat['baseline_sales'] = df_nat.apply(
        lambda row: row['sales'] / interval_multipliers[row['price_interval_id']],
        axis=1
    )
    df_nat['baseline_hosp'] = df_nat['active_hosp'].copy()
    
    return df_nat, interval_multipliers, interval_natural_growth, interval_prices


def process_price_elasticity_169(df_monthly: pd.DataFrame, annual_growth_rate: float = None) -> tuple:
    df_nat = df_monthly.copy()
    df_nat = df_nat.sort_values('date_ym').reset_index(drop=True)
    df_nat['price'] = df_nat['date_ym'].apply(get_price_169)
    df_nat['is_price_drop_month'] = 0
    df_nat.loc[df_nat['date_ym'] == pd.Timestamp('2024-01-01'), 'is_price_drop_month'] = 1
    df_nat.loc[df_nat['date_ym'] == pd.Timestamp('2025-01-01'), 'is_price_drop_month'] = 1
    df_nat['price_change'] = df_nat['price'].diff()
    df_nat['is_price_down'] = (df_nat['price_change'] < -100).astype(int)
    
    price_interval_list = []
    current_interval_id = 0
    for idx, row in df_nat.iterrows():
        if row['is_price_down'] == 1:
            current_interval_id += 1
        price_interval_list.append(current_interval_id)
    df_nat['price_interval_id'] = price_interval_list
    
    interval_0_data = df_nat[df_nat['price_interval_id'] == 0]
    base_sales = interval_0_data['sales'].mean()
    base_price = interval_0_data['price'].mean()
    base_hosp = interval_0_data['active_hosp'].mean()
    base_year = interval_0_data['date_ym'].min().year
    
    if annual_growth_rate is None:
        df_baseline = df_nat[df_nat['price_interval_id'] == 0].copy()
        if len(df_baseline) > 0:
            df_baseline['year'] = df_baseline['date_ym'].dt.year
            yearly_stats = df_baseline.groupby('year').agg({
                'sales': 'mean',
                'active_hosp': 'mean'
            }).reset_index()
            yearly_stats['per_hosp_sales'] = yearly_stats['sales'] / yearly_stats['active_hosp']
            yearly_stats['yoy_growth'] = yearly_stats['per_hosp_sales'].pct_change()
            growth_rates = yearly_stats['yoy_growth'].dropna()
            if len(growth_rates) > 0:
                annual_growth_rate = growth_rates.median()
            else:
                annual_growth_rate = 0.20
        else:
            annual_growth_rate = 0.20
    
    interval_multipliers = {}
    interval_prices = {}
    interval_natural_growth = {}
    
    for interval_id in sorted(df_nat['price_interval_id'].unique()):
        interval_data = df_nat[df_nat['price_interval_id'] == interval_id]
        interval_sales = interval_data['sales'].mean()
        interval_price = interval_data['price'].mean()
        interval_hosp = interval_data['active_hosp'].mean()
        interval_start_year = interval_data['date_ym'].min().year
        
        if interval_id == 0:
            natural_growth_coef = 1.0
            multiplier = 1.0
        else:
            years_elapsed = interval_start_year - base_year
            hosp_growth_mult = interval_hosp / base_hosp
            per_hosp_growth_coef = (1 + annual_growth_rate) ** years_elapsed
            natural_growth_coef = hosp_growth_mult * per_hosp_growth_coef
            actual_growth_mult = interval_sales / base_sales
            multiplier = actual_growth_mult / natural_growth_coef
        
        interval_multipliers[interval_id] = multiplier
        interval_prices[interval_id] = interval_price
        interval_natural_growth[interval_id] = natural_growth_coef
    
    df_nat['baseline_sales'] = df_nat.apply(
        lambda row: row['sales'] / interval_multipliers[row['price_interval_id']],
        axis=1
    )
    df_nat['baseline_hosp'] = df_nat['active_hosp'].copy()
    
    return df_nat, interval_multipliers, interval_natural_growth, interval_prices


def restore_actual_sales(baseline_predictions: np.ndarray, price_interval_multiplier: float) -> np.ndarray:
    return baseline_predictions * price_interval_multiplier


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


def create_holistic_samples_162_recent(df, target_date):
    """
    Create prediction samples for specified forecast point (Spec 162)
    Args:
        df: DataFrame with columns 'date_ym', 'sales', 'mom_rate', 'month'
        target_date: Timestamp of forecast starting point
    """
    future_months = pd.date_range(target_date + pd.DateOffset(months=1), periods=6, freq='MS')
    df_historical = df[df['date_ym'] <= target_date].copy()
    
    # Feature 1: Last year same period 6 months sales
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
    
    # Feature 2: Recent 6 months sales and mom
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
    
    # Feature 2.1: Recent 6 months active hospital features
    if 'active_hosp' in df_historical.columns:
        active_recent = []
        for i in range(6):
            month_curr = recent_6_months[i]
            val = df_historical[df_historical['date_ym'] == month_curr]['active_hosp'].values
            active_recent.append(val[0] if len(val) > 0 else np.nan)
    else:
        active_recent = [np.nan] * 6

    if 'new_active_hosp' in df_historical.columns:
        new_active_recent = []
        for i in range(6):
            month_curr = recent_6_months[i]
            val = df_historical[df_historical['date_ym'] == month_curr]['new_active_hosp'].values
            new_active_recent.append(val[0] if len(val) > 0 else np.nan)
    else:
        new_active_recent = [np.nan] * 6
    
    # Feature 3: Historical same month mom
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
    
    # Feature 4: Recent 6 months slope
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
    
    # Feature 5: Last year same period slope
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
    
    # Base sales at forecast point
    base_sale = df_historical[df_historical['date_ym'] == target_date]['sales'].values
    base_sale = base_sale[0] if len(base_sale) > 0 else np.nan

    # Base active hospital at forecast point
    if 'active_hosp' in df_historical.columns:
        base_active = df_historical[df_historical['date_ym'] == target_date]['active_hosp'].values
        base_active = base_active[0] if len(base_active) > 0 else np.nan
    else:
        base_active = np.nan

    if 'new_active_hosp' in df_historical.columns:
        base_new_active = df_historical[df_historical['date_ym'] == target_date]['new_active_hosp'].values
        base_new_active = base_new_active[0] if len(base_new_active) > 0 else np.nan
    else:
        base_new_active = np.nan
    
    # Assemble feature dictionary
    features = {
        # Recent 6 months features
        'rec_sales_mean': np.nanmean(sales_recent),
        'rec_sales_sum': np.nansum(sales_recent),
        'rec_sales_std': np.nanstd(sales_recent),
        'rec_mom_mean': np.nanmean(mom_recent),
        'rec_mom_median': np.nanmedian(mom_recent),
        'rec_slope': slope_rec,
        'rec_r2': r_rec,
        
        # Last year same period features
        'ly_sales_mean': np.nanmean(sales_last_year),
        'ly_mom_mean': np.nanmean(mom_last_year),
        'ly_slope': slope_ly,
        
        # Base sales
        'base_sales': base_sale,

        # Active hospital features
        'rec_active_hosp_mean': np.nanmean(active_recent),
        'rec_active_hosp_sum': np.nansum(active_recent),
        'rec_new_active_hosp_mean': np.nanmean(new_active_recent),
        'rec_new_active_hosp_sum': np.nansum(new_active_recent),
        'base_active_hosp': base_active,
        'base_new_active_hosp': base_new_active,
        
        # Seasonal features
        'future_start_month': future_month_nums[0],
        'future_end_month': future_month_nums[-1],
    }
    
    # Monthly granular features
    month_abbrs = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for i in range(6):
        features[f'rec_M{i+1}_sales'] = sales_recent[i]
        features[f'rec_M{i+1}_mom'] = mom_recent[i]
        features[f'future_M{i+1}_hist_mom_median'] = hist_mom_median[i]
        
        # Month one-hot encoding
        month_num = future_month_nums[i]
        for m in range(1, 13):
            features[f'future_M{i+1}_is_{month_abbrs[m-1]}'] = 1 if month_num == m else 0
    
    # Extract targets
    targets_abs = []
    for m in future_months:
        sale = df[df['date_ym'] == m]['sales'].values
        targets_abs.append(sale[0] if len(sale) > 0 else np.nan)
    
    # Calculate mom targets
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


def export_trained_model_162(df_all, output_path='trained_model_162.pkl'):
    """
    Export trained model for Spec 162
    Args:
        df_all: DataFrame with all data containing 'spec' column
        output_path: Path to save the trained model
    """
    df_162 = df_all[df_all['spec'] == '162'].copy()
    if df_162.empty:
        raise ValueError('No data for spec 162')
    
    df_162['date_ym'] = pd.to_datetime(df_162['date_ym'])
    df_qty = (
        df_162.groupby('date_ym', as_index=False)
        .agg(sales=('qty', 'sum'))
        .sort_values('date_ym')
    )
    
    if 'tophncode' in df_162.columns:
        df_active = (
            df_162[df_162['qty'] > 0]
            .groupby('date_ym')['tophncode']
            .nunique()
            .reset_index()
            .rename(columns={'tophncode': 'active_hosp'})
        )
    else:
        df_active = df_qty.copy()
        df_active['active_hosp'] = np.nan
    
    df_monthly = df_qty.merge(df_active, on='date_ym', how='left')
    df_nat, interval_multipliers, interval_natural_growth, interval_prices = process_price_elasticity_162(df_monthly)
    df_nat = df_nat[df_nat['date_ym'] >= pd.Timestamp('2024-01-01')].copy()
    df_nat = df_nat.sort_values('date_ym')
    df_nat['mom_rate'] = df_nat['baseline_sales'].pct_change() * 100
    df_nat['month'] = df_nat['date_ym'].dt.month
    
    train_start = pd.Timestamp('2024-06-01')
    train_end = pd.Timestamp('2025-04-01')
    train_dates = pd.date_range(train_start, train_end, freq='MS')
    
    train_samples = []
    train_targets_abs = []
    train_targets_mom = []
    
    for date in train_dates:
        features, targets_abs, targets_mom, _ = create_holistic_samples_162_recent(df_nat, date)
        if not any(np.isnan(targets_abs)) and not any(np.isnan(targets_mom)):
            train_samples.append(features)
            train_targets_abs.append(targets_abs)
            train_targets_mom.append(targets_mom)
    
    if not train_samples:
        raise ValueError('No valid training samples for spec 162')
    
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
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(train_X_scaled, train_y_abs[:, i])
        models_abs.append(model)
    
    models_mom = []
    for i in range(6):
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(train_X_scaled, train_y_mom[:, i])
        models_mom.append(model)
    
    latest_interval_id = int(df_nat['price_interval_id'].max())
    latest_multiplier = interval_multipliers[latest_interval_id]
    
    model_data = {
        'models_abs': models_abs,
        'models_mom': models_mom,
        'scaler': scaler_X,
        'feature_stats': feature_stats,
        'optimal_weights': {
            'w_mom': 0.3,
            'w_abs': 0.7
        },
        'price_interval_multipliers': interval_multipliers,
        'latest_multiplier': latest_multiplier,
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
