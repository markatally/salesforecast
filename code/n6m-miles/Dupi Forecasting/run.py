from forecast_api import SalesForecastAPI
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).resolve().parents[3] / 'data' / 'sales_6m_monthly.csv'


def main():
    """Dupixent sales forecasting - 162 and 169 specs only"""
    api = SalesForecastAPI()
    api.setup(
        model_path_162='trained_model_162.pkl',
        model_path_169='trained_model_169.pkl'
    )
    
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['prodmdmcode'] = df_raw['prodmdmcode'].astype(str)
    df_raw = df_raw.rename(columns={'tomdphncode': 'tophncode'})
    
    dupixent_results = pd.DataFrame()
    
    df_162 = df_raw[df_raw['prodmdmcode'] == '162'].copy()
    if not df_162.empty:
        result_162 = api.forecast_162_from_raw(df_162)
        result_162['product'] = 'Dupixent 300mg'
        dupixent_results = pd.concat([dupixent_results, result_162], ignore_index=True)
        print(f"✓ Dupixent 300mg (162) forecast completed")
    else:
        print(f"⚠ Warning: spec 162 not found in {DATA_PATH}")
    
    # Dupixent 169 (200mg)
    df_169 = df_raw[df_raw['prodmdmcode'] == '169'].copy()
    if not df_169.empty:
        result_169 = api.forecast_169_from_raw(df_169)
        result_169['product'] = 'Dupixent 200mg'
        dupixent_results = pd.concat([dupixent_results, result_169], ignore_index=True)
        print(f"✓ Dupixent 200mg (169) forecast completed")
    else:
        print(f"⚠ Warning: spec 169 not found in {DATA_PATH}")
    
    # Output results
    if not dupixent_results.empty:
        dupixent_results = dupixent_results[['product', 'month', 'pred']]
        dupixent_results = dupixent_results.sort_values(['month', 'product']).reset_index(drop=True)
        print("\n" + "=" * 80)
        print("Dupixent Forecast Results:")
        print("=" * 80)
        print(dupixent_results)
    else:
        print("\n⚠ No forecast results generated. Please check data files.")


if __name__ == '__main__':
    main()
