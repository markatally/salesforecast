from forecast_api import SalesForecastAPI
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')


def main():
    """Dupixent sales forecasting - 162 and 169 specs only"""
    api = SalesForecastAPI()
    api.setup(
        model_path_162='trained_model_162.pkl',
        model_path_169='trained_model_169.pkl'
    )
    
    # Dupixent 162 (300mg)
    dupixent_162_path = r'.\data\162_22-25.csv'
    dupixent_169_path = r'.\data\169_22-25.csv'
    
    dupixent_results = pd.DataFrame()
    
    if os.path.exists(dupixent_162_path):
        df_162 = pd.read_csv(dupixent_162_path, encoding='gbk')
        df_162['spec'] = '162'
        df_162['date_ym'] = pd.to_datetime(df_162['bizym'], format='%Y%m')
        result_162 = api.forecast_162_from_raw(df_162)
        result_162['product'] = 'Dupixent 300mg'
        dupixent_results = pd.concat([dupixent_results, result_162], ignore_index=True)
        print(f"✓ Dupixent 300mg (162) forecast completed")
    else:
        print(f"⚠ Warning: {dupixent_162_path} not found")
    
    # Dupixent 169 (200mg)
    if os.path.exists(dupixent_169_path):
        df_169 = pd.read_csv(dupixent_169_path, encoding='gbk')
        df_169['spec'] = '169'
        df_169['date_ym'] = pd.to_datetime(df_169['bizym'], format='%Y%m')
        result_169 = api.forecast_169_from_raw(df_169)
        result_169['product'] = 'Dupixent 200mg'
        dupixent_results = pd.concat([dupixent_results, result_169], ignore_index=True)
        print(f"✓ Dupixent 200mg (169) forecast completed")
    else:
        print(f"⚠ Warning: {dupixent_169_path} not found")
    
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
