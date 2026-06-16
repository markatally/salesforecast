from forecast_api import SalesForecastAPI
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def main():
    api = SalesForecastAPI()
    api.setup(model_path_170='trained_model_170.pkl', model_path_171='trained_model_171.pkl')

    df_raw = pd.read_csv(r'.\data\Soliqua.csv', encoding='gbk')

    # Soliqua 1:1
    result_170 = api.forecast_from_raw(df_raw)
    result_170['product'] = 'Soliqua 1:1'

    # Soliqua 2:1
    result_171 = api.forecast_171_from_raw(df_raw)
    result_171['product'] = 'Soliqua 2:1'


    all_results = pd.concat([result_170, result_171], ignore_index=True)
    all_results = all_results[['product', 'month', 'pred']]
    all_results = all_results.sort_values(['month', 'product']).reset_index(drop=True)

    print(all_results)


if __name__ == '__main__':
    main()
