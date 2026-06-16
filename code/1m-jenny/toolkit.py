import pandas as pd

def get_previous_month(year_month:int, months_back:int) -> int:
    """获取指定年月向前N个月对应的年月"""
    year = year_month // 100
    month = year_month % 100
    total_months = year * 12 + month - 1  # -1 因为月份从1开始
    new_total_months = total_months - months_back
    new_year = new_total_months // 12
    new_month = new_total_months % 12 + 1  # +1 恢复月份从1开始
    return new_year * 100 + new_month

def read_data(data_path):
    # 1、读取数据
    df_ori = pd.read_csv(data_path)
    # 2、日期类型转换
    df_ori['transdate'] = pd.to_datetime(df_ori['transdate'])
    # 3、聚合列名
    cols = df_ori.columns.tolist()
    cols.remove('qty')
    # 4、数量聚合
    df_ori = df_ori.groupby(cols)[['qty']].sum().reset_index()
    # 5、去除为0流向
    df_ori = df_ori[df_ori['qty']!=0]
    return df_ori

def calculate_yoy_growth(df_data_ym, ym):
    ym_13 = get_previous_month(ym,13)
    ym_12 = get_previous_month(ym,12)
    ym_1 = get_previous_month(ym,1)
    qty_13 = df_data_ym[df_data_ym['bizym']==ym_13]['qty'].values[0]
    qty_12 = df_data_ym[df_data_ym['bizym']==ym_12]['qty'].values[0]
    qty_1 = df_data_ym[df_data_ym['bizym']==ym_1]['qty'].values[0]
    qty = df_data_ym[df_data_ym['bizym']==ym]['qty'].values[0]
    growth_yoy = qty_12/qty_13-1
    growth_cur = qty/qty_1-1
    return growth_yoy, growth_cur