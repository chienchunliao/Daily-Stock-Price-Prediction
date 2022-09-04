# -*- coding: utf-8 -*-

import pandas as pd
import os
from datetime import datetime
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

def Fin_BS(df):
    df = df.iloc[13:90,:]
    df = df.dropna(how='all')
    date = lambda st: pd.to_datetime(st.split('\n')[-1])
    column_names = df.iloc[0,:].map(date)
    df = df.set_axis(column_names, axis=1)
    df = df.set_axis(df.iloc[:,0], axis=0)
    df = df.iloc[1:,1:].T
    return df

def Fin_HC(df):
    df = df.iloc[12:, :]
    df = df.dropna(how='all')
    df = df.set_axis(df.iloc[0,:], axis=1)
    df = df.set_axis(df.iloc[:,0], axis=0)
    df = df.iloc[2:,1:].T
    df.index = df.index.map(pd.Timestamp)
    return df

def Fin_M(df):
    df = df.iloc[12:,:]
    df = df.set_axis(df.iloc[0,:], axis=1)
    df = df.dropna()
    fun = lambda x: x+'_Average'
    df = df.set_axis(df.iloc[:,0].map(fun))
    df = df.iloc[:,1:].T.iloc[1:,:]
    df.index = df.index.map(pd.Timestamp)
    return df

def Fin_R(df):
    df = df.iloc[10:158, :]
    df.dropna(how='all', inplace=True)
    df = df.set_axis(df.iloc[:,0], axis=0).iloc[:,1:]
    df.dropna(how='all', inplace=True)
    trans = lambda x: pd.to_datetime(x.split('\n')[-1])
    column_name = df.iloc[0,:].map(trans)
    df = df.set_axis(column_name, axis=1)
    df = df.iloc[1:,:].T
    return df

def get_Name_Type(name):
    import re
    name = re.split(r'[. ]', name)
    if name[-1] == 'xls':
        ind = name.index('Financials')
        company_code = name[ind-1]
        company_name_lis = name[0:ind-2]
        type_list = name[ind:-1]
        data_type = ''
        company_name = ''
        for i in type_list:
            data_type += i
        for j in company_name_lis:
            company_name += j
        return company_name, company_code, data_type
    else:
        print("not excel")
        return None

def replace_error(cell):
    if type(cell) == str:
        return np.nan
    else:
        return cell
com_code_lis = ['WDC', 'STX', 'NTAP', 'ACN', 'AAPL', 'AMD', 'NVDA', 'QCOM', 'AVGO', 'ADBE', 'ADI', 'HPQ', 'DELL', 'V', 'TXN', 'MA', 'MSFT', 'CRM', 'INTC', 'ORCL']
file_names = os.listdir()
df_fin_dic = {key:{} for key in com_code_lis}
for name in file_names:
    if name.split('.')[-1] == 'xls':
        df = pd.read_excel(name)
        company_name, company_code, data_type = get_Name_Type(name)
        if data_type == "FinancialsBalanceSheet":
            df_save = Fin_BS(df)
            try:
                df_save = df_save[['Short Term Investments', 
                                   'Long-term Investments', 
                                   'Total Assets']]
                df_save = df_save.applymap(replace_error)
                df_save['IOA'] = (df_save['Short Term Investments'] + df_save['Long-term Investments']) / df_save['Total Assets']
                df_save.drop(['Short Term Investments', 'Long-term Investments', 'Total Assets'], axis=True, inplace=True)
            except Exception as e:
                print(e)
                print((company_code, data_type))
                continue

        elif data_type == 'FinancialsHistoricalCapitalization':
            df_save = Fin_HC(df)
            try:
                df_save = df_save[['Shares Out.']]
            except Exception as e:
                print(e)
                print((company_code, data_type))
                continue

        elif data_type == 'FinancialsMultiples':
            df_save = Fin_M(df)
            try:
                df_save = df_save[['TEV/LTM Total Revenue_Average', 
                                   'P/LTM EPS_Average', 
                                   'P/NTM EPS_Average', 
                                   'P/BV_Average', 
                                   'TEV/LTM Unlevered FCF_Average', 
                                   'Market Cap/LTM Levered FCF_Average']]
            except Exception as e:
                print(e)
                print((company_code, data_type))
                continue
            
        elif data_type == 'FinancialsRatios':
            df_save = Fin_R(df)
            #print(df_save.columns)
            try: 
                df_1 = df_save[['  Return on Assets %', 
                                '  EBIT Margin %']]
                df_gross = df_save[['  Gross Profit']].iloc[:,0]
                df_total = df_save[['  Total Assets']].iloc[:,0]
                df_save = pd.merge(df_1, df_gross, left_index=True, right_index=True, how='outer')
                df_save = pd.merge(df_save, df_total, left_index=True, right_index=True, how='outer')
                #print(df_save.columns)
            except Exception as e:
                print(e)
                print((company_code, data_type))
                continue
        
        time_index = (df_save.index > datetime(2009,12,31)) & (df_save.index < datetime(2020,1,1))
        df_save = df_save.loc[time_index, :]
        df_save = df_save.applymap(replace_error)
        df_fin_dic[company_code][data_type] = df_save

fin_dic = {}
for com_code in df_fin_dic:
    datas = list(df_fin_dic[com_code].values())
    df_comb = datas[0]
    for data in datas[1:]:
        df_comb = df_comb.join(data, how='outer')
        #df_comb['IOA'] = (df_comb.iloc[:,0] + df_comb.iloc[:,1]) / df_comb.iloc[:,2]
        #df_comb.drop(['Short Term Investments', 'Long-term Investments', '  Total Assets'], axis=1, inplace=True)
    df_comb['company code'] = com_code
    fin_dic[com_code] = df_comb

file = open('financial_dic_indiv.subdata', 'wb')
pickle.dump(fin_dic, file)
file.close()
    