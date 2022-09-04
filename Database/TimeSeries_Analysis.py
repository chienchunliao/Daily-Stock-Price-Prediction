# -*- coding: utf-8 -*-

import TimeSeries_Models as TS, yfinance as yf, pandas as pd, numpy as np, pickle, copy, time, itertools
from statistics import mean

def get_all_file(ftype='.txt'):
    import os, glob, re
    file_lis = []
    for path in glob.glob('**', recursive=True):
        if os.path.isfile(path) and ftype in path:
            file_lis.append(path)
    return file_lis

company_info_dic = {'Accentureplc': 'ACN',
                    'AdobeInc': 'ADBE',
                    'AdvancedMicroDevicesInc': 'AMD',
                    'AnalogDevicesInc': 'ADI',
                    'AppleInc': 'AAPL',
                    'BroadcomInc': 'AVGO',
                    'DellTechnologiesInc': 'DELL',
                    'HewlettPackardEnterpriseCompany': 'HPE',
                    'HPInc': 'HPQ',
                    'IntelCorporation': 'INTC',
                    'MastercardIncorporated': 'MA',
                    'MicrosoftCorporation': 'MSFT',
                    'NetAppInc': 'NTAP',
                    'NVIDIACorporation': 'NVDA',
                    'OracleCorporation': 'ORCL',
                    'QUALCOMMIncorporated': 'QCOM',
                    'SalesforceInc': 'CRM',
                    'SeagateTechnologyHoldingsplc': 'STX',
                    'TexasInstrumentsIncorporated': 'TXN',
                    'VisaInc': 'V',
                    'WesternDigitalCorporation': 'WDC'}
company_code_lis = ['ACN', 'ADBE', 'AMD', 'ADI', 'AAPL', 'AVGO', 'HPQ', 'INTC', 'MA', 'MSFT', 'NTAP', 'NVDA', 'ORCL', 'QCOM', 'CRM', 'STX', 'TXN', 'V', 'WDC']
model_lis_non = ['DETS', 'ETS']
model_lis = ['ARIMA', 'ARMA', 'AR_k', 'DMA_k', 'MA_k']

path_lis = get_all_file('.database')
#%%
for path in path_lis:
    if 'non-stock' in path:
        file = open(path, 'rb')
        other_db = pickle.load(file)
        file.close()
    else:
        file = open(path, 'rb')
        stock_db = pickle.load(file)
        file.close()


#%% Gridsearch 
cv = 5
num_best = 3
price_focus = 'Close'
model_par = {'ARIMA': [2,2,2],
              'ARMA': [2,2],
              'AR': [3],
              'DMA': [3],
              'MA': [3],
              'DETS': [None],
              'ETS': [None]}

result_lis = []
cols = ['model name', 'parameter', 'validation score', 'test score', 'model']

for m in model_par:
    if m == 'ARIMA':
        max_p = model_par[m][0]
        max_d = model_par[m][1]
        max_q = model_par[m][2]
        par_try = itertools.product(range(1, max_p+1), range(1, max_d+1), range(1, max_q+1))
        for par in par_try:
            model = TS.ARIMA(*par)
            com_val_scor_lis = []
            com_test_scor_lis = []
            for com in company_code_lis:
                df_price_train = stock_db['Train'][com][price_focus]
                df_price_test = stock_db['Test'][com][price_focus]
                train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
                val_scor_lis = []
                test_scor_lis = []
                for train, valid in train_vali:
                    try:
                        model.fit(train)
                        val_scor = model.score(valid)
                        val_scor_lis.append(val_scor)
                    except:
                        pass
                model.fit(df_price_train)
                com_test_scor = model.score(df_price_test)
                com_test_scor_lis.append(com_test_scor)
                com_val_scor = mean(val_scor_lis)
                com_val_scor_lis.append(com_val_scor)
            par_val_scor = mean(com_val_scor_lis)
            par_test_scor = mean(com_test_scor_lis)
            result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.ARIMA(*par)], index=cols))
            
    if m == 'ARMA':
        max_p = model_par[m][0]
        max_q = model_par[m][1]
        par_try = itertools.product(range(1, max_p+1), range(1, max_q+1))
        for par in par_try:
            model = TS.ARMA(*par)
            com_val_scor_lis = []
            com_test_scor_lis = []
            for com in company_code_lis:
                df_price_train = stock_db['Train'][com][price_focus]
                df_price_test = stock_db['Test'][com][price_focus]
                train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
                val_scor_lis = []
                test_scor_lis = []
                for train, valid in train_vali:
                    try:
                        model.fit(train)
                        val_scor = model.score(valid)
                        val_scor_lis.append(val_scor)
                    except:
                        pass
                model.fit(df_price_train)
                com_test_scor = model.score(df_price_test)
                com_test_scor_lis.append(com_test_scor)
                com_val_scor = mean(val_scor_lis)
                com_val_scor_lis.append(com_val_scor)
            par_val_scor = mean(com_val_scor_lis)
            par_test_scor = mean(com_test_scor_lis)
            result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.ARMA(*par)], index=cols))
            
    if m == 'AR':
        max_p = model_par[m][0]
        for par in range(1, max_p+1):
            model = TS.AR(par)
            com_val_scor_lis = []
            com_test_scor_lis = []
            for com in company_code_lis:
                df_price_train = stock_db['Train'][com][price_focus]
                df_price_test = stock_db['Test'][com][price_focus]
                train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
                val_scor_lis = []
                test_scor_lis = []
                for train, valid in train_vali:
                    try:
                        model.fit(train)
                        val_scor = model.score(valid)
                        val_scor_lis.append(val_scor)
                    except:
                        pass
                model.fit(df_price_train)
                com_test_scor = model.score(df_price_test)
                com_test_scor_lis.append(com_test_scor)
                com_val_scor = mean(val_scor_lis)
                com_val_scor_lis.append(com_val_scor)
            par_val_scor = mean(com_val_scor_lis)
            par_test_scor = mean(com_test_scor_lis)
            result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.AR(par)], index=cols))
            
    if m == 'MA':
        max_q = model_par[m][0]
        for par in range(2, max_q+1):
            model = TS.MA(par)
            com_val_scor_lis = []
            com_test_scor_lis = []
            for com in company_code_lis:
                df_price_train = stock_db['Train'][com][price_focus]
                df_price_test = stock_db['Test'][com][price_focus]
                train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
                val_scor_lis = []
                test_scor_lis = []
                for train, valid in train_vali:
                    try:
                        model.fit(train)
                        val_scor = model.score(valid)
                        val_scor_lis.append(val_scor)
                    except:
                        pass
                model.fit(df_price_train)
                com_test_scor = model.score(df_price_test)
                com_test_scor_lis.append(com_test_scor)
                com_val_scor = mean(val_scor_lis)
                com_val_scor_lis.append(com_val_scor)
            par_val_scor = mean(com_val_scor_lis)
            par_test_scor = mean(com_test_scor_lis)
            result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.MA(par)], index=cols))
            
    if m == 'DMA':
        max_p = model_par[m][0]
        for par in range(2, max_p+1):
            model = TS.DMA(par)
            com_val_scor_lis = []
            com_test_scor_lis = []
            for com in company_code_lis:
                df_price_train = stock_db['Train'][com][price_focus]
                df_price_test = stock_db['Test'][com][price_focus]
                train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
                val_scor_lis = []
                test_scor_lis = []
                for train, valid in train_vali:
                    try:
                        model.fit(train)
                        val_scor = model.score(valid)
                        val_scor_lis.append(val_scor)
                    except:
                        pass
                model.fit(df_price_train)
                com_test_scor = model.score(df_price_test)
                com_test_scor_lis.append(com_test_scor)
                com_val_scor = mean(val_scor_lis)
                com_val_scor_lis.append(com_val_scor)
            par_val_scor = mean(com_val_scor_lis)
            par_test_scor = mean(com_test_scor_lis)
            result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.DMA(par)], index=cols))
            
    if m == 'ETS':
        par = None
        model = TS.ETS()
        com_val_scor_lis = []
        com_test_scor_lis = []
        for com in company_code_lis:
            df_price_train = stock_db['Train'][com][price_focus]
            df_price_test = stock_db['Test'][com][price_focus]
            train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
            val_scor_lis = []
            test_scor_lis = []
            for train, valid in train_vali:
                try:
                    model.fit(train)
                    val_scor = model.score(valid)
                    val_scor_lis.append(val_scor)
                except:
                    pass
            model.fit(df_price_train)
            com_test_scor = model.score(df_price_test)
            com_test_scor_lis.append(com_test_scor)
            com_val_scor = mean(val_scor_lis)
            com_val_scor_lis.append(com_val_scor)
        par_val_scor = mean(com_val_scor_lis)
        par_test_scor = mean(com_test_scor_lis)
        result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.ETS()], index=cols))

    if m == 'DETS':
        par = None
        model = TS.DETS()
        com_val_scor_lis = []
        com_test_scor_lis = []
        for com in company_code_lis:
            df_price_train = stock_db['Train'][com][price_focus]
            df_price_test = stock_db['Test'][com][price_focus]
            train_vali = TS.gen_ts_cv(df_price_train, n_split=cv)
            val_scor_lis = []
            test_scor_lis = []
            for train, valid in train_vali:
                try:
                    model.fit(train)
                    val_scor = model.score(valid)
                    val_scor_lis.append(val_scor)
                except:
                    pass
            model.fit(df_price_train)
            com_test_scor = model.score(df_price_test)
            com_test_scor_lis.append(com_test_scor)
            com_val_scor = mean(val_scor_lis)
            com_val_scor_lis.append(com_val_scor)
        par_val_scor = mean(com_val_scor_lis)
        par_test_scor = mean(com_test_scor_lis)
        result_lis.append(pd.Series([m, par, par_val_scor, par_test_scor, TS.DETS()], index=cols))

df_result = pd.DataFrame(result_lis, columns=cols)
df_best_models_n = df_result.sort_values(by='validation score',
                                          ascending=True).iloc[:num_best,:]
df_best_models_n.to_pickle('best_models.model')
#%%
# df_TSmodel_dic = dict(zip(df_best_models_n['model name'] + '_' + df_best_models_n['parameter'].astype(str), 
#                           [[]]*num_best
#                           )
#                       )

# for com in company_code_lis:
#     df_other = other_db[com]
#     df_other = df_other.fillna(method='ffill',axis=0)
#     df_stock_train = stock_db['Train'][com][price_focus]
#     df_stock_test = stock_db['Test'][com][price_focus]
#     pred_lis = []
#     for idne, row in df_best_models_n.iterrows():
#         model = row['model']
#         model_name = row['model name'] + '_' + str(row['parameter'])
#         model.fit(df_stock_train)
#         pred = model.predict(df_stock_test)
#         pred.name = model_name
#         pred_lis.append(pred)
#     df_pred = pd.DataFrame(pred_lis).T
#     df_major = df_other.join([df_pred, df_stock_test], how='outer')
#     time_range = (df_major.index >= pd.Timestamp('2010-01-01')) & (df_major.index <= pd.Timestamp('2019-12-31'))
#     df_major = df_major.loc[time_range, :]
#     df_major['News Score_indiv'] = df_major['News Score_indiv'].rolling(3).sum()
#     df_major['News Score_overall_WSJ'] = df_major['News Score_overall_WSJ'].rolling(3).sum()
#     df_major.name = model_name
#     df_TSmodel_dic[model_name].append(df_major)
# for key, val in df_TSmodel_dic.items():
#     df = pd.concat(val)
#     df.dropna(inplace=True)
#     df_TSmodel_dic[key] = df
#     path = key + '.obj'
#     df.to_pickle(path=path)
#     df.to_csv(path=path)

