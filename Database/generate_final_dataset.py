# -*- coding: utf-8 -*-

import pickle, pandas as pd

def get_all_file(ftype='.txt'):
    import os, glob
    file_lis = []
    for path in glob.glob('**', recursive=True):
        if os.path.isfile(path) and ftype in path:
            file_lis.append(path)
    return file_lis

for path in get_all_file('.database'):
    if 'non-stock' in path:
        file = open(path, 'rb')
        other_db = pickle.load(file)
        file.close()
    else:
        file = open(path, 'rb')
        stock_db = pickle.load(file)
        file.close()

file = open("best_models.model", 'rb')
df_best_models_n = pickle.load(file)
file.close()

company_code_lis = ['ACN', 'ADBE', 'AMD', 'ADI', 'AAPL', 'AVGO', 'HPQ', 'INTC', 'MA', 'MSFT', 'NTAP', 'NVDA', 'ORCL', 'QCOM', 'CRM', 'STX', 'TXN', 'V', 'WDC']
num_best = df_best_models_n.shape[0]
price_focus = 'Close'

df_TSmodel_dic = dict(zip(company_code_lis, 
                          [[]]*len(company_code_lis)
                          )
                      )
model_name_lis = []
for idne, row in df_best_models_n.iterrows():
    model = row['model']
    model_name = row['model name'] + '_' + str(row['parameter'])
    model_name_lis.append(model_name)


for com in company_code_lis:
    df_other = other_db[com]
    df_other = df_other.fillna(method='ffill',axis=0)
    df_stock_train = stock_db['Train'][com][price_focus]
    df_stock_test = stock_db['Test'][com][price_focus]
    pred_lis = []
    for idne, row in df_best_models_n.iterrows():
        model = row['model']
        model_name = row['model name'] + '_' + str(row['parameter'])
        model.fit(df_stock_train)
        pred = model.predict(df_stock_test)
        pred.name = model_name
        pred_lis.append(pred)
    df_pred = pd.DataFrame(pred_lis).T
    df_major = df_other.join([df_pred, df_stock_test], how='outer')
    time_range = (df_major.index >= pd.Timestamp('2010-01-01')) & (df_major.index <= pd.Timestamp('2019-12-31'))
    df_major = df_major.loc[time_range, :]
    df_major['News Score_indiv'] = df_major['News Score_indiv'].rolling(3).sum()
    df_major['News Score_overall_WSJ'] = df_major['News Score_overall_WSJ'].rolling(3).sum()
    df_TSmodel_dic[com] = df_major
df = pd.concat(df_TSmodel_dic.values())
df.dropna(inplace=True)
result = [model_name_lis, df]
with open('Major_Dataset.majordb', 'wb') as f:
    pickle.dump(result, f)
# for key, val in df_TSmodel_dic.items():
#     df = pd.concat(val)
#     df.dropna(inplace=True)
#     df_TSmodel_dic[key] = df
#     path_save = key + '.majordb'
#     with open(path_save, 'wb') as f:
#         pickle.dump(df, f)

