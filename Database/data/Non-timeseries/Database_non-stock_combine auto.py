import pandas as pd, yfinance as yf, os, pickle, re
from multiprocessing import Process
from subprocess import call

def arithmetic_fill(s):
    s_1 = s.copy()
    import more_itertools as it
    for start, end in it.pairwise(s_1[s_1.notnull()].index):
        s_fil = s_1[start:end][1:-1]
        d = (s_1[end]-s_1[start])/(s_fil.size+1)
        c = 1
        for ind,val in s_fil.iteritems():
            s_fil[ind] = s_1[start] + d*c
            c += 1
    s_1[start:end][1:-1] = s_fil
    return s_1

def get_all_file(ftype='.txt'):
    import os, glob, re
    file_lis = []
    for path in glob.glob('**', recursive=True):
        if os.path.isfile(path) and re.search(ftype, path):
            file_lis.append(path)
    return file_lis
#%%
col_ori =  ['IOA',
            'Shares Out.', 
            'TEV/LTM Total Revenue_Average', 
            'P/LTM EPS_Average',
            'P/NTM EPS_Average', 
            'P/BV_Average', 
            'TEV/LTM Unlevered FCF_Average',
            'Market Cap/LTM Levered FCF_Average', 
            '  Return on Assets %',
            '  EBIT Margin %', 
            '  Gross Profit', 
            '  Total Assets']

var_dic = {}
subdata_lis = get_all_file('.subdata')
for path in subdata_lis:
    file = open(path, 'rb')
    var = pickle.load(file)
    file.close()
    var_dic[path.split('\\')[-1].split('.')[0]] = var

Indiv_dic = {key:{} for key in var_dic['financial_dic_indiv'].keys()}
#%%
for com_code in Indiv_dic:
    df_com = pd.merge(var_dic['News Score_Individual_CIQ'][com_code], var_dic['News Score_overall_WSJ'], left_index=True, right_index=True, how='outer')
    df_com['News Score_indiv'].fillna(0, inplace=True)
    df_com['News Score_overall_WSJ'].fillna(0, inplace=True)
    
    for i in range(1,4):
        name_in = 'News Score_indiv_' + str(i)
        ind = df_com['News Score_indiv'].rolling(i).sum()
        name_ov = 'News Score_overall_' + str(i)
        ove = df_com['News Score_overall_WSJ'].rolling(i).sum()
        for j in range(i):
            ind[j] = df_com['News Score_indiv'][j]
            ove[j] = df_com['News Score_overall_WSJ'][j]
        df_com[name_in] = ind
        df_com[name_ov] = ove
    
    df_com = pd.merge(df_com, var_dic['industry'], left_index=True, right_index=True, how='outer')
    df_com = pd.merge(df_com, var_dic['macro_index'], left_index=True, right_index=True, how='outer')
    df_com = pd.merge(df_com, var_dic['financial_dic_indiv'][com_code], left_index=True, right_index=True, how='outer')
    time_index = (df_com.index > pd.Timestamp(2009,12,31)) & (df_com.index < pd.Timestamp(2020,1,1))
    df_com = df_com.loc[time_index,:]
    df_com.drop('BusEq_equal', axis=1, inplace=True)
    lis = ['BusEq_non-equal',
            'CPI (Sticky Price Consumer Price Index)',
            'Cushing, OK WTI Spot Price FOB (Dollars per Barrel)',
            'DJI (Dow Jones Industrial Average)', 
            'INDPRO (Industrial Production)',
            'COMP (NASDAQ Composite Index)', 
            'SPX (S&P 500)',
            "WAAA (Moody's Seasoned Aaa Corporate Bond Yield)"]
    for col in lis:
        df_com[col] = arithmetic_fill(df_com[col])
    lis_2 = ['IOA',
              'Shares Out.', 
              'TEV/LTM Total Revenue_Average', 
              'P/LTM EPS_Average',
              'P/NTM EPS_Average', 
              'P/BV_Average', 
              'TEV/LTM Unlevered FCF_Average',
              'Market Cap/LTM Levered FCF_Average', 
              '  Return on Assets %',
              '  EBIT Margin %', 
              '  Gross Profit', 
              '  Total Assets' 
              ]
    for col in lis_2:
        df_com[col].fillna(method='ffill', inplace=True)
    df_com['company code'] = com_code
    Indiv_dic[com_code] = df_com

obj = Indiv_dic
df_1 = pd.read_excel('1.xlsx')
df_1.index = df_1.iloc[:,0]
df_1.drop('Company', axis=1, inplace=True)
df_1.set_axis(col_ori, axis=1, inplace=True)

for com in df_1.index:
    df = obj[com]
    for i in col_ori:
        df.at[pd.Timestamp('2010-01-01'),i] = df_1.at[com,i]
        df.loc[:,i].fillna(method='ffill', inplace=True)
        df.loc[:,i].fillna(method='bfill', inplace=True)
    df.columns = map(str.strip, df.columns)
del obj['DELL']

file = open('database_non-stock.database', 'wb')
pickle.dump(obj, file)
file.close()