# -*- coding: utf-8 -*-
import pandas as pd, re, os

def trans_timestamp(cell):
    import pandas as pd
    try:
        time = pd.to_datetime(cell, format='%Y%m%d')
    except:
        time = cell
    return time

for path in os.listdir():
    if re.search('.csv', path):
        df = pd.read_csv(path)
        df.dropna(how='all', inplace=True)
        df = df.iloc[:-1,:]
        sep = []
        for inde, val in df.iloc[:,0].iteritems():
            result = re.search('Average (Value)? ?(Equal)? ?Weighted', str(val))
            if result:
                if result.group(2):
                    sep.append([inde, '_equal'])
                else:
                    sep.append([inde, '_non-equal'])
        df_1 = df.loc[sep[0][0]+1:sep[1][0]-1,:]
        df_2 = df.loc[sep[1][0]+1:,:]
        df_1.fillna(sep[0][1], inplace=True)
        df_2.fillna(sep[1][1], inplace=True)
        
        df_1.columns = df_1.iloc[0,:] + sep[0][1]
        df_2.columns = df_2.iloc[0,:] + sep[1][1]
        
        df_1.index = df_1.iloc[:,0].apply(trans_timestamp)
        df_2.index = df_2.iloc[:,0].apply(trans_timestamp)
        
        df_1 = df_1.iloc[1:,1:]
        df_2 = df_2.iloc[1:,1:]
        
        df_out = df_1.join(df_2)[['BusEq_non-equal', 'BusEq_equal']]
        df_out.index.name = None
        start = pd.Timestamp('2009-12-31')
        end = pd.Timestamp('2020-01-01')
        time_fil = (df_out.index > start) & (df_out.index < end)
        df_out = df_out.loc[time_fil,:]
        df_out = df_out.astype('float64')
        df_out.to_pickle('industry.subdata')