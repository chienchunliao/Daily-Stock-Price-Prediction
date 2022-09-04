# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, yfinance as yf
import os,re
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

filename_lis = os.listdir()
df_news_dic = {}
scor_dic = {}
for i in filename_lis:
    if i.split('.')[-1] == 'xls':
        c_name = i.split('_')[1].replace(',', '').replace('.', '')
        c_code = company_info_dic[c_name]
        df = pd.read_excel(i)
        df = df.dropna().iloc[1:,:]
        time = pd.to_datetime(df.iloc[:,1])
        df.index = time
        df = df.iloc[:,:-1]
        df = df.set_axis(['News'], axis=1)
        def trans_content(cell):
            new_cont = ' '.join(cell.split('\n')[1:])
            #print(cell.split('\n')[0])
            return new_cont
        df = df.applymap(trans_content)
        analyzer = SentimentIntensityAnalyzer()
        scores = df['News'].apply(analyzer.polarity_scores)
        df['News Score_indiv'] = pd.Series(scores)
        fun = lambda x: x['compound']
        df['News Score_indiv'] = df['News Score_indiv'].map(fun)
        def reset_date(date):
            start = date.floor(freq='D') 
            end = start + pd.Timedelta(16,'h')
            if (date >= start) & (date < end):
                return start
            else:
                return start + pd.Timedelta(1,'D')
        df.index = df.index.map(reset_date)
        time_index = (df.index > datetime(2009,12,31)) & (df.index < datetime(2020,1,1))
        df = df.loc[time_index,:]
        df = df['News Score_indiv']
        df = df.groupby(df.index).sum()
        
        df_stock = yf.download('AAPL',start='2010-01-01', end='2019-12-31', progress=False)
        df_com = pd.merge(df_stock, df, left_index=True, right_index=True, how='outer')

        df_new = df_com[df_com['Open'].isnull()]['News Score_indiv']
        def rearrange(s_in):
            s = s_in.copy()
            cut = []
            delta = pd.Timedelta(1, 'd')
            temp = []
            for ind in s.index:
                if (ind-delta not in s.index) & (ind+delta in s.index):
                    upp = ind
                    temp.append(upp)
                if (ind+delta not in s.index) & (ind-delta in s.index):
                    low  = ind
                    temp.append(low)
                if len(temp) == 2:
                    cut.append(temp)
                    temp = []
            for up,low in cut:
                s_t = s[up:low]
                x = s_t.sum()
                s[low+delta] = x
            return s
        df_new_1 = rearrange(df_new)
        df_com = pd.merge(df_com, df_new_1, left_index=True, right_index=True, how='outer')
        df_com['News Score_indiv_y'].fillna(0, inplace=True)
        df_com['News Score_indiv'] = df_com['News Score_indiv_x'] + df_com['News Score_indiv_y']
        df_com = df_com.drop(['News Score_indiv_x', 'News Score_indiv_y'], axis=1)
        df_com.dropna(inplace=True)
        df_com = df_com['News Score_indiv']
        df_news_dic[c_code]=df_com
#%%        
import pickle
file = open('News Score_Individual_CIQ.subdata', 'wb')
pickle.dump(df_news_dic, file)
file.close()
        
