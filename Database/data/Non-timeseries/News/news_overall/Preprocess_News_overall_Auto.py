# -*- coding: utf-8 -*-

import os, pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

def reset_date(date):
    start = date.floor(freq='D') 
    end = start + pd.Timedelta(16,'h')
    if (date >= start) & (date < end):
        return start
    else:
        return start + pd.Timedelta(1,'D')

df = pd.read_csv("Wall Street Journal Twitter.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].map(reset_date)
df.index = df['timestamp']
df.drop('timestamp', axis=1, inplace=True)

analyzer = SentimentIntensityAnalyzer()
scores = df['text'].apply(analyzer.polarity_scores)
df['News Score_overall_WSJ'] = pd.Series(scores)
df['News Score_overall_WSJ'] = df['News Score_overall_WSJ'].map(lambda x: x['compound'])
time_index = (df.index > pd.Timestamp(2009,12,31)) & (df.index < pd.Timestamp(2020,1,1))
df = df.loc[time_index,:]
df = df['News Score_overall_WSJ']
df = df.groupby(df.index).sum()

import yfinance as yf
df_stock = yf.download('AAPL',start='2010-01-01', end='2019-12-31', progress=False)
df_com = pd.merge(df_stock, df, left_index=True, right_index=True, how='outer')

df_new = df_com[df_com['Open'].isnull()]['News Score_overall_WSJ']

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
df_com['News Score_overall_WSJ_y'].fillna(0, inplace=True)
df_com['News Score_overall_WSJ'] = df_com['News Score_overall_WSJ_x'] + df_com['News Score_overall_WSJ_y']
df_com = df_com.drop(['News Score_overall_WSJ_x', 'News Score_overall_WSJ_y'], axis=1)
df_com.dropna(inplace=True)
df_com = df_com['News Score_overall_WSJ']

            

file = open('News Score_overall_WSJ.subdata', 'wb')
pickle.dump(df_com, file)
file.close()