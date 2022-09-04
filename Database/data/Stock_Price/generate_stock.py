# -*- coding: utf-8 -*-

import pickle, yfinance as yf

stock_db = {'Train':{},
            'Test':{}}
company_lis = ['ACN', 'ADBE', 'AMD', 'ADI', 'AAPL', 'AVGO', 'HPQ', 'INTC', 'MA', 'MSFT', 'NTAP', 'NVDA', 'ORCL', 'QCOM', 'CRM', 'STX', 'TXN', 'V', 'WDC']
for company_code in company_lis:
    df_train = yf.download(company_code, start='2009-01-01', end='2016-12-31')
    df_test = yf.download(company_code, start='2017-01-01', end='2019-12-31')
    stock_db['Train'][company_code] = df_train
    stock_db['Test'][company_code] = df_test
file = open("database_stock.database", 'wb')
pickle.dump(stock_db, file)
file.close()