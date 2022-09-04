# -*- coding: utf-8 -*-

import pandas as pd, pickle, time, re, copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

def get_all_file(ftype='.txt'):
    import os, glob
    file_lis = []
    for path in glob.glob('**', recursive=True):
        if os.path.isfile(path) and ftype in path:
            file_lis.append(path)
    return file_lis

#%% Loading File
df_lis = []
for path in get_all_file('.majordb'):
    file = open(path, 'rb')
    model_name_lis, df = pickle.load(file)
    file.close()

for m in combinations(model_name_lis,len(model_name_lis)-1):
    df_lis.append(df.drop(list(m), axis=1))


#%% Calculating Correlations Between Varia
corr = df.corr()

#%% Auto-training and Auto-tuning Process
df_grid_dic = {}
result_lis = []
inde = ['TS Model', 'Algorithm', 'PCA/NonPCA', 'Hyper Parameters', 'Test Scrore(MSE)', 'Test Scrore(R^2)', 'Time Cosumtion', 'Predicted Y', 'True Y', 'Best model']
for df in df_lis:
    train_index = (df.index >= pd.Timestamp('2010-01-01')) & (df.index <= pd.Timestamp('2016-12-31'))
    test_index = (df.index >= pd.Timestamp('2017-01-01')) & (df.index <= pd.Timestamp('2019-12-31'))
    df = df.drop(['company code', 'DJI (Dow Jones Industrial Average)', 'SPX (S&P 500)', 'P/LTM EPS_Average'],axis=1)
    df_train = df.loc[train_index, :]
    df_test = df.loc[test_index, :]
    df_train.index = range(df_train.shape[0])
    df_test.index = range(df_test.shape[0])
    x_train = df_train.iloc[:,:-1]
    x_test = df_test.iloc[:,:-1]
    y_train = df_train.iloc[:,-1]
    y_test = df_test.iloc[:,-1]
    
    TS_model = df.columns[-2]
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    n = -1
    cv = 5
    param = {"n_components": list(range(10, 21))}
    pca = PCA()
    grid_search = GridSearchCV(pca, param, cv=5, return_train_score=True)
    grid_search.fit(x_train_scaled, y_train)
    PCA_cv = list(grid_search.best_params_.values())[0]
    pca1 = PCA(n_components=PCA_cv)
    pca1.fit(x_train_scaled, y_train)
    x_train_reduced = pca1.fit_transform(x_train_scaled)
    x_test_reduced = pca1.fit_transform(x_test_scaled)
    pca_nonpca = [['PCA',x_train_reduced,y_train, x_test_reduced,y_test], ['Non-PCA',x_train_scaled, y_train, x_test_scaled, y_test]]
    for pca_state, x_train, y_train, x_test, y_test in pca_nonpca:
        model_dic = {'LR': [LinearRegression(n_jobs=n), 
                            {
                                }
                            ],
                      'Lasso': [Lasso(), 
                                {'alpha' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
                                }
                                ], 
                      'Ridge': [Ridge(), 
                                {'alpha' :  [0.001, 0.01, 0.1, 1, 10,100]
                                }
                                ], 
                      'KNN_R': [KNeighborsRegressor(n_jobs=n), 
                                {'n_neighbors': range(1, 26)
                                }
                                ], 
                      'GB_R': [GradientBoostingRegressor(), 
                              {"n_estimators": [100, 200, 300, 400, 500],
                                "learning_rate":[0.2, 0.4, 0.6, 0.8, 1]
                                }
                              ], 
                      'LinearSVR': [LinearSVR(), 
                                    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 100000]
                                    }
                                    ], 
                      'SVR': [SVR(), 
                              {'kernel': ['poly', 'rbf'],
                               'C': [0.001, 0.01, 0.1, 1, 10, 100, 10000],
                               'gamma': [0.0001, 0.001,0.001,0.1,1,10]
                              }
                              ], 
                      'RF_R': [RandomForestRegressor(n_jobs=n), 
                              {'max_depth': [5, 10, None],
                                'max_features': ['auto', 'log2'],
                                'n_estimators': [20,50,100,150,200,300]
                                }
                              ]
                      }
        #model_try = re.split('[, ]', input("Input the model you want to try (LR,Lasso, Ridge, KNN_R, GB_R, LinearSVR, SVR, RF_R):\nFormatt as 'LR,KNN_R'"))
        for model_name in model_dic:
            model = copy.deepcopy(model_dic[model_name][0])
            param_dic = model_dic[model_name][1]
            t1 = time.time()
            model_cv = GridSearchCV(model, param_dic, cv=cv, n_jobs=n)
            model_cv.fit(x_train, y_train)
            best_param = model_cv.best_params_
            best_model = model_cv.best_estimator_
            pred = pd.Series(model_cv.best_estimator_.predict(x_test))
            test_score_mse = mean_squared_error(y_test, pred)
            test_score_r2 = r2_score(y_test, pred)
            t2 = time.time()
            timeuse = t2-t1
            se_temp = pd.Series([TS_model, model_name, pca_state, best_param, test_score_mse, test_score_r2 , timeuse, pred, y_test, best_model], 
                                index = inde)
            result_lis.append(se_temp)
            del model
            
#%% Generating the Result DataFrame
df_model_result = pd.DataFrame(result_lis, columns=inde)
df_model_result_bestN = df_model_result.sort_values(by='Test Scrore(R^2)', ascending=False).iloc[:5,:]

#%% Output the result
df_model_result.to_csv('Final_Result_all.csv')
with open('Final_Result_all.result', 'wb') as f:
    pickle.dump(df_model_result, f)

df_model_result_bestN.to_csv('Final_Result_top5.csv')
with open('Final_Result_top5.result', 'wb') as f:
    pickle.dump(df_model_result_bestN, f)

        

    