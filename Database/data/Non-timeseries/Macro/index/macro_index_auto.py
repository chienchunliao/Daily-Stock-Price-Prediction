import pandas as pd, os, pickle

file_lis = os.listdir()
df_lis = []
for name in file_lis:
    if name.split('.')[-1] == 'xls':
        df = pd.read_excel(name)
        df.index = df.iloc[:,0]
        df = df.iloc[:,1:]
        time_index = (df.index > pd.Timestamp(2009,12,31)) & (df.index < pd.Timestamp(2020,1,1))
        df = df.loc[time_index,:]
        df_lis.append(df)
     
df_MacroInd_comb = df_lis[0]
for df in df_lis[1:]:
    df_MacroInd_comb = df_MacroInd_comb.join(df, how='outer')

file = open('macro_index.subdata', 'wb')
pickle.dump(df_MacroInd_comb, file)
file.close()