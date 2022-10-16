import calendar
import pandas as pd
import numpy as np

def remove_high_corr(df,target='Israelis_Count',threshold=0.5):
  '''
  return dataframe without corrlation that can be drop.
  
  args:
  df = dataframe
  target = string of the target
  threshold = default 0.5
  '''
  target_col = df.pop(target)
  df.insert(len(df.columns), target, target_col)
  cor_matrix = df.corr().abs()
  corr_df = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
  #מתודה שאומרת בי בקורלורציה עם מי
  cols = corr_df.columns.to_list()
  list_corr_not_empty=[]

  for i in range(len(cols)-1):
      tmp = []
      for j in range(len(cols)-1):
        if abs(corr_df.iloc[i,j]) >= threshold and cols[i] is not cols[j] :
          tmp.append(cols[j])
      if len(tmp)>0:
          tmp.append(cols[i])
          list_corr_not_empty.append(tmp)
  def Key(p):
   return  corr_df[target][p]
  stay = [max(sub,key=Key) for sub in list_corr_not_empty]
  drops = [ c for sub in list_corr_not_empty for c in sub if c not in stay ]
  return df.drop(list(set(drops)),axis=1)
  pass

def remove_outliers(df,target_name='Israelis_Count'):
  '''
  return df without outliers.
  
  args:
  df = dataframe
  target = string of the target
  '''
  import matplotlib.pyplot as plt
  plt.cla()
  bp = plt.boxplot(df[target_name])
  minimums = [round(item.get_ydata()[0], 4) for item in bp['caps']][::2]
  maximums = [round(item.get_ydata()[0], 4) for item in bp['caps']][1::2]
  return df.drop(df [ (df[target_name]>maximums[0])  | (df[target_name]<minimums[0])].index)



def plot_line(prediction,actual,title='',path_save=None,file_name=None,fig_size_tuple=(18,8),xlim=None,ylim=None
,alpha_prediction=1,alpha_actual=1):
  '''plot the line graph of the resualts 
  title can be added with title
  if want to save add path_name and file_name

  arugments: prediction, actual, title='' ,path_save=None ,file_name=None , fig_size_tuple=(18,8) xlim=None,ylim=None

  example for saving:
  path_name= 'folder1/save_here_folder/'
  path_file = file_name.png
  '''
  import os
  from pylab import rcParams
  rcParams['figure.figsize'] = fig_size_tuple[0],fig_size_tuple[1]
  import matplotlib.pyplot as plt
  import pandas as pd
  res = pd.DataFrame(data={
    'Predictions':prediction,
    'Actual':actual
  })
  plt.plot(res.index, res['Predictions'], color='r', label='Predicted Visitors',alpha=alpha_prediction)
  plt.plot(res.index, res['Actual'], color='b', label='Actual Visitors',alpha=alpha_actual)
  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.title(title, family='Arial', fontsize=26)
  plt.ylabel('Visitors', family='Arial', fontsize=22)
  plt.xticks(rotation=45, fontsize=16)
  plt.yticks(rotation=45, fontsize=16)
  plt.xlim(xlim)
  plt.ylim(ylim)
  
  if path_save is not None:
    isExist = os.path.exists(path_save)
    if not isExist:
      os.makedirs(path_save)
    plt.savefig(path_save+file_name)
  plt.show()



def plot_residuals(prediction,actual,title='',path_save=None,file_name=None,fig_size_tuple=(18,8),xlim=None,ylim=None):
  '''plot the residuales of the resualts 
  if want to save add path_name and file_name
  
  arugments: prediction, actual, title='' ,path_save=None ,file_name=None, fig_size_tuple=(18,8) xlim=None,ylim=None
  example:
  path_name= 'folder1/save_here_folder/'
  path_file = file_name.png
  '''
  import os
  from pylab import rcParams
  rcParams['figure.figsize'] = fig_size_tuple[0],fig_size_tuple[1]
  import matplotlib.pyplot as plt
  import pandas as pd
  res = pd.DataFrame(data={
    'Predictions':prediction,
    'Actual':actual
  })
  res['residuals'] = res['Predictions'] - res['Actual']
  plt.plot(res.Predictions,res.residuals,color='r',marker='.',linestyle='None')
  plt.xlabel('Visitors', family='Arial', fontsize=22)
  plt.ylabel('Residuals', family='Arial', fontsize=22)
  plt.plot(res.Predictions,res.residuals*0,color='b')
  plt.title(title, family='Arial', fontsize=26)
  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.yticks(rotation=45, fontsize=16) 
  plt.xlim(xlim)
  plt.ylim(ylim)
  
  
  plt.xticks(rotation=45, fontsize=16)
  if path_save is not None:
    isExist = os.path.exists(path_save)
    if not isExist:
      os.makedirs(path_save)
    plt.savefig(path_save+file_name)
  plt.show()



def split_date(dataframe):
  '''
  split the date in the df to columns years
  month and days 
  
  return df with column year,month,day
  '''

  import pandas as pd
  dataframe = dataframe.set_index("Date")
  dataframe['day'] = dataframe.index.day
  dataframe['month'] = dataframe.index.month
  dataframe['year'] = dataframe.index.year
  dataframe.reset_index(drop=False,inplace=True)

  return dataframe

def get_rmse(x,y):
  from sklearn.metrics import mean_squared_error
  from math import sqrt
  return sqrt(mean_squared_error(x,y))

def remove_unique_one(df):
  '''
  remove columns with 1 feature only
  
  return df without columns with 1 feature only 
  '''
  drop_one_unique = [x for x in df.columns if len(df[x].value_counts())==1]
  return df.drop(drop_one_unique,axis=1)


  
def remove_pollution_site(dataset):
  '''
  remove the feature
   'nox','pm10','pm2.5','so2','is_Site_exceeded_pm10','is_Site_exceeded_pm2.5', 'is_Site_exceeded_nox','is_Site_exceeded_so2'
  '''
  print('remove pollution site Successfully')
  return dataset.drop(['nox','pm10','pm2.5','so2','is_Site_exceeded_pm10','is_Site_exceeded_pm2.5', 'is_Site_exceeded_nox','is_Site_exceeded_so2'],axis=1)



def move_target_to_last(dataset,target='Israelis_Count'):
  t = dataset[target]
  dataset.drop(target,axis=1,inplace=True)
  dataset[target] = t
  return dataset

def remove(df , to_remove):
  cols = df.columns
  if to_remove in cols:
    df.drop(to_remove , inplace=True , axis = 1)

  return df


def get_weekday(dataset):

  dataset['week_Day'] = dataset.Date.apply(lambda date : calendar.day_name[date.weekday()])
  days = pd.get_dummies(dataset['week_Day'])
  dataset = pd.concat([dataset , days] , axis=1)
  dataset = remove(dataset , 'week_Day')
  return dataset

  def add_last_visitors_for_all_sites_in_df(df,target='Israelis_Count'):
    '''
    use the method last_year_entries_info for each site in the dataframe

    return df
    '''
    dataset = df.copy()
    dataset['Last_year_visitors_IL'] = 0
    sites = dataset.Site_Name.unique()
    dataset = function.split_date(dataset)
    dataset = function.move_target_to_last(dataset, target)
    dataset = dataset.sort_values(['year','month','day'])
    for site in sites:
      print(site)
      site_dataset = dataset.loc[dataset.Site_Name==site]
      site_dataset = function.last_year_entries_info(site_dataset,target)
      # print(site_dataset.Last_year_visitors_IL  )
      dataset.loc[dataset.Site_Name==site,'Last_year_visitors_IL'] = site_dataset.Last_year_visitors
      pass

    print('**********************************************')
    print('Add All Sites Last year visitors Successfully')
    print('**********************************************')
    return dataset 
from datetime import date
def date_diff(a , b, target=1):

    year,month,day = a.year,a.month,a.day
    d0 = date(year,month,day)

    year,month,day = b.year,b.month,b.day
    d1= date(year,month,day)

    delta = d1 - d0
    if abs(delta.days) == target:
        return True
    else : return False
    
def outputLimeAsDf(exp):
  d = {}
  for i in exp.as_list():
      sign = 1
      if('=0' in i[0]):
          sign = -1
      value = i[-1]*sign
      name = i[0].replace('<','')
      name = name.replace('>','')
      name = name.replace('=','')
      name = name.replace(' ','')
      name = name.replace('.','')
      name = name.replace('0','')
      name = name.replace('1','')
      name = name.replace('2','')
      name = name.replace('3','')
      name = name.replace('4','')
      name = name.replace('5','')
      name = name.replace('6','')
      name = name.replace('7','')
      name = name.replace('8','')
      name = name.replace('9','')
      name = name.replace('pm','pm10')
      name = name.replace('_',' ')
      str.replace
      # print(i[0],i[-1])
      d[name]=value
  return pd.DataFrame.from_dict(d,orient='index')
def printOutputeCoef(coef):
  d = {}
  for tup in coef:
      d[tup[-1]] = tup[0]
  return pd.DataFrame.from_dict(d,orient='index')





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
def mlrModelResTrainTestCoeff(dataframe,shaffle=False):
    '''
    return 3 df: train result, test results, coeff results. 
    get df as data, with Data columns!.
    '''
    dataframe.dropna(inplace=True)
    dataframe.sort_values('Date')
    y = dataframe[['Date','Israelis_Count']]
    X = dataframe.drop('Israelis_Count',axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shaffle, test_size = 0.2)

    X_train_scaler = MinMaxScaler()
    X_test_scaler = MinMaxScaler()

    X_train_scaled = X_train_scaler.fit_transform(X_train.drop('Date',axis=1))
    X_test_scaled = X_test_scaler.fit_transform(X_test.drop('Date',axis=1))
    
    mlr = linear_model.LinearRegression()
    mlr.fit(X_train_scaled,y_train.Israelis_Count)
    
    prediction = mlr.predict(X_train_scaled)
    resTrain =  pd.DataFrame(
        data={
            'Prediction':prediction,
            'Actual': y_train.Israelis_Count.values    },
        index=y_train.Date
    )
    
    prediction = mlr.predict(X_test_scaled)
    resTest = pd.DataFrame(
        data={
            'Prediction':prediction,
            'Actual': y_test.Israelis_Count.values    },
        index=y_test.Date
    )
    coef = sorted( list(zip(np.round(mlr.coef_,5).T,X_train.drop("Date",axis=1).columns)))
    d = {}
    for tup in coef:
        d[tup[-1]] = tup[0]
    coefDF = pd.DataFrame.from_dict(d,orient='index')
    
    return resTrain,resTest,coefDF
    
def printRes(res ,plotLine=True ,plotResiduals = False, n = 10):
    '''
    print results from df reuslts and n samples 
    '''
    res = res.sort_index()
    print('rmse',get_rmse(res.Prediction, res.Actual))
    print('std',np.std(res.Actual))
    
    if plotResiduals:
        plot_residuals(actual=res.Actual,prediction=res.Prediction)
    if plotLine:
        plot_line(actual=res.Actual,prediction=res.Prediction)
        
    print('Sample rows:')
    print( res.sample(n))