
import numpy as np
import pandas as pd
#from scipy.stats import randint
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from pandas import set_option
#plt.style.use('ggplot')
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import RFE
#from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
#from xgboost import XGBClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel
#from sklearn import metrics
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
#from sklearn.metrics import classification_report
BankCredit = pd.read_csv("UCI_Credit_Card.csv")
print(f'The shape of the dataframe is {BankCredit.shape}')
print()
print(BankCredit.info())
print()
BankCredit.replace(to_replace='?', value=np.NaN, inplace=True)
print()
print(BankCredit.describe())
print()
print(BankCredit['ID'].value_counts())
#BankCredit.isnull().sum(),sns.countplot(x='ID', data=BankCredit, linewidth=3)
#plt.show()
BankCredit[['AGE','EDUCATION','MARRIAGE','SEX','PAY_0','BILL_AMT1']].hist(bins=60,figsize=(15,8))
plt.show()
