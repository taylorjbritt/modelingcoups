import numpy as np
import pandas as pd
from inv_dict import wb_cow_dict

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

def get_cc(val):
    if val in wb_cow_dict:
        return wb_cow_dict[val]
    else:
        return 0
    
def get_year(val):
    return int(val)

def add_wd_rows(reign_df, wdi_df, variable_list):
    joint_df = reign_df.copy()
    yearlist = [str(i) for i in np.arange(1960, 2020)]
    for i in variable_list:
        df = wdi_df[wdi_df['Indicator Name'] == i]
        dfx = pd.melt(df, id_vars = ['Country Name'], value_vars=yearlist)
        dfx['ccode'] = dfx['Country Name'].apply(get_cc)
        dfx['year'] = dfx['variable'].apply(get_year)
        dfx['yearcode'] = (dfx['year']) + 10000*dfx['ccode']
        dfx[i] = dfx['value']
        dfx_limited = dfx[[i, 'yearcode']]
        joint_df = joint_df.join(dfx_limited.set_index('yearcode'), on='yearcode', how = 'inner')
    return joint_df

def upsampler(X_train, y_train, target = 'pt_attempt', ratio = 1.0):
    '''
    Args: X_train and y_train
    Optional: what is the target
    Returns: y_train, and X_train with the target rows sampled with replacement to equal 
    the number of non-target rows (makes X_train much bigger)
    '''
    y_train = pd.Series(y_train)
    
    X = pd.concat([X_train, y_train], axis=1) 
    no_coup = X[X[target]==0]
    coup = X[X[target]==1]
    coups_upsampled = resample(coup,
                          replace=True, # sample with replacement
                          n_samples=int(len(no_coup)*ratio), # match number in majority class
                          random_state=29)
    upsampled = pd.concat([no_coup, coups_upsampled])
    y_up = upsampled[target]
    X_up = upsampled.drop(target, axis = 1)
    return X_up, y_up

def metric_test(model, X_test, y_test):
    '''
    Prints out the accuracy, recall, precision, and f1 score for the 
    fit model when it predicts on the test data
    '''
    preds = model.predict(X_test)
    print('accuracy = ' + str(accuracy_score(y_test, preds)))
    print('recall = ' + str(recall_score(y_test, preds)))
    print('precision = ' + str(precision_score(y_test, preds)))
    print('f1 score = ' + str(f1_score(y_test, preds)))
    
def get_feature_weights(model, feature_labels):
    '''
    returns coefficients for features in a model (intended for logistic regression) 
    args: model, feature_labels
    returns: a sorted series in ascending order of feature weights.
    '''
    d_log_vals = {}
    for idx, feat in enumerate(model.coef_[0]):
        d_log_vals[feature_labels[idx]] = feat  
    s_log_vals = (pd.Series(d_log_vals)).sort_values()
    return s_log_vals


if __name__ == '__main__':

    variable_list = ['Life expectancy at birth, female (years)', 'GDP growth (annual %)', 'Mineral rents (% of GDP)', 'Oil rents (% of GDP)', 'Trade (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)', 'Natural gas rents (% of GDP)', 'Population ages 0-14 (% of total population)', 'Rural population (% of total population)',  'Population growth (annual %)', 'Arable land (hectares per person)',
    'Merchandise exports (current US$)',
    'Merchandise imports (current US$)',
    'Primary education, duration (years)']

    l1drops = ['direct_recent', 'Merchandise imports (current US$)', 'Foreign direct investment, net inflows (% of GDP)', 'elected', 'Presidential Democracy']


    new_drops = ['ref_recent',
    'ref_ant',
    'Party-Military',
    'Party-Personal-Military Hybrid',
    'Personal Dictatorship',
    'Provisional - Military',
    'Warlordism',
    'anticipation',
    'tenure_months',
    'militarycareer',
    'age',
    'Natural gas rents (% of GDP)',
    'Rural population (% of total population)',
    'Arable land (hectares per person)',
    'lead_recent',
    'exec_ant',
    'leg_ant',
    'leg_recent',
    'indirect_recent',
    'election_now',
    'Merchandise exports (current US$)',
    'precip',
    'defeat_recent',
    'prev_conflict',
    'exec_recent',
    'loss',
    'delayed',
    'change_recent']

    wdi_df = pd.read_pickle('../data/wdi_complete.pkl')
    reign_df = pd.read_pickle('../data/year_agg.pkl')
    dummies = pd.get_dummies(reign_df['government'])
    df_dumb = reign_df.join(dummies)
    df_dumb['pt_attempt'] = df_dumb['coupyear']
    df_dumb['pt_suc'] = df_dumb['coupsuc']
    df = df_dumb.drop(['ccode', 'country', 'leader', 'month', 'government', 'coupyear', 'coupsuc'], axis = 1)
    joint_df = add_wd_rows(df, wdi_df, variable_list)
    joint_df_x = joint_df.dropna()

    y = joint_df_x ['pt_attempt']
    X = joint_df_x .drop(['pt_attempt','pt_suc'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25, random_state= 40, stratify = y)
    ridge_scaled = LogisticRegressionCV(
            cv=5, dual=False,
            penalty='l1', 
            scoring='recall',
            solver='saga', 
            n_jobs = 2,
            tol=0.0001,
            max_iter=100,)
    X_up, y_up = upsampler(X_train, y_train, ratio = 1)

    logl1pipe = Pipeline([('scaler', StandardScaler()),('ridge_scaled', ridge_scaled)])
    logl1pipe.fit(X_up, y_up)