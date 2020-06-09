import numpy as np
import pandas as pd
from inv_dict import wb_cow_dict
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
import pickle

def get_cc(val):
    if val in wb_cow_dict:
        return wb_cow_dict[val]
    else:
        return 0
    
def get_year(val):
    return int(val)

def add_wd_rows(reign_df, wdi_df, variable_list):
    '''
    pulls the indicators in variable list from the wdi_df, assigns them the country
    code from REIGN, generates a year-code based on the combination of the year 
    '''
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
                          random_state=30)
    upsampled = pd.concat([no_coup, coups_upsampled])
    y_up = upsampled[target]
    X_up = upsampled.drop(target, axis = 1)
    return X_up, y_up


def downsampler(X_train, y_train, target = 'pt_attempt'):
    '''
    Args: X_train and y_train
    Optional: what is the target
    Returns: y_train, and X_train with the non-target rows sampled with replacement to equal 
    the number of target rows (makes X_train much smaller)

    '''
    X = pd.concat([X_train, y_train], axis=1) 
    no_coup = X[X[target]==0]
    coup = X[X[target]==1]
    coups_downsampled = resample(no_coup,
                          replace=True, # sample with replacement
                          n_samples=len(coup), # match number in majority class
                          random_state=29)
    downsampled = pd.concat([coup, coups_downsampled])
    y_down = downsampled[target]
    X_down = downsampled.drop(target, axis = 1)
    return X_down, y_down

def smoter(X_train, y_train, ratio = 1.0):
    '''
    Args: X_train and y_train
    Optional: ratio
    Returns: y_train, and X_train with new target rows synthetically added to equal 
    the number of target rows (makes X_train much smaller) (or a different)
    '''
    sm = SMOTE(random_state=29, ratio=ratio)
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
    return X_train_sm, y_train_sm


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

def prepare_dataframe(reign_df, wdi_df, variable_list, drop_list):
    '''
    merges the prepared aggregated yearly reign data, and the selected columns from the 
    world development indicator dataframe, drops the variable in the drop list
    '''
    dummies = pd.get_dummies(reign_df['government'])
    df_dumb = reign_df.join(dummies)
    df_dumb['pt_attempt'] = df_dumb['coupyear']
    df_dumb['pt_suc'] = df_dumb['coupsuc']
    df = df_dumb.drop(['ccode', 'leader', 'month', 'government', 'coupyear', 'coupsuc'], axis = 1)
    joint_df = add_wd_rows(df, wdi_df, variable_list)
    joint_df_drops = joint_df.drop(drop_list, axis = 1)
    joint_df_x = joint_df_drops.dropna().copy()
    country_year_idx = joint_df_x[['country', 'year']]
    joint_df_x = joint_df_x.drop('country', axis =1)
    joint_df_x['constant'] = 1
    return joint_df_x, country_year_idx  
     
    
def apply_model(df, model):
    '''
    applies the passed in model to the passed in dataframe,
    currently designed to only predict on pt_attempt with upsampling
    and standard scaling, given that this is what worked best.
    '''
    y = df['pt_attempt']
    X = df.drop(['pt_attempt','pt_suc'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25, random_state= 40, stratify = y)
    X_up, y_up = upsampler(X_train, y_train, ratio = 1)
    pipe = Pipeline([('scaler', StandardScaler()),('model', model)])
    pipe.fit(X_up, y_up)
    weights = get_feature_weights(model, X.columns)
    metric_test(pipe, X_test, y_test)
    return model, weights

def get_prediction(df, idx, country, year, model, scaler):
    
    index = idx[(idx['country'] == country) & (idx['year'] == year)].index[0]
    row = df[df.index == index].drop(['pt_attempt', 'pt_suc'], axis = 1)
    row_scaled = scaler.transform(row)
    return model.predict_proba(row_scaled)[:, 1][0]


if __name__ == '__main__':

    variable_list = ['Life expectancy at birth, female (years)', 'GDP growth (annual %)', 'Mineral rents (% of GDP)', 'Oil rents (% of GDP)', 'Trade (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)', 'Natural gas rents (% of GDP)', 'Population ages 0-14 (% of total population)', 'Rural population (% of total population)',  'Population growth (annual %)', 'Arable land (hectares per person)',
    'Merchandise exports (current US$)',
    'Merchandise imports (current US$)',
    'Primary education, duration (years)']

    #these were calculated by iteratively using a LASSO regression and seeing what features were pulled to 0
    revised_drops = ['age',
    'tenure_months',
    'yearcode', 
    'Personal Dictatorship', 
    'exec_ant', 
    'leg_recent',
    'nochange_recent', 
    'lastelection',
    'lead_recent',
    'victory_recent',
    'ref_recent',
    'irregular',
    'Primary education, duration (years)',
    'direct_recent',
    'Merchandise imports (current US$)',
    'Foreign direct investment, net inflows (% of GDP)', 
    'elected', 
    'Presidential Democracy']

    wdi_df = pd.read_pickle('../data/wdi_complete.pkl')
    reign_df = pd.read_pickle('../data/year_agg.pkl')

    df, idx = prepare_dataframe(reign_df, wdi_df, variable_list, revised_drops)

    df_with_pt = df.join(idx, rsuffix='_idx')
    df_with_pt.to_pickle('../data/pickles/model_ready_df.pkl')

    elastic_scaled = LogisticRegressionCV(
            cv=5, dual=False,
            penalty='elasticnet', 
            scoring='recall',
            solver='saga', 
            n_jobs = 2,
            tol=0.001,
            max_iter=200,
            l1_ratios = [0, .3, .5, .7, 1])

    model, weights = apply_model(df, elastic_scaled)
    pipe = Pipeline([('scaler', StandardScaler()),('elastic_scaled', model)])

    hypo_row = df[df.index == 75].copy().drop(['pt_attempt', 'pt_suc'], axis = 1) 

    #pred = get_prediction(df, idx, 'USA', 2016, model, scaler)
    #print(pred)

    #filename = '../data/pickles/elastic_scaled.sav'
    #pickle.dump(model, open(filename, 'wb'))

    

        
        