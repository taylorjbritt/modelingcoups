import numpy as np
import pandas as pd

if __name__ == '__main__':
# read in dataframes
    reign_df = pd.read_csv('../data/REIGN_2020_5.csv')
    nmc_df = pd.read_csv('../data/NMC_5_0.csv')
    trade_df = pd.read_csv('../data/COW_trade_3.0/national_trade_3.0.csv')

# add yearcode columns to link the dataframes
    reign_df['yearcode'] = reign_df['ccode']*10000 + reign_df['year']
    nmc_df['yearcode'] = nmc_df['ccode']*10000 + nmc_df['year']
    trade_df['yearcode'] = trade_df['ccode']*10000 + trade_df['year']

# join the dataframes on 'yearcode'
    joint_df = reign_df.join(trade_df.set_index('yearcode'), on='yearcode', how = 'inner', rsuffix = '_tradedf' )
    df = joint_df.join(nmc_df.set_index('yearcode'), on='yearcode', how = 'inner', rsuffix = '_nmcdf' )

# create an accurate population columns

    df['population'] = df['tpop']*1000

# create dummy variables for govt type
    dummies = pd.get_dummies(df['government'])
    df_dumb = df.join(dummies)
    drop_list = ['couprisk', 'pctile_risk', 'yearcode', 'country_tradedf', 'year_tradedf', 'alt_imports', 'alt_exports', 'source1', 'source2',
       'version', 'stateabb', 'ccode_nmcdf', 'year_nmcdf', 'version_nmcdf', 'tpop', 'ccode_tradedf']
    df_drop = df_dumb.drop(drop_list, axis = 1)

#pickling for some EDA which used the dummy variables
    df_drop.to_pickle('../data/pickles/df_drop.pkl')

#drop the dummy variable ['Parliamentary Democracy'] and assign to a new DF

df_one_hot = df_drop.drop('Parliamentary Democracy', axis =1 )

#do some more feature engineering - add trade balance and percent of the population that is urban

df_one_hot['trade balance'] = df_one_hot['exports'] - df_one_hot['imports']
df_one_hot['urban_percent'] = (df_one_hot['upop']*1000)/df_one_hot['population']
df_one_hot['mil_percent'] = (df_one_hot['milper']*1000)/df_one_hot['population']


#drop the remaining non-integer variables to prepare for a random forest
drop_list_2 = ['ccode', 'country', 'leader', 'government']

df_one_hot_num = df_one_hot.drop(drop_list_2, axis = 1)

df_one_hot_num.to_pickle('../data/pickles/df_one_hot_num.pkl')





