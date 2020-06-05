import pandas as pd
import numpy as np


nmc_df = pd.read_csv('../data/NMC_5_0.csv')
trade_df = pd.read_csv('../data/COW_trade_3.0/national_trade_3.0.csv')
r_wb_df =  pd.read_pickle('../data/pickles/joint_df.pkl')

nmc_df['yearcode'] = nmc_df['ccode']*10000 + nmc_df['year']
trade_df['yearcode'] = trade_df['ccode']*10000 + trade_df['year']

joint_df = r_wb_df.join(trade_df.set_index('yearcode'), on='yearcode', how = 'inner', rsuffix = '_tradedf' )
df = joint_df.join(nmc_df.set_index('yearcode'), on='yearcode', how = 'inner', rsuffix = '_nmcdf' )

dummies = pd.get_dummies(df['government'])
df_dumb = df.join(dummies)
df_dumb['pt_attempt'] = df_dumb['coupyear']
df_dumb['pt_suc'] = df_dumb['coupsuc']
drop_list = ['yearcode', 'Parliamentary Democracy', 'ccode', 'country', 'leader', 'coupyear', 'coupsuc', 'government', 'country_tradedf', 'year_tradedf', 'alt_imports', 'alt_exports', 'source1', 'source2',
       'version', 'stateabb', 'ccode_nmcdf', 'year_nmcdf', 'version_nmcdf', 'tpop', 'ccode_tradedf']
df_maximus = df_dumb.drop(drop_list, axis = 1)

df_maximus.to_pickle('../data/pickles/df_maximus.pkl')







