import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv('../data/REIGN_2020_5.csv')
    df['yearcode'] = df['ccode']*10000+df['year']
    coupcount = df.groupby('yearcode').sum()
    coup_agg = coupcount[['pt_attempt','pt_suc']]
    yearly = df[df['month'] == 1]
    coup_yearly = yearly.join(coup_agg, how = 'left', on = 'yearcode', rsuffix = '_year')
    coup_yearly_drops = coup_yearly.drop(['pt_suc',
        'pt_attempt', 'couprisk', 'pctile_risk', 'yearcode_year'], axis =1 )
    x = coup_yearly_drops.merge(coup_agg, on = 'yearcode')
    x.drop(['pt_attempt_year','pt_suc_year'], axis = 1, inplace = True)
    x['coupyear'] = x['pt_attempt'] > 0
    x['coupsuc'] = x['pt_suc'] > 0
    x = x.drop(['pt_attempt', 'pt_suc'], axis = 1)
    x.to_pickle('../data/pickles/year_agg.pkl')