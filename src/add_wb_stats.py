from inv_dict import wb_cow_dict
import pandas as pd
import numpy as np



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
      

populated_vars = ['Adolescent fertility rate (births per 1,000 women ages 15-19)', 
'Age dependency ratio (% of working-age population)', 
'Birth rate, crude (per 1,000 people)', 
'Death rate, crude (per 1,000 people)', 
'Fertility rate, total (births per woman)',
'Land area (sq. km)', 'Life expectancy at birth, female (years)',
'Life expectancy at birth, male (years)', 
'Mortality rate, adult, male (per 1,000 male adults)',
'Population ages 0-14 (% of total population)', 
'Population growth (annual %)',
'Rural population (% of total population)', 
'Urban population growth (annual %)']

hp_vars = ['GDP (constant 2010 US$)', 
'Gross national expenditure (% of GDP)', 
'GINI index (World Bank estimate)']

five_year_impute = ['International migrant stock, total', 
'Mobile cellular subscriptions (per 100 people)']

if __name__ == '__main__':

    #populated_vars.extend(hp_vars)
    variable_list = populated_vars
    reign_df = pd.read_pickle('../data/year_agg.pkl')
    wdi_df = pd.read_pickle('../data/wdi_complete.pkl')
    joint_df = add_wd_rows(reign_df, wdi_df, variable_list)
    joint_df.to_pickle('../data/joint_df.pkl')






