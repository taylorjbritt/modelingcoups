import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 'vertical');
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.savefig('../images/correlation_matrix')

if __name__ == '__main__':

    # checking out the REIGN dataset

    reign_df = pd.read_csv('../data/REIGN_2020_5.csv')

    #aggregate coup attempts/successes by year
    yearly_df = reign_df.groupby('year').sum()

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(yearly_df.index.values, yearly_df['pt_attempt'], color = 'blue')
    ax.bar(yearly_df.index.values, yearly_df['pt_suc'], color = 'red', width = .3)
    plt.xticks(rotation=45, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    ax.set_ylabel('Coups', fontsize = 16)
    ax.set_xlabel("Year", fontsize = 16)
    ax.set_title('Coups and Coup Attempts by Year', fontsize = 18)
    fig.savefig('../images/coupsyearly.png')


    #aggregate coup attempts/successes by country
    attempted_coups_bycountry = reign_df.groupby('country').sum()['pt_attempt'].sort_values()
    suc_coups_bycountry = reign_df.groupby('country').sum()['pt_suc'].sort_values()

    num_ = 15
    # plot the number of coups for a given number of countries
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(attempted_coups_bycountry.index[-num_:], attempted_coups_bycountry[-num_:], color = 'blue', label = 'Coups Attempts')
    ax.bar(attempted_coups_bycountry.index[-num_:], suc_coups_bycountry[-num_:], color = 'red', width = .5, label = 'Successful Coups')
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=45, Fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title("Attempted Coups – Top 15 Countries", fontsize = 18)
    ax.legend(loc = 'best', fontsize = 14)
    ax.set_ylabel('Coups', fontsize = 14)
    ax.set_xlabel("Country", fontsize = 16)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)

    fig.savefig('../images/coupsbycountry.png')

    #aggregate coup attempts by government type (I created a new dataframe for this for indexing consistency with the percentages)
    govt_grouped = reign_df.groupby('government').sum()[['pt_suc','pt_attempt']]
    govt_counts = reign_df.groupby('government').count()

    #calculate coup attempts divided by government types
    govt_grouped['pt_attempt_percent'] = govt_grouped['pt_attempt']/ govt_counts['leader']
    govt_grouped['pt_suc_percent'] = govt_grouped['pt_suc']/ govt_counts['leader']

    govt_grouped = govt_grouped.sort_values(['pt_attempt'])

    #plotting coups by government type
    #total

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt'], color = 'blue')
    ax.bar(govt_grouped.index.values, govt_grouped['pt_suc'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.yticks(rotation=0, fontsize = 14)
    plt.xticks(rotation=90, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (total)', fontsize = 18)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    fig.savefig('../images/coupsbygovttotal.png')

    #percentages
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt_percent'], color = 'blue')
    ax.bar(govt_grouped.index.values, govt_grouped['pt_suc_percent'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=90, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (percent)', fontsize = 18)
    fig.savefig('../images/coupsbygovtpercent.png')



    # add a column for tenure in years instead of months, and then aggregate by tenure
    reign_df['tenure_years'] = reign_df['tenure_months']//12
    attempted_coups_tenure = reign_df.groupby('tenure_years').sum()['pt_attempt'].sort_index()
    suc_coups_tenure = reign_df.groupby('tenure_years').sum()['pt_suc'].sort_index()

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(attempted_coups_tenure.index, attempted_coups_tenure, color = 'blue', label = 'Coups Attempts')
    ax.bar(attempted_coups_tenure.index, suc_coups_tenure, color = 'red', width = .5, label = 'Successful Coups')
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=0, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_xlim(-1, 25)
    ax.set_title("Coup Attempts vs Leader's Tenure, Years", fontsize = 18)
    ax.legend(loc = 'best')
    ax.set_ylabel('Coups', fontsize = 14)
    ax.set_xlabel("Leader's Tenure, Years", fontsize = 14)
    fig.savefig('../images/coupsbyleadertenure.png')


    #aggregate coup attempts/successes by year
    yearly_df = reign_df.groupby('year').sum()

    fig, ax = plt.subplots(figsize = (16, 8))
    ax.bar(yearly_df.index.values, yearly_df['pt_attempt'], color = 'blue', label = 'Coups Attempts')
    ax.bar(yearly_df.index.values, yearly_df['pt_suc'], color = 'red', width = .3, label = 'Successful Coups')
    plt.xticks(rotation=45, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    ax.set_ylabel('Coups', fontsize = 16)
    ax.set_xlabel("Year", fontsize = 16)
    ax.set_title('Coups and Coup Attempts by Year', fontsize = 18)
    ax.legend(loc = 'best')
    fig.savefig('../images/coupsyearlywider.png')

#plotting coups by government type
    #total

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt'], color = 'blue')
    ax.bar(govt_grouped.index.values, govt_grouped['pt_suc'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.yticks(rotation=0, fontsize = 14)
    plt.xticks(rotation=90, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (total)', fontsize = 18)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    fig.savefig('../images/coupsbygovttotal.png')

    #percentages
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt_percent'], color = 'blue')
    ax.bar(govt_grouped.index.values, govt_grouped['pt_suc_percent'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=90, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (percent)', fontsize = 18)
    fig.savefig('../images/coupsbygovtpercent.png')

    #aggregate coup attempts/successes by year
    yearly_df = reign_df.groupby('year').sum()

    fig, ax = plt.subplots(figsize = (16, 8))
    ax.bar(yearly_df.index.values, yearly_df['pt_attempt'], color = 'green', label = 'Coups Attempts')
    #ax.bar(yearly_df.index.values, yearly_df['pt_suc'], color = 'red', width = .3, label = 'Successful Coups')
    plt.xticks(rotation=45, fontsize = 20)
    plt.yticks(rotation=0, fontsize = 20)
    ax.set_ylabel('Coups', fontsize = 20)
    ax.set_xlabel("Year", fontsize = 20)
    ax.set_title('Coup Attempts by Year', fontsize = 24)
    #ax.legend(loc = 'best')
    fig.savefig('../images/coupsattemptsonlyyeary.png')

    # 
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt'], color = 'blue')
    #ax.bar(govt_grouped.index.values, govt_grouped['pt_suc'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.yticks(rotation=0, fontsize = 14)
    plt.xticks(rotation=90, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    #percentages

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt_percent'], color = 'blue')
    ax.bar(govt_grouped.index.values, govt_grouped['pt_suc_percent'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=90, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (percent)', fontsize = 18)
    fig.savefig('../images/coupsbygovtpercent.png')


    num_ = 15
    # plot the number of coups for a given number of countries
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(attempted_coups_bycountry.index[-num_:], attempted_coups_bycountry[-num_:], color = 'green', label = 'Coups Attempts')
    #ax.bar(attempted_coups_bycountry.index[-num_:], suc_coups_bycountry[-num_:], color = 'red', width = .5, label = 'Successful Coups')
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=45, Fontsize = 18)
    plt.yticks(rotation=0, fontsize = 18)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title("Attempted Coups – Top 15 Countries", fontsize = 24)
    #ax.legend(loc = 'best', fontsize = 14)
    ax.set_ylabel('Coups', fontsize = 20)
    ax.set_xlabel("Country", fontsize = 20)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)

    fig.savefig('../images/coupattempsonlybycountry.png')



    fig, axs = plt.subplots(2, figsize = (12, 8))
    fig.suptitle('Vertically stacked subplots')
    axs[0].bar(govt_grouped.index.values, govt_grouped['pt_attempt'], color = 'blue')
    axs[1].bar(govt_grouped.index.values, govt_grouped['pt_attempt_percent'], color = 'blue')
    ax.set_title('Coups and Coup Attempts by Government Type', fontsize = 18)
    fig.savefig('../images/coupsbygovtsubplots.png')
    plt.yticks(rotation=0, fontsize = 14)
    #plt.xticks(rotation=90, fontsize = 14)

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.index.values, govt_grouped['pt_attempt'], color = 'green')
    #ax.bar(govt_grouped.index.values, govt_grouped['pt_suc'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.yticks(rotation=0, fontsize = 14)
    plt.xticks(rotation=90, fontsize = 14)
    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    fig.tight_layout(pad = 3)
    ax.set_title('Coup Attempts by Government Type (Total)', fontsize = 18)
    fig.savefig('../images/coupsattemptsonlybygovttotal.png')

    #percentages

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(govt_grouped.sort_values('pt_attempt_percent').index.values, govt_grouped.sort_values('pt_attempt_percent')['pt_attempt_percent'], color = 'green')
    #ax.bar(govt_grouped.index.values, govt_grouped['pt_suc_percent'], color = 'red', width = .5)
    # N = len(attempted_coups_govtype.index)
    plt.xticks(rotation=90, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    ax.set_ylabel('Coups per Government Type-Years', fontsize = 16)

    # ticklocations = np.arange(0,N)
    # ax.set_xticks(ticks = ticklocations -1)
    ax.set_title('Coups and Coup Attempts by Government Type (Scaled)', fontsize = 18)
    fig.savefig('../images/coupsattemptsonlybygovtpercent.png')

    fig, ax = plt.subplots(figsize = (16, 8))
    ax.bar(yearly_df.index.values, yearly_df['pt_attempt'], color = 'green', label = 'Coups Attempts')
    #ax.bar(yearly_df.index.values, yearly_df['pt_suc'], color = 'red', width = .3, label = 'Successful Coups')
    plt.xticks(rotation=45, fontsize = 20)
    plt.yticks(rotation=0, fontsize = 20)
    ax.set_ylabel('Coups', fontsize = 20)
    ax.set_xlabel("Year", fontsize = 20)
    ax.set_title('Coup Attempts by Year', fontsize = 24)
    plt.tight_layout(pad=3, h_pad=None, w_pad=None, rect=None)
    #ax.legend(loc = 'best')
    fig.savefig('../images/coupsattemptsonlyyearly.png')




