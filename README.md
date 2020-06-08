# Modeling Military Coups 

![Whipala](images/misc/whipala.png)
A protestor with a flag representing the indigenous communities of the Andes makes an entreaty to solidiers in Bolivia in 2019. [Source](https://www.mintpressnews.com/media-silent-bolivia-massacre-indigenous-protesters/262858/)

## Introduction

I spent a few years living in Argentina and travelled around a few other countries in South America, and I was always struck by how much of their 20th century history was defined by military coups, which often led to periods of chaos, repression, and terrible atrocities. The removal of Evo Morales in Bolivia last year and the somewhat farcical attempted overthrow of the government in Venezuela this month showed that these events are ongoing. 

I recently read a Soviet science fiction novel in which scientists from Earth travel to another planet that is also populated by humans, but mired in a dark ages level of technology and social organization. When one of the scientists (posing as a local semi-divine lord) notices an emerging fascist movement in the city he lives in, the other scientists ignore his pleas for help, as this political development contradicts their models of how human civilization develops. 

These two topics provoked me to wonder if I could apply machine learning techniques to predict when coups happen in different countries based on economic and political statistics. Fortunately, I discovered that political scientists were also interested in this question, and had created a dataset that tracked military coups called REIGN. I decided to try to create my own model using REIGN and supplementary data to see I could model coups fairly accurately and see what features influenced the occurence of coups. 

## The Data

I used the REIGN dataset (see citation below) from One Earth Future for details on country leaders, government types, and when successful and attempted coups occured. I supplemented this with indicators from the Correlates of War project, which had information on population, trade, and militarization. 

The Reign dataset contained 135,637 rows covering January 1950 unitl March 2020, with 200 countries, 466 attempted coups (successful or unsuccessful) and 233 successful ones. 

By filtering the dataframe for months in which attempted coups took place and a certain country, it was sometimes possible to trace their tumultuous journey through the latter half of the 20th century. For instance, looking at Argentina below, you can see the overthrow of Juan Perón in 1955, followed by a decade and a half of successive coups. Perón returned to Argentina in 1973 but died the following year – his Vice President and wife, Isabel, took power, but was overthrown in 1976 and replaced by a junta  

![argentina_df](images/misc/argentina.png)

To supplement the REIGN dataset, I downloaded the full set of World Development Indicators from the World Bank, which consists of over 1700 indicators. Unfortunately, I was often constrained by when and how widely different indicators were collected: often demographic information was widely recorded, but metrics like the Gini Index, which I thought would be very interesting to analyze, were just too sparse to make use of. 

## Data Cleaning

In order to make the REIGN data more reasonable to predict on (and to make the granularity of the data match up with the other datasets), I decided to just pull out the information from January of each year for every country, but added a column to track whether at least one coup or coup attempt happened during that entire year.  

In many ways, the largest challenge of this project was finding an effective way to merge the REIGN and World Bank datasets, which were both organized differently and often used different names for the same country. I created a dictionary that associated the names of World Bank countries with the equivalent country code in the REIGN dataset, and used this to create a function that pulled out interesting indicators from the World Bank data, reshaped them to match the REIGN data, and assigned a unqiue "year code" based on a combination of the REIGN country code and the year that could be used to merge the indictator with the equivalent country-year row from the REIGN data. 

## EDA

Overall, coups appear to have spiked in the 1960's and subsequently declined, albeit with spikes in the mid-1970's (probably linked to operation CONDOR in South America) and the early 1990's (likely linked to the instability caused by the fall and dissolution of the USSR.)

![coups by year](images/coupsyearly.png)


Bolivia had the largest number of both coups and attempted coups, with a number of other Latin American nations in the top 15.


![coups by country](images/coupsbycountry.png)


Personal Dictatorships and Presidential Democracies suffered the greatest number of coups overall. This would seem to indicate that coups tend to occur (overall) in places which power is invested in a single person, whether democratically or not. 


![coups by governments](images/coupsbygovttotal.png)


When scaled by the number of months that each government type existed in the dataset, however, the interim government types (provisional civilian and provisional military) experienced the highest rate of coups, with indirect military rule also experiencing a high rate. This suggests, fairly intuitively, the government types associated with instability or miltary rule experience the most coups.


![coups by governments %](images/coupsbygovtpercent.png)


Likewise, when looking at leader tenures, the largest number of coups happen within the first year of a leader's rule


![coups by leader tenure](images/coupsbyleadertenure.png)


## Modeling 

A major challenge in modeling was the extreme imbalance of the classes: out of 135,637 thousands rows in the REIGN data, only 466 had an attempted coup. A model that always predicted a coup wouldn't happen would automatically feature an accuracy upwards of 99.5%, but would completely miss the point of the investigation. Instead of focusing on accuracy, I was really curious about training a model with strong recall. However, the thing I really cared about was interpretability of the model, so I decided to first focus on a creating an inferential model using a logistic regression. 



In order to deal with the problem of imbalanced classes, I experimented with several techniques while evaluating the performance of a simple logistic regression. I found that oversampling and SMOTE tended to perform fairly well, but oversampling offered slightly better recall, so I used it as my resampling method (however, in my pipeline I did build in the option to try downsampling and SMOTE instead.) I also made sure to stratify my target column when using a test train split to ensure that the model would have a reasonable number of targets to predict on when I tested it. 

I experimented with using Variance Inflation Factors to identify collinear features, but instead decided to use a logistic regression with L1 regularization to pull features to 0. After dropping those features, I used a logistic regression with 5-fold cross validation and an elastic net regularizer to derive a list of relative feature importances and their direction:

## Results: Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy| 79% |
| Recall|  82% |


## Strongest Positive Indicators

| Feature                | Coefficient | Description |
|------------------------|-------------|-------------|
| milex                  |14.0   | Military expenditures |
| trade balance          | 5.1   | Trade balance (exports - imports) |
| lastelection           | 4.67    | Months since the last election|
| prev_conflict          | 2.4    | Dummy variable for a violent civil conflict in the past 6 months
| Provisional - Civilian | 1.9    | Dummy variable for an interim civilian coverment |
| milper                 | 1.6   | Total Military Personnel |
| Warlordism             | 1.5    | Dummy variable for rule by warlords (currently applies to Yemen and Libya) |
| Military               |  1.4    | Rule by a military junta |
| Indirect Military      | 1.4    | Rule by a military junta with a civilian puppet |
| Military-Personal      |  1.3    | Rule by a military junta consolidated around a single figure (ex. Pinochet in Chile) |

## Strongest Negative Indicators

| Feature             | Coefficient |  Description |
|------------------|------------|---|
| irst             | -56.0 | Iron and steel production |
| indirect_recent  | -9.4  | Dummy variable for the 6 months following an indirect election (controlled by elites rather than popular vote)  For example, Xi Jinping's elevation by party elites in China|
| population       | -7.6  | Total population  |
| irregular        | -4.9  | Dummy variable for an an anticipated irregular election  |
| mil_percent      | -4.5  | Military Personnel as a percentage of the total popuation  |
| ref_ant          | -2.2  | A referendum is expected in the next 6 months  | 
| year             | -2.0  | Year  |
| precip           | -0.8  | Precipition relative to histoical average  |
| loss             | -0.7  | Number of months since the incumbent or political party has lost an election (or changed, in the absence of elections) |
| leg_ant          | -0.7  | Dummy variable for a legislative election expected in the next 6 months  |





## Citations

### REIGN Dataset

Bell, Curtis. 2016. The Rulers, Elections, and Irregular Governance Dataset (REIGN). Broomfield, CO: OEF Research. Available at oefresearch.org

The REIGN Dataset was constructed using the following resources:

Goemans, Henk E., Kristian Skrede Gleditch, and Giacomo Chiozza. 2009. "Introducing Archigos: A Dataset of Political Leaders" Journal of Peace Research, 46(2): 269-183. {{http://privatewww.essex.ac.uk/~ksg/archigos.html}}

Ellis, Cali Mortenson, Michael C. Horowitz, and Allan C. Stam. 2015. "Introducing the LEAD Data Set." International Interactions, 41(4): 718-741. {{http://www.tandfonline.com/doi/abs/10.1080/03050629.2015.1016157}}

Marshall, Monty G., Ted Robert Gurr, and Keith Jaggers. 2016. Polity IV Project. Center for Systemic Peace. {{http://www.systemicpeace.org/inscr/p4manualv2015.pdf}}

Barbara Geddes, Joseph Wright, and Erica Frantz. 2014. “Autocratic Regimes and Transitions.” Perspectives on Politics. 12(2).{{http://dictators.la.psu.edu/}}

Powell, Jonathan & Clayton Thyne. 2011. Global Instances of Coups from 1950-Present. Journal of Peace Research 48(2):249-259.{{http://www.jonathanmpowell.com/coup-detat-dataset.html}}

Erik Melander, Therése Pettersson, and Lotta Themnér (2016) Organized violence, 1989–2015. Journal of Peace Research 53(5) {{http://www.pcr.uu.se/research/ucdp/datasets/replication_datasets/}}
