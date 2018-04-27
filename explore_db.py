'''
explore_db

Created on Apr 08 2018 11:16 
#@author: Kevin Le

ToDo
- Import data
- Clean data
- Explore data

Data to Clean
- Missing data
- Invalid column names

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def main():
    root = 'EmployeeTurnOverPrediction/'
    data = root + 'data/HR_comma_sep.csv'
    uncleaneddf = pd.read_csv(data)
    df = cleanScrubData(uncleaneddf)

    exploreDistribution(df)

def cleanScrubData(df):
    # Rename for readability
    df = df.rename (columns={'satisfaction_level': 'satisfaction',
                             'last_evaluation': 'evaluation',
                             'number_project': 'projectCount',
                             'average_montly_hours': 'averageMonthlyHours',
                             'time_spend_company': 'yearsAtCompany',
                             'Work_accident': 'workAccident',
                             'promotion_last_5years': 'promotion',
                             'sales': 'department',
                             'left': 'turnover'
                             })
    return df

def isMissingData(df):
    missingData = df.columns[df.isnull().any()]
    if len(missingData) > 0:
        print 'Missing data in columns: {}'.format(missingData)


def conductTtest(df, featureColumn):
    mean0 = df[featureColumn][df['turnover'] == 0].mean ()
    mean1 = df[featureColumn][df['turnover'] == 1].mean ()
    tTest = stats.ttest_1samp (a=df[featureColumn][df['turnover'] == 1],
                               popmean=mean0)

    nDegreesFreedom = len(df[df['turnover'] == 1])
    leftQuantile = stats.t.ppf(0.025, nDegreesFreedom)
    rightQuantile = stats.t.ppf(0.975, nDegreesFreedom)

    if tTest.statistic < leftQuantile or tTest.statistic > rightQuantile:
        rejectNull = True
    else:
        rejectNull = False

    metrics = {
        'mean_no_turnover': mean0,
        'mean_turnover': mean1,
        'tTest': tTest,
        'leftQuantile':leftQuantile,
        'rightQuantile': rightQuantile,
        'rejectNull': rejectNull
    }

    for k, v in metrics.iteritems ():
        print(k,': ', v)

def exploreDistribution(df):
    # Distribution Target Variable
    x, bins, rang = plt.hist(df['turnover'], bins=2)
    plt.title('Distribution of Employees that left')
    plt.show()

    # Show correlation
    corr = df.corr()
    corr = (corr)
    sns.heatmap(corr,
                 xticklabels=corr.columns.values,
                 yticklabels=corr.columns.values,
                 annot=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()

    # Satisfaction, Evaluation, AverageMonthlyHours
    features = ['satisfaction', 'evaluation', 'averageMonthlyHours']
    color = ['g', 'r', 'b']
    figs, axarr = plt.subplots (1, len (features), figsize=(15, 4))
    for i, feature in enumerate (features):
        sns.distplot (df[feature], color=color[i], ax=axarr[i], kde=False).set_title (
            'Employee {} distribution'.format (feature))
    plt.show ()

    # Kernel Density Plots
    fig, axarr = plt.subplots(1, len(features), figsize=(15,4))
    for i,feature in enumerate(features):
        sns.kdeplot(df.loc[df['turnover'] == 0, feature], color='b', shade=True, label='no turnover', ax=axarr[i])
        sns.kdeplot(df.loc[df['turnover'] == 1, feature], color='r', shade=True, label='turnover', ax=axarr[i])
        axarr[i].set_xlabel('Employee {}'.format(feature))
        axarr[i].set_title('Employee {} Distribution - turnover vs no turnover'.format(feature), fontsize=10)
    plt.show()

    # Department
    fig, axarr = plt.subplots(2,1, figsize=(15,10))
    colorTypes = ['#78C850', '#F08030', '#6890F0', '#A8B820', '#A8A878', '#A040A0', '#F8D030',
                   '#E0C068', '#EE99AC', '#C03028', '#F85888', '#B8A038', '#705898', '#98D8D8', '#7038F8']
    sns.countplot (x='department', data=df, palette=colorTypes, ax=axarr[0]).set_title ('Employee Department Distribution')
    plt.xticks (rotation=-45)
    sns.countplot(y="department", hue='turnover', data=df, ax=axarr[1]).set_title('Employee Department Turnover Distribution')
    plt.show()

    # yearsAtCompany
    plt.figure()
    sns.countplot(x='yearsAtCompany', hue='turnover', data=df).set_title('Employee YearsAtCompany distribution')
    plt.show()

    # yearsAtCompany vs Satisfaction
    plt.figure()
    sns.boxplot(x='yearsAtCompany', y='satisfaction', hue='turnover', data=df)
    plt.title('YearsAtCompany vs Satisfaction')
    plt.show()

    # Salary
    plt.figure()
    sns.countplot(y='salary', hue='turnover', data=df).set_title('Employee salary distribution')
    plt.show()

    # ProjectCount
    plt.figure()
    sns.countplot(x='projectCount',  hue='turnover', data=df).set_title('Employee project count distribution')
    plt.show()

    plt.figure()
    sns.boxplot(x='projectCount', y='averageMonthlyHours', hue='turnover', data=df)
    plt.title('ProjectCount vs AverageMonthlyHours')
    plt.show()

    plt.figure()
    sns.boxplot(x='projectCount', y='evaluation', hue='turnover', data=df)
    plt.title('ProjectCount vs Evaluation')
    plt.show()

    # Satisfaction vs Evaluation
    plt.figure()
    sns.lmplot(x='satisfaction', y='evaluation', hue='turnover', data=df, legend_out=True, fit_reg=False, legend=True)
    plt.title('Satisfaction vs Evaluation')
    plt.show()





if __name__ == '__main__':
    main()
