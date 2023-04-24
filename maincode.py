# Import necessary libraries
import pandas as pd
import numpy as np
import math
import os
import glob
import zipfile
import concurrent.futures
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------------
# Set the path to the Excel file containing survey results
path = r'D:\Data Science\SurveyResults.xlsx'

# Creating date time based on individual columns of time, date, start time, end time etc.
excel = pd.read_excel(path, usecols=['ID', 'Start time', 'End time', 'date', 'Stress level'], dtype={'ID': str})
excel['Stress level'].replace('na', np.nan, inplace=True)
excel.dropna(inplace=True)

excel['Start datetime'] = pd.to_datetime(excel['date'].map(str) + ' ' + excel['Start time'].map(str))
excel['End datetime'] = pd.to_datetime(excel['date'].map(str) + ' ' + excel['End time'].map(str))
excel.drop(['Start time', 'End time', 'date'], axis=1, inplace=True)

# Convert SurveyResults.xlsx to GMT-00:00
daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

excel1 = excel[excel['End datetime'] <= daylight].copy()
excel1['Start datetime'] = excel1['Start datetime'].apply(lambda x: x + timedelta(hours=5))
excel1['End datetime'] = excel1['End datetime'].apply(lambda x: x + timedelta(hours=5))

excel2 = excel.loc[excel['End datetime'] > daylight].copy()
excel2['Start datetime'] = excel2['Start datetime'].apply(lambda x: x + timedelta(hours=6))
excel2['End datetime'] = excel2['End datetime'].apply(lambda x: x + timedelta(hours=6))

excel = pd.concat([excel1, excel2], ignore_index=True)
excel['datetime'] = excel['End datetime'] - excel['Start datetime']


# Creating minutes columns for labeling
def minutes(x):
    return x.seconds / 60


excel['time'] = excel['datetime'].apply(minutes)

excel.reset_index(drop=True, inplace=True)

# --------------------------------------------------------------------------------------------
# To tell the time of execution
# %%time
# glob package is used to get all the folders inside the path specified
folders = glob.glob(os.path.join(r"D:\Data Science\Data\*"))
# --------------------------------------------------------------------------------------------

a = 0
# creating new dataframe to hold all the data
final = pd.DataFrame()

# looping through all the folders to extract the files
# for j in range(len(folders)):
for j in range(1):
    # glob package is again used for each of the looping folders to extract .zip files
    files = glob.glob(os.path.join(folders[j] + '\*.zip'))

    # looping through each zip files to unzip the files
    for i in range(len(files)):

        zf = zipfile.ZipFile(files[i])


        def unzip(file):
            zf.extract(file)


        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(unzip, zf.infolist())

        # --------------------------------------------------------------------------------------------
        # unzipping and reaching the files
        ACC = pd.read_csv('ACC.csv')
        BVP = pd.read_csv('BVP.csv')
        EDA = pd.read_csv('EDA.csv')
        HR = pd.read_csv('HR.csv')
        file_stat = os.stat('IBI.csv')
        # if file_stat.st_size > 0:
        #    IBI = pd.read_csv('IBI.csv')
        TEMP = pd.read_csv('TEMP.csv')

        # --------------------------------------------------------------------------------------------
        # Creating Date Time for each dataset based on timestamp

        ACC['id'] = folders[j][-2:]


        def process_df(df):
            start_timestamp = df.iloc[0, 0]
            sample_rate = df.iloc[1, 0]
            new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
            new_df['datetime'] = [(start_timestamp + i / sample_rate) for i in range(len(new_df))]
            return new_df


        ACC = process_df(ACC)
        BVP = process_df(BVP)
        EDA = process_df(EDA)
        HR = process_df(HR)
        TEMP = process_df(TEMP)
        # --------------------------------------------------------------------------------------------
        # renaming the columns to our needs
        ACC.rename(
            {ACC.columns[0]: 'accelerometer_X', ACC.columns[1]: 'accelerometer_Y', ACC.columns[2]: 'accelerometer_Z'},
            axis=1, inplace=True)
        BVP.rename({BVP.columns[0]: 'BVP'}, axis=1, inplace=True)
        EDA.rename({EDA.columns[0]: 'EDA'}, axis=1, inplace=True)
        HR.rename({HR.columns[0]: 'heart_rate'}, axis=1, inplace=True)
        # IBI.rename({IBI.columns[0]:'IBI_0',IBI.columns[1]:'IBI_1'},axis = 1,inplace = True)
        TEMP.rename({TEMP.columns[0]: 'temp'}, axis=1, inplace=True)

        # --------------------------------------------------------------------------------------------
        # Merging the Data
        new = ACC.merge(EDA, on='datetime', how='outer')
        new = new.merge(TEMP, on='datetime', how='outer')
        new = new.merge(HR, on='datetime', how='outer')

        # Fill null value by forward and backward fill

        new.fillna(method='ffill', inplace=True)
        new.fillna(method='bfill', inplace=True)
        new.reset_index(inplace=True, drop=True)

        # --------------------------------------------------------------------------------------------
        # Appending and concating the final merged files
        if j == 0:
            final = new.append(final)
        else:
            final = pd.concat([final, new], ignore_index=True)
        a += 1
        print('The file number:- ', a, ' of folder:- ', j, ' is done')

    a = 0
    print('The folder number:-', j, ' is done')

    # Labelling based on datetime
    # Convert the 'id' column in 'final' and 'ID' column in 'excel' to string data type
    final['id'] = final['id'].astype('str')
    excel['ID'] = excel['ID'].astype('str')


    # Define a function to label rows in 'final' as 1 or 0 based on the 'datetime' column
    def label(x):
        if x >= excel['time'].max():  # if the datetime value is greater than or equal to the maximum time in 'excel'
            return 1  # assign the label 1
        else:
            return 0  # otherwise, assign the label 0


    # Apply the 'label' function to the 'datetime' column in 'final' and assign the result to a new 'Label' column
    final['Label'] = final['datetime'].apply(label)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv('Preprocessed.csv')

    # drop unecessary columns
    data.drop({'Unnamed: 0', 'datetime', 'id'}, axis=1, inplace=True)

    data.head()

    data.isnull().sum()

    # check duplicates
    data.duplicated().sum()

    # drop duplicates
    data.drop_duplicates(inplace=True)

    data.dtypes

    data.describe()

    data.shape

    # imbalanced which can be balanced by upsampling using SMOTE or downsampling (not preferred method)
    data['Label'].value_counts()

    sns.countplot(data['Label'])

    from sklearn.utils import resample

    # seperating zeros and ones because we have to downsample zero and keep the ones same
    downsize_0 = data[data['Label'] == 0]
    keep_ones = data[data['Label'] == 1]

    zeros = resample(downsize_0, n_samples=len(keep_ones), replace=True)

    # Finally merging ones and resampled zeros
    new_data = pd.concat([keep_ones, zeros]).reset_index(drop=True)

    new_data.head()

    new_data['Label'].value_counts()

    sns.countplot(new_data['Label'])

    # Correlation tab shows us which features are highly positivey correlated and highly negatively correlated
    new_data.corr()

    sns.heatmap(new_data.corr(), annot=True)

    # Histogram suggests which feature is normally distributed
    new_data.hist(figsize=(10, 10))

    from sklearn.model_selection import train_test_split

    X = new_data.drop({'Label'}, axis=1)
    y = new_data['Label']

    # Splitting the data into training and testing using sklearn package
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100)

    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_classif

    # Feature selection using sklearn by select the best number of k features and mutual information of feature selection
    feature = SelectKBest(score_func=mutual_info_classif, k='all')

    feature.fit(X_train, y_train)

    feature_value = []
    for i in range(len(feature.scores_)):
        print('Feature importance of %d: %f' % (i, feature.scores_[i]))
        feature_value.append(feature.scores_[i])

        # Set the x-axis labels
        labels = list(new_data.columns)[0:6]

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(np.arange(len(feature_value)), feature_value)
        ax.set_xticks(np.arange(len(feature_value)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Percentage')
        ax.set_title('Features Importance')

        plt.show()

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report

        Logistic_regression = LogisticRegression(random_state=0).fit(X_train, y_train)

        y_pred_1 = Logistic_regression.predict(X_test)

        print(classification_report(y_test, y_pred_1))

        report_1 = classification_report(y_test, y_pred_1, output_dict=True)

        sns.heatmap(pd.DataFrame(report_1).iloc[:-1, :].T, annot=True)

        from sklearn.neighbors import KNeighborsClassifier

        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)

        # Predicting
        y_pred_2 = neigh.predict(X_test)

        print(classification_report(y_test, y_pred_2))

        report_2 = classification_report(y_test, y_pred_2, output_dict=True)

        sns.heatmap(pd.DataFrame(report_2).iloc[:-1, :].T, annot=True)

        from sklearn.naive_bayes import GaussianNB

        Naive_bayes = GaussianNB()
        Naive_bayes.fit(X_train, y_train)

        y_pred_3 = Naive_bayes.predict(X_test)

        print(classification_report(y_test, y_pred_3))

        report_3 = classification_report(y_test, y_pred_3, output_dict=True)

        sns.heatmap(pd.DataFrame(report_3).iloc[:-1, :].T, annot=True)

        

