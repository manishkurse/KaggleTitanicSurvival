
# coding: utf-8

# # Predicting survival on the Titanic

# Goal of this Kaggle competition is to predict survival of a passengers on the titanic by applying supervised learning on a training set provided to us.

# <b>VARIABLE DESCRIPTIONS:</b>
# 
# 1. survival - Survival (0 = No; 1 = Yes)
# 2. pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 3. name - Name
# 4. sex - Sex
# 5. age - Age
# 6. sibsp - Number of Siblings/Spouses Aboard
# 7. parch - Number of Parents/Children Aboard
# 8. ticket - Ticket Number
# 9. fare - Passenger Fare
# 10. cabin - Cabin
# 11. embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[60]:

import numpy as np
import pandas as pd
import pylab as P
import decimal
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
get_ipython().magic('matplotlib inline')


# In[61]:

# Load data into a Pandas data frame
filename = '/Users/manishkurse/Documents/DataScience/DataScienceProjects/Kaggle/TitanicShipwreck/train.csv'

# # Create a data frame reading from CSV file
df = pd.read_csv(filename, header=0)

# View the data
df.head(4)


# ## Clean data and Feature Engineering

# Here are some of the changes that were made:
# <li>Filing missing ages with median values for that gender and Class</li>
# <li>Mapping categorical variables to numerical categories</li>
# <li>Using median values for Fare where they are missing</li>
# <li>Continuous variable Fare being divided into discrete categories</li>
# <li>Creating new binary categorical variable indicating whether the ticket is a number only or there are alphabets in it</li>
# 

# In[62]:

# Clean data : remove missing data and create new variables
def clean_data(df):
    # Mapping gender data to a numerical 0,1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Fill missing age values with median in the corresponding Pclass,gender and create a new variable
    df['AgeFill'] = df['Age']
    
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[(df['Gender']==i)&(df['Pclass']==j+1)]['Age'].dropna().median()
            df['AgeFill'].loc[(df['Gender']==i)&(df['Pclass']==j+1)&(df['Age'].isnull())] =median_ages[i,j]

    # Create a new variable family size which is sum of siblings, parents and children
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Fill missing values for Embarked with the mode
    df['Port']  = df['Embarked']
    df.Port = df.Port.fillna(df.Port.mode()[0])

    # Mapping port to numeric data 0,1,2
    df['Port'] = df['Port'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

    # Fill missing values of fare with median value
    df.Fare[df.Fare.isnull()] = df.Fare.median()
    
    df['Ticket_AlphaNum'] = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if df.Ticket[i][0].isalpha():
            df.Ticket_AlphaNum[i] = 1
        else:
            df.Ticket_AlphaNum[i] = 0

            
    # Categorize the Fares to form groups.
    num_fare_groups = 6
    df['FareCat'] = df['Fare']*-1
    # Use percentile to categorize the fares
    perc = np.linspace(0,100,num_fare_groups) 

    
    for i in range(0,len(perc)-1):
        df['FareCat'][(df['Fare']>=np.percentile(df['Fare'],perc[i]))        & (df['Fare']<=np.percentile(df['Fare'],perc[i+1]))]=i
            
    # Remove features you don't plan to use
    df =df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age'], axis=1)
    return df


# Convert dataframe to array
def get_array_from_dataframe(df,featureNames):
    df = df[featureNames]
    data = df.values
    data = data.astype(np.float)
    return data

# Clean the data. Filling nans
print(df.shape, ': Before cleaning')
df = clean_data(df) 
df.head(3)
print(df.shape, ': After cleaning')


# ## Exploratory Data Analysis

# In[63]:

# Understand the data
# See how many nans exist
print('Out of ',df.shape[0] * df.shape[1],' values, ',      (df.shape[0] * df.shape[1]) - df.count().sum(),' nans exist')
print(df.AgeFill.max(), 'is max age and', df.AgeFill.min(), 'is min age')


# In[64]:

def prepare_plot_area(ax):
    # Remove plot frame lines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    
    # X and y ticks on bottom and left
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')    
def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),                ha='center', va='bottom')
    
colrcode = [(31, 119, 180), (255, 127, 14),             (44, 160, 44), (214, 39, 40),              (148, 103, 189),  (140, 86, 75),              (227, 119, 194), (127, 127, 127),              (188, 189, 34), (23, 190, 207)]

for i in range(len(colrcode)):  
    r, g, b = colrcode[i]  
    colrcode[i] = (r / 255., g / 255., b / 255.)
    
def plot_survival(df,label_name,label_values,ax,per_flag,label_values_names_list):
    ind = [0,1]
    alpha_plots= .5
    label_values = np.array(label_values) # Labels which you want to plot
    p = np.zeros((len(label_values),2)) # 2 columns for survived no and yes.
    for x in range(len(label_values)):
        count_label = df[df[label_name] == label_values[x]]['Survived'].count()
        if per_flag == True:
            if count_label != 0:
                p[x,1] = df[df[label_name] == label_values[x]]['Survived'].sum()/count_label*100
                p[x,0] = 100-p[x,1]
            else:
                p[x,0] = 0
                p[x,1] = 0               
        else:
            p[x,1] = df[df[label_name] == label_values[x]]['Survived'].sum()
            p[x,0] = count_label-p[x,1]
    
    # Plot the 2 groups: No survived and survived.
    rect_width = .2
    if per_flag == False:
        rects_surv_no = plt.bar(label_values-rect_width,p[:,0],width=rect_width,                                alpha = alpha_plots, color=colrcode[0],edgecolor = colrcode[0],label = 'Non-surv')
        autolabel(rects_surv_no,ax)
        plt.ylabel('Survival')
        plt.title(label_name)
    else:
        plt.ylabel('Survival %')
        plt.title([label_name,' %Survival'])

    rects_surv_yes = plt.bar(label_values,p[:,1],width=.2,alpha = alpha_plots, color=colrcode[1],edgecolor = colrcode[1],label = 'Surv')
    
    plt.xticks(label_values,label_values_names_list)
    plt.xlim(np.min(label_values)-.5,np.max(label_values)+.5)
    plt.ylim(0,np.max(p)+10)
    autolabel(rects_surv_yes,ax)
    plt.legend(loc = 'best')

    
def plot_raw_data_ver1(df):  
        
    # Figure object parameters
    alpha_plots= 1
    
    label_name_list = ['Pclass','Port','SibSp','Parch','FamilySize','Gender']
    label_values_list = [[1,2,3],[0,1,2],list(range(df.SibSp.max()+1)),list(range(df.Parch.max()+1)),list(range(df.FamilySize.max()+1)),[0,1]]
    label_values_names_list = [['Class 1','Class 2','Class 3'],['Cherbourg','Queenstown','Southampton'],                               list(range(9)),list(range(7)),list(range(11)),['Female','Male']]


    fig = plt.figure(figsize=(16,13))
    for x in range(len(label_name_list)):
        ax = plt.subplot(3,2,x+1)  
        prepare_plot_area(ax)
        plot_survival(df,label_name_list[x],label_values_list[x],ax,False,label_values_names_list[x])

    label_name_list_subset = ['Pclass','Port','Gender']
    label_values_list_subset = [[1,2,3],[0,1,2],[0,1]]
    label_values_names_list_subset = [['Class 1','Class 2','Class 3'],['Cherbourg','Queenstown','Southampton'],                               ['Female','Male']]

    
    fig = plt.figure(figsize=(16,9))
    for x in range(len(label_name_list_subset)):
        ax = plt.subplot(2,2,x+1)  
        prepare_plot_area(ax)
        plot_survival(df,label_name_list_subset[x],label_values_list_subset[x],ax,True,label_values_names_list_subset[x])

    
plot_raw_data_ver1(df)


# ### Insight on data:

# Based on the above plots we can see that:
# <li>A much larger % of women survived compared to males. This makes sense since it is known that women and children were asked to board the lifeboats first.</li>
# <li>A much larger % of 1st Class passengers survived compared to 3rd class. </li>
# <li>Looks like passengers embarking in Cherbourg had a larger percentage of survival.</li>
# 

# ## Performing Machine Learning on the Cleaned Dataset

# Random Forest Classification is performed on a select set of features. scikit-learn is used for this. A 10-fold Cross Validation is performed to select parameters of the Random Forest Classifier.

# In[65]:

##Training
def normalizeFeatures(dataArr):
    return((dataArr - np.mean(dataArr))/(np.std(dataArr)))

featureNames = ['Survived','Fare','AgeFill','Gender','FareCat','Pclass','Port','FamilySize','Ticket_AlphaNum','Parch','SibSp']
train_data = get_array_from_dataframe(df,featureNames)
# cv_data = get_array_from_dataframe(cv_df,featureNames)
features2Normalize = [1,2]
for count in range(0,len(features2Normalize)):
    train_data[:,features2Normalize[count]] = normalizeFeatures(train_data[:,features2Normalize[count]])

# Number of samples in the training data
num_samples = train_data.shape[0]

# Create the random forest object which will include all the parameters
# for the fit
num_folds = 10

classifier  = RandomForestClassifier(n_estimators = 100,criterion = "entropy", max_depth=10, max_features = 7,min_samples_leaf = 6, max_leaf_nodes = 20, oob_score=True)

score_cv = np.zeros(num_folds)
score_train = np.zeros(num_folds)
num_cv_sam = np.zeros(num_folds)
all_out = np.ones((num_cv_sam.shape[0],num_samples))*999

# Get k-fold cross-validation indices 
kf = cross_validation.KFold(train_data.shape[0], n_folds=num_folds,shuffle = True)

count = 0
for train_index, cv_index in kf:
    # Fit the training data to the Survived labels and create the decision trees
    num_cv_sam[count] = len(cv_index)
    classifier  = classifier.fit(train_data[train_index,1::],train_data[train_index,0])
    score_cv[count] = classifier.score(train_data[cv_index,1::],train_data[cv_index,0])
    score_train[count] = classifier.score(train_data[train_index,1::],train_data[train_index,0])
    outp = classifier.predict(train_data[:,1::])
    all_out[count,0::] = np.array(outp) 
    count = count+1

print('The mean score on the training dataset is : ',np.around(np.mean(score_train),2))
print('The mean score on the cross-validation dataset is : ',np.around(np.mean(score_cv),2))
plt.figure()
plt.bar(list(range(len(classifier.feature_importances_))),classifier.feature_importances_,color = colrcode[0])
p = plt.xticks(list(range(len(classifier.feature_importances_))),['Fare','AgeFill','Gender','FareCat','Pclass','Port','FamilySize','Ticket_AlphaNum','Parch','SibSp'],rotation=60)


# Based on the feature importances it is clear that gender was the biggest factor followed by Fare, age and Class. The training and 10-fold cross validation errors being almost equivalent shows that the model is not overfitting.

# In[66]:

# Get Test data from the csv file
filename = '/Users/manishkurse/Documents/DataScience/DataScienceProjects/Kaggle/TitanicShipwreck/test.csv'
# Create a data frame reading from CSV file
df_test = pd.read_csv(filename, header=0)
clean_data(df_test)

featureNames = featureNames[1::]
test_data = get_array_from_dataframe(df_test,featureNames)
features2Normalize = [0,1] # Since survival is not there
for count in range(0,len(features2Normalize)):
    test_data[:,features2Normalize[count]] = normalizeFeatures(test_data[:,features2Normalize[count]])

passengerid = get_array_from_dataframe(df_test,'PassengerId')

# Take the same decision trees and run it on the test data
classifier  = classifier.fit(train_data[:,1::],train_data[:,0])
output = classifier.predict(test_data)
num_test_data = np.size(test_data[:,0])
pred = np.zeros((num_test_data,2),dtype = np.int)
pred[:,0] = passengerid
pred[:,1] = output

headers = ['PassengerId','Survived']
f= open('/Users/manishkurse/Documents/DataScience/DataScienceProjects/Kaggle/TitanicShipwreck/random_forest_py.csv','w')
f_csv = csv.writer(f)
f_csv.writerow(headers)
f_csv.writerows(pred)
f.close()
print('CSV file created!')

