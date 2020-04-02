import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

############# load data set ##########################

cust_data = pd.read_csv("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\DSA_HACKTHON_1\\Grand_Hackthon\\cust_Bank.csv")
cust_data.dtypes
cust_data.columns
#Index(['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y'],
cust_data.info()
cust_data.shape
print ('The data has {0} rows and {1} columns'.format(cust_data.shape[0],cust_data.shape[1]))
cust_data.head()

bank_validation_data = pd.read_csv("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\DSA_HACKTHON_1\\Grand_Hackthon\\Bank_Validation.csv")
bank_validation_data.shape
bank_validation_data.info()


################# Data preparation #########################E#

#check missing values

cust_data.columns[cust_data.isnull().any()]
print(cust_data.isnull().any())
"""NO missing values"""

cust_data['education'].unique()
cust_data['education'] = np.where(cust_data['education'] == 'basic.4y', 'Basics', cust_data['education'])
cust_data['education'] = np.where(cust_data['education'] == 'basic.6y', 'Basics', cust_data['education'])
cust_data['education'] = np.where(cust_data['education'] == 'basic.9y', 'Basics', cust_data['education'])
cust_data['education'].unique()

#divide numeric and non numeric data
numeric_data = cust_data.select_dtypes(include=[np.number])
cat_data = cust_data.select_dtypes(exclude=[np.number])

numeric_data.columns
cat_data.columns

################ data analysis #####################

#Handling Imbalanced Data with SMOTE and Near Miss Algorithm
cust_data['y'].value_counts()

sns.countplot(x='y', data=cust_data, palette='hls')
plt.show()

count_no_sub = len(cust_data[cust_data['y'] == 'no'])
count_of_sub = len(cust_data[cust_data['y'] == 'yes'])
pct_no_sub = count_no_sub/(count_no_sub+count_of_sub)
pct_sub = count_of_sub/(count_no_sub+count_of_sub)
print("percentage of no subscription is",pct_no_sub*100)
print("percentage of subscription is",pct_sub*100)


cust_data.head(2)
cust_data.groupby('y').mean()
#observation: The average age of customers who bought the term deposit is higher than that of the customers who didn’t.

cust_data.groupby('job').mean()
#observation: retired employees bought more

cust_data.groupby('marital').mean()

cust_data.groupby('education').mean()

########################## visualization tells which predictors will be good to modeel ###################

"""The frequency of purchase of the deposit depends a great deal on the job title. 
Thus, the job title can be a good predictor of the outcome variable."""
pd.crosstab(cust_data.job, cust_data.y).plot(kind='bar')
plt.title("Purchase Frequency for Job Title")
plt.xlabel("Job")
plt.ylabel("Frequency of Purchase")

table=pd.crosstab(cust_data.marital,cust_data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')
"""The marital status does not seem a strong predictor for the outcome variable"""

table=pd.crosstab(cust_data.education,cust_data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')
"""Education seems a good predictor of the outcome variable"""

table=pd.crosstab(cust_data.default,cust_data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
plt.title('Stacked Bar Chart of default vs Purchase')
plt.xlabel('default')
plt.ylabel('Proportion of Customers')
"""The default does not seem a strong predictor for the outcome variable"""

plt.figure(figsize=(20,8))
plt.title("Histogram of Age")
plt.hist(cust_data["age"])
"""Most of the customers of the bank in this dataset are in the age range of 30–40."""

plt.figure(figsize=(20,8))
sns.barplot(data = cat_data)

############# Create dummy variables ##############################

cat_vars = cat_data.columns.values.tolist()
cat_vars.pop()
cat_vars
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(cust_data[var], prefix=var)
    #print(cat_list)
    data1 = cust_data.join(cat_list)
    cust_data = data1

data_vars = cust_data.columns.values.tolist()
data_vars
to_keep = [i for i in data_vars if i not in cat_vars]
to_keep

cust_data_final = cust_data[to_keep]
cust_data_final.columns.values
cust_data_final.head(2)

cust_data_final['y1'] = cust_data_final['y'].apply(lambda x: 1 if x == 'yes' else 0)
cust_data_final.columns

cust_data_final_1 = cust_data_final.drop(columns=['y'])
cust_data_final_1.columns

cust_data_final_2 = cust_data_final_1.rename(columns={'y1': 'y'})
cust_data_final_2.columns
cust_data_final_2.head(10)


############## divide data to train and test ##############

X = cust_data_final_2.loc[:, cust_data_final_2.columns != 'y']
y = cust_data_final_2.loc[:, cust_data_final_2.columns == 'y']
X.columns
y.columns
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_train.columns
y_train.columns
y_train.head(100)
X_train

y_test

"""
# Shuffle your dataset
shuffle_df = cust_data.sample(frac=1)
# Define a size for your train set
train_size = int(0.7 * len(cust_data))
# Split your dataset
train_data = shuffle_df[:train_size]
test_data = shuffle_df[train_size:]
train_data.head(10)
"""

############## feature selection ##############

# data_final_vars=cust_data_final_2.columns.values.tolist()
# y=['y']
# X=[i for i in data_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

final_cols = ['job_admin.','job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','marital_divorced','marital_married','marital_single','education_Basics','education_high.school','education_illiterate','education_university.degree','housing_no','housing_yes']

############ Train Models ############################

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

########### logistic regression #################

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

################ metrics to evaluate ##############
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

####### Compute precision, recall, F-measure and support ---- classification_report#######
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

################### ROC Curve #######################
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


################### SVM Model #########################

from sklearn import svm
svm_clf =  svm.SVC(C=5, gamma=.05, kernel='rbf')
svm_clf.fit(X_train, y_train.values.ravel())

svm_pred = svm_clf.predict(X_test)
print(confusion_matrix(y_test, svm_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, svm_pred)

################# Decision Trees #############################

from sklearn.tree import DecisionTreeClassifier

# Make a decision tree and train
tree = DecisionTreeClassifier()
tree
tree.fit(X_train, y_train.values.ravel())
tree_pred = tree.predict(X_test)
print(f'Model Accuracy: {tree.score(y_test, tree_pred)}')







