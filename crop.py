import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.simplefilter(action = "ignore")



dataset = pd.read_csv("cropdata.csv")
print(dataset.head())

print(dataset.shape)



print(dataset.size)



df = dataset.copy()

print(df.head())

print(df.describe(include = "all"))

print(df.columns)
print(df['label'].unique)
print(df['label'].value_counts())
print(df.dtypes)

sns.distplot(df['N'])
plt.show()

sns.distplot(df['P'])
plt.show()

sns.distplot(df['K'])
plt.show()

sns.distplot(df['temperature'])
plt.show()

sns.distplot(df['humidity'])
plt.show()

sns.distplot(df['ph'])
plt.show()

sns.distplot(df['rainfall'])
##plt.show()

# sns.heatmap(df.corr(),annot=True)
##plt.show()


x = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df[['label']]

print(x)

print(y)

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size = 0.2,random_state =2)

print(Xtrain)

print(Ytest)
print(Xtest)
print(Ytrain)

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = df[['N','P','K','temperature','humidity','ph','rainfall']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif["Feature"]=variables.columns

print(vif)

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)



Predicted_values = DecisionTree.predict(Xtest)

x1 = metrics.accuracy_score(Ytest,Predicted_values)
acc.append(x1)
model.append('Decision Tree')

print("Decision Tree accuracy is : " , x1*100)
print(classification_report(Ytest,Predicted_values))

from sklearn.model_selection import cross_val_score

score = cross_val_score(DecisionTree,x,y,cv = 5)
print(score)

import pickle
DT_Model_pkl = open('DT_pkl_file','wb')
pickle.dump(DecisionTree , DT_Model_pkl)
DT_Model_pkl.close()




from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x1 = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x1)
model.append('Naive Bayes')

print("Naive Bayes's Accuracy is: ", x1)

print(classification_report(Ytest,predicted_values))

score = cross_val_score(NaiveBayes,x,y,cv=5)
score

NB_Model_pkl = open('NB_pkl_file', 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
NB_Model_pkl.close()


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
X_test_norm = norm.transform(Xtest)

SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm,Ytrain)

predicted_values = SVM.predict(X_test_norm)
x1 = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x1)
model.append('SVM')

print("SVM's Accuracy is: ", x1)

print(classification_report(Ytest,predicted_values))

SVM_Model_pkl = open('SVM_pkl_file', 'wb')
pickle.dump(SVM, SVM_Model_pkl)
SVM_Model_pkl.close()

#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

#RF = RandomForestClassifier(n_estimators = 20 , random_state = 0)
LR = LogisticRegression() 
LR.fit(Xtrain,Ytrain)

Predicted_values = LR.predict(Xtest)
x1 = metrics.accuracy_score(Ytest,Predicted_values)
acc.append(x1)
model.append('LR')

print("Logistic Regression accuracy is : " , x1)
print(classification_report(Ytest,Predicted_values))

score = cross_val_score(LR,x,y,cv=5)
score

LR_Model_pkl = open('LR_pkl_file', 'wb')
pickle.dump(LR, LR_Model_pkl)
LR_Model_pkl.close()

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')
plt.show()

data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = SVM.predict(data)
print(prediction)

df = pd.read_csv("crop_production.csv")
data = df.dropna()
test = df[~df["Production"].notna()].drop("Production",axis=1)
sum_maxp = data["Production"].sum()
data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)



rice_df = data[data["Crop"]==prediction[0]]
# print(rice_df.shape)
# print(rice_df[:3])

# sns.barplot("Season","Production",data=rice_df)
# plt.show()
