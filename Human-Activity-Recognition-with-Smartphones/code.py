# --------------
import pandas as pd
from collections import Counter

# Load dataset

data = pd.read_csv(path)

print(data.isnull().sum())

print(data.info)


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data['Activity']

# plot the countplot
sns.countplot( x = label)

plt.xticks(rotation=90)
plt.show()


# --------------
# make the copy of dataset
data_copy = data.copy()

# Create an empty column 

data_copy['duration'] = '' 


# Calculate the duration
duration_df = data_copy.groupby([label[(label=='WALKING_UPSTAIRS') | (label=='WALKING_DOWNSTAIRS')],'subject'])['duration'].count()*1.28

duration_df=pd.DataFrame(duration_df)
plot_data=duration_df.sort_values(by='duration').reset_index()
print(plot_data)
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')

# Sort the values of duration





# --------------
#exclude the Activity column and the subject column
feature_cols = data.drop(['Activity', 'subject'], 1).columns
print(feature_cols)

#Calculate the correlation values

correlated_values=data[feature_cols].corr().stack().reset_index()
correlated_values=pd.DataFrame(correlated_values)


#stack the data and convert to a dataframe

correlated_values.rename(columns={"level_0":"Feature_1","level_1":"Feature_2",0:"Correlation_score"},inplace=True)

#create an abs_correlation column

correlated_values['abs_correlation']=correlated_values['Correlation_score'].abs()

s_corr_list=correlated_values.sort_values(by='abs_correlation',ascending=False)
#Picking most correlated features without having self correlated pairs
top_corr_fields=s_corr_list[(s_corr_list['abs_correlation']>0.8)]
top_corr_fields = top_corr_fields[(top_corr_fields['Feature_1'])!=(top_corr_fields['Feature_2'])].reset_index(drop=True)


# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
le.fit(data['Activity'])
data['Activity'] = le.transform(data['Activity'])

# split the dataset into train and test
X = data.drop('Activity',axis=1)
y = data['Activity']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)

# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train,y_train)
y_pred = clf.predict(X_test)
precision,accuracy,f_score,_ = error_metric(y_test,y_pred,average = 'weighted')
print(precision)
print(accuracy)
print(f_score)
model1_score = accuracy_score(y_test,y_pred)
print(model1_score)


# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel



# Feature selection using Linear SVC

lsvc = LinearSVC(penalty='l1',dual=False,C=0.01,random_state=42)
lsvc.fit(X_train,y_train)
model_2 = SelectFromModel(lsvc,prefit=True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)


# model building on reduced set of features
classfier_2=SVC()
clf_2=classfier_2.fit(new_train_features,y_train)
y_pred_new=clf_2.predict(new_test_features)
model2_score=accuracy_score(y_pred_new,y_test)
precision,recall,f_score,_=error_metric(y_test,y_pred_new,average = 'weighted')
print(precision)
print(recall)
print(f_score)
print(model2_score)



# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = {'kernel': ['linear', 'rbf'],'C': [100, 20, 1, 0.1]}

# Usage of grid search to select the best hyperparmeters

svc = SVC()
selector = GridSearchCV(svc,parameters,scoring='accuracy')
selector.fit(new_train_features,y_train)

print(selector.best_params_)
print(selector.cv_results_)
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
params = selector.cv_results_['params']
print(means,stds,params)


# Model building after Hyperparameter tuning

classifier_3 = SVC(C= 20, kernel= 'rbf')
clf_3 = classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)
model3_score = accuracy_score(y_test,y_pred_final)
precision,recall,f_score,_= error_metric(y_test,y_pred_final,average = 'weighted')
print(precision)
print(recall)
print(f_score)
print(model3_score)



