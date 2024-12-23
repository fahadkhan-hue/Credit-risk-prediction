#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.chdir('C:\\Users\\ASUS\\desktop\\DL')


# In[75]:


##importing alll the required libraries

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import pandas as pd
import numpy as np


df = pd.read_csv("Assignment1Data.csv")
df.head(10)


# In[5]:


#checking the dimensoins of the dataset/dataframe
df.shape


# In[7]:


#getting sum of null values columnwise
df.isna().sum()


# In[9]:


#dropping columns with null values
df = df.dropna()
df.shape



# In[11]:


#Converting categorical data columns into labelled columns
columns_to_label = [
                    "checkingstatus1",
                    "history",
                    "purpose",
                    "savings",
                    "employ",
                    "status",
                    "others",
                    "property",	
                    "otherplans",
                    "housing",	
                    "job",
                    "foreign"
    
]
label_encoders = {}
for col in columns_to_label:
    le_en = LabelEncoder()
    df[col] = le_en.fit_transform(df[col])
    label_encoders[col] = le_en
#saving the transformed data in a csv file named transformed_asgnmnt1_data.csv.
df.to_csv("transformed_asgnmnt1_data.csv", index= False)
print("data has been saved to : transformed_asgnmnt1_data.csv ")


# In[13]:


#Reading transformed dataset
df1 = pd.read_csv('transformed_asgnmnt1_data.csv')


# In[19]:


df1.head()


# In[15]:


#crosschecking the transformed data set
df1.isna().sum()


# In[17]:


#dropping the target variable column from the dataset
X = df1.drop('Default', axis=1)
y = df1['Default']


# In[19]:


#splitting data into test train in a ratio of 0.2 : 0.8 respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model1 = DecisionTreeClassifier(random_state=0)
model1.fit(X_train, y_train)




# In[21]:


#Performing 10 fold cross validation in data set and recoridng the mean score through ten iterations
cv_scores = cross_val_score(model1, X, y, cv=10)
mean_cv_score = cv_scores.mean()


# In[23]:


#saving the predictied values from test data in y_pred variable
y_pred = model1.predict(X_test)


# In[25]:


#preparing the evaluation matrices
c_matrix_1 = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model1.predict_proba(X_test)[:,1])


# In[27]:


#printing the evaluation matrices
print("Mean Cross-Validation Accuracy: {:.3f}".format(mean_cv_score))
print("Model accuracy:", accuracy)
print("ROC AUC: {:.3f}".format(roc_auc))


# In[29]:


#prinitng the cost matrix which is primary evaluation matrix for our business case
cost_matrix = np.array([[0, 1 ], [10, 0]])
coost = np.sum(cost_matrix * c_matrix_1)
print("Cost Matrix:\n", cost_matrix)
print("Total Cost:", coost)


# In[31]:


plt.figure(figsize=(6, 4))
sns.heatmap(c_matrix_1, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()


# In[33]:


#print decision tree
tree = export_text(model1, feature_names=list(X.columns))
print(tree)


# In[35]:


#visualising the importance of features/attributes to assist in omitting less important features
importances = model1.feature_importances_

# Create a DataFrame for visualization
features = X.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Print feature importances
print(importances_df)


# In[37]:


# Plot feature importances
plt.figure(figsize=(10,6))
plt.barh(importances_df['Feature'], importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Decision Tree')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()


# In[39]:


#Trying to omit attributes which have importance less than 0.030799 to improve model performance based on above results

df_new_1 = df1.filter(items=[
                            "Default", "amount","checkingstatus1", "duration", "purpose", "status" ,
                            "property","history", "savings", "residence", "age", "employ"
                           ])

df_new_1


# In[41]:


###
#####
########
##########REPEATING ALL ABOVE STEPS AGAIN IN BELOW CODE


# In[43]:


df_new_1.isna().sum()


# In[43]:


X = df_new_1.drop('Default', axis=1)
y = df_new_1['Default']


# In[51]:


#splitting data again into test train using sklearn test train split with cusotme valuke of test as 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)


# In[45]:


#Performing 10 fold cross validation in data set and recoridng the mean score through ten iterations
cv_scores = cross_val_score(model2, X, y, cv=10)
mean_cv_score = cv_scores.mean()

y_pred = model2.predict(X_test)

c_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model2.predict_proba(X_test)[:,1])
print("Mean Cross-Validation Accuracy: {:.3f}".format(mean_cv_score))
print("Accuracy: {:.3f}".format(accuracy))

print("Confusion Matrix:\n", c_matrix)
print("ROC AUC: {:.3f}".format(roc_auc))


# In[47]:


#printing the cost matrix

cost_matrix = np.array([[0, 1 ], [10, 0]])
coost = np.sum(cost_matrix * c_matrix)
print("Cost Matrix:\n", cost_matrix)
print("Total Cost:", coost)

## printing visual confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(c_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()


# In[49]:


df_new_2 = df.filter(items=[
                            "Default", "amount","checkingstatus1", "duration", "purpose",
                            "property","history", "age", "employ"
                           ])


# In[55]:


X = df_new_2.drop('Default', axis=1)
y = df_new_2['Default']


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model3 = DecisionTreeClassifier(random_state=42)
model3.fit(X_train, y_train)

#Performing 10 fold cross validation in data set and recoridng the mean score through ten iterations
cv_scores = cross_val_score(model3, X, y, cv=10)
mean_cv_score = cv_scores.mean()

y_pred = model3.predict(X_test)

c_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model3.predict_proba(X_test)[:,1])
print("Mean Cross-Validation Accuracy: {:.3f}".format(mean_cv_score))
print("Accuracy: {:.3f}".format(accuracy))

print("Confusion Matrix:\n", c_matrix)
print("ROC AUC: {:.3f}".format(roc_auc))


# In[51]:


#printing the cost matrix

cost_matrix = np.array([[0, 1 ], [10, 0]])
coost = np.sum(cost_matrix * c_matrix)
print("Cost Matrix:\n", cost_matrix)
print("Total Cost:", coost)

##visual conf matrix

plt.figure(figsize=(6, 4))
sns.heatmap(c_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()


# In[53]:


X = df_new_2.drop('Default', axis=1)
y = df_new_2['Default']


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model4 = DecisionTreeClassifier(random_state=42, max_depth = 5)
model4.fit(X_train, y_train)

########Performing 10 fold cross validation in data set and recoridng the mean score through ten iterations.
cv_scores = cross_val_score(model4, X, y, cv=10)
mean_cv_score = cv_scores.mean()

y_pred = model4.predict(X_test)

c_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model4.predict_proba(X_test)[:,1])
print("Mean Cross-Validation Accuracy: {:.3f}".format(mean_cv_score))
print("Accuracy: {:.3f}".format(accuracy))

print("Confusion Matrix:\n", c_matrix)
print("ROC AUC: {:.3f}".format(roc_auc))


# In[55]:


##############printing the cost matrix

cost_matrix = np.array([[0, 1 ], [10, 0]])
coost = np.sum(cost_matrix * c_matrix)
print("Cost Matrix:\n", cost_matrix)
print("Total Cost:", coost)


# In[57]:


############vizual conf matrix

plt.figure(figsize=(6, 4))
sns.heatmap(c_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()


# In[59]:


prob_y_pred= model4.predict_proba(X_test)[:, 1] 

#calculating and plotting ROC curve
fpr, tpr, thresholds = roc_curve(y_test, prob_y_pred)
roc_auc = auc(fpr, tpr)


# In[61]:


# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='black') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




