#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np 
import pandas as pd
#visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
#EDA
from ydata_profiling import ProfileReport
#data splitting
from sklearn.model_selection import train_test_split
#Data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#Data Modeling
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[16]:


# Data Import 
data = pd.read_csv(r"C:\Users\Lenovo\Desktop\dataset\palmertech short.csv")
data.head()


# # Data Analyse

# In[17]:


data.isnull().sum()


# In[18]:


data1 = data.dropna()


# In[19]:


data1.head()


# In[20]:


data1.isnull().sum()


# In[21]:


data1.info()


# In[22]:


#Statistical Analysis
data1.describe()


# In[23]:


#Exploratory Data Analysis
from pandas_profiling import ProfileReport
design_report = ProfileReport(data1)
design_report


# In[24]:


data1.shape


# In[65]:


le = LabelEncoder()
data2 = data1.apply(le.fit_transform)
data2.head(50)


# In[31]:


# Data imbalancing check
target_value = data2["is_promoted"].value_counts()/len(data2)
target_value


# In[32]:


#splitting the data
x = data2.drop(columns="is_promoted",axis=1)
y = data2["is_promoted"]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 50,test_size = .2)


# In[33]:


# data balancing
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
x_smot,y_smot = oversample.fit_resample(x,y)


# In[34]:


print(x_smot.shape)
print(y_smot.shape)
print(x_train.shape)
print(y_train.shape)
print(x.shape)
print(y.shape)


# In[35]:


# resplitting
X = x_smot
Y = y_smot
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=15,test_size=.2)


# In[36]:


print(y_test.unique())
Counter(y_train)


# In[37]:


#Scaling Data
ss =StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# # Apply Algorithms

# In[38]:


#model 1 = Logistic Regression 
lr = LogisticRegression(C=0.1)
model = lr.fit(X_train,Y_train)
lr_pred = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(Y_test,lr_pred)
lr_acc_score = accuracy_score(Y_test,lr_pred)
print("confusion_matrix:")
print(lr_conf_matrix)
print("Accuracy of Logistic Regression:",lr_acc_score*100,"\n")
print("Classification Report:","\n",classification_report(Y_test,lr_pred))


# In[39]:


#model 2 = Naive Bayes 
nb = GaussianNB()
nb.fit(X_train,Y_train)
nb_pred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(Y_test,nb_pred)
nb_acc_score = accuracy_score(Y_test,nb_pred)
print("confusion_matrix:","\n",nb_conf_matrix)
print("Accuracy of Navive Bayes model:",nb_acc_score*100)
print("Classification Report:","\n",classification_report(Y_test,nb_pred))


# In[40]:


# model 3 = Random Forest Classifier
rf = RandomForestClassifier(n_estimators=11,random_state=10,max_depth=29)
rf.fit(X_train,Y_train)
rf_pred = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(Y_test,rf_pred)
rf_acc_score = accuracy_score(Y_test,rf_pred)
print("Confusion Matrix:","\n",rf_conf_matrix)
print("Accuracy Score of model:",rf_acc_score*100)
print("Classification Report:","\n",classification_report(Y_test,rf_pred))


# In[41]:


#Model 4  = K-NeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,Y_train)
knn_pred = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(Y_test,knn_pred)
knn_acc_score = accuracy_score(Y_test,knn_pred)
print("ConfusionMatrix:","\n",knn_conf_matrix)
print("Accuracy_score of knn model:",knn_acc_score*100)
print("Classification Report:","\n",classification_report(Y_test,knn_pred))


# In[42]:


#model 5 = Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt_pred = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(Y_test,dt_pred)
dt_acc_score = accuracy_score(Y_test,dt_pred)
print("Confusion matrix:","\n",dt_conf_matrix)
print("Accurecy score of dt model:",dt_acc_score*100)
print("Classification Report:","\n",classification_report(Y_test,dt_pred))


# In[46]:


# model 6 = Support vector machine
svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
svc.fit(X_train,Y_train)
svc_pred = svc.predict(X_test)
svc_conf_matrix =confusion_matrix(Y_test,svc_pred)
svc_acc_score = accuracy_score(Y_test,svc_pred)
print("Confusion matrix:","\n",svc_conf_matrix)
print("Acccuracy score of svc model:",svc_acc_score*100)
print("Classification report:","\n",classification_report(Y_test,svc_pred))


# In[58]:


# Extreme Boost Gradient 
import xgboost as xgb
xb = xgb.XGBClassifier(n_estimators=100, max_depth = 50, learning_rate = 0.7)
xb.fit(X_train,Y_train)
xgb_pred = xb.predict(X_test)
xgb_conf_matrix = confusion_matrix(Y_test,xgb_pred)
xgb_acc_score = accuracy_score(Y_test,xgb_pred)
print("Confusion matrix: \n",xgb_conf_matrix)
print("Accuracy_score:",xgb_acc_score*100)
print("Classifier report:\n",classification_report(Y_test,xgb_pred))


# In[87]:


#Save the XGBoost model
import pickle
# Dump the trained Naive Bayes classifier with Pickle
xb_pkl_filename = 'C:/Users/Lenovo/Desktop/PROJECT SAVED/xgboost_model.pkl'
# Open the file to save as pkl file
xb_Model_pkl = open(xb_pkl_filename, 'wb')
pickle.dump(xb, xb_Model_pkl)
# Close the pickle instances
xb_Model_pkl.close()


# # Accuracy Comparision 

# In[59]:


model_ev =pd.DataFrame({"models":["Logistic regression","Naive Bayes","Random Tree Forest",
                                "K-Nearest Neighbor", "Decision Tree","Support Vector Machine","XGBoost Classifier"],
                        "Accuracy":[lr_acc_score*100,nb_acc_score*100,rf_acc_score*100,
                                     knn_acc_score*100,dt_acc_score*100,svc_acc_score*100,xgb_acc_score*100]})
display(model_ev)                        


# # Graphical Representation 

# In[88]:


import matplotlib
matplotlib.use('TkAgg')
lr_false_positive_rate,lr_true_negative_rate,lr_threshold = roc_curve(Y_test,lr_pred)
nb_false_positive_rate,nb_true_negative_rate,nb_threshold = roc_curve(Y_test,nb_pred)
rf_false_positive_rate,rf_true_nagative_rate,rf_threshold = roc_curve(Y_test,rf_pred)
knn_false_positive_rate,knn_true_negative_rate,knn_threshold = roc_curve(Y_test,knn_pred)
dt_false_positive_rate,dt_true_nagative_rate,dt_threshold = roc_curve(Y_test,dt_pred)
svc_false_positive_rate,svc_true_negative_rate,svc_threshold = roc_curve(Y_test,svc_pred)
svc_false_positive_rate,svc_true_negative_rate,svc_threshold = roc_curve(Y_test,xgb_pred)

sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.title("Reciever Operating Characteristic Curve")
plt.plot(lr_false_positive_rate,lr_true_negative_rate,label="Logistic Regression")
plt.plot(nb_false_positive_rate,nb_true_negative_rate,label="Naive Bayes")
plt.plot(rf_false_positive_rate,rf_true_nagative_rate,label="Random Forest")
plt.plot(knn_false_positive_rate,knn_true_negative_rate,label="K-Nearest Neighbor")
plt.plot(dt_false_positive_rate,dt_true_nagative_rate,label="Decision Tree")
plt.plot(svc_false_positive_rate,svc_true_negative_rate,label="Support Vector Classifier")
plt.plot(svc_false_positive_rate,svc_true_negative_rate,label="XGBoost Classifier")

plt.plot([0,1],ls="--")
plt.plot([0,0],[1,0],c=".5")
plt.plot([1,1],c='.5')         
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")  
plt.legend()
plt.show()         


# # Data Prediction

# In[67]:


data3 = np.array([[12213,7,31,0,0,2,0,15,4,2,1,0,11]])
prediction = xb.predict(data3)
print(prediction)


# In[70]:


data3 = np.array([[14504,7,20,0,1,2,0,13,4,5,1,0,12]])
prediction = xb.predict(data3)
print(prediction)


# In[ ]:




