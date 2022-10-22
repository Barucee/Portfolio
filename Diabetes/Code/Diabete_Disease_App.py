###################################################################################################################
## This dataset has been found on Kaggle. However it is originally from the National Institute of Diabetes and   ##
## Digestive and Kidney Diseases.                                                                                ##
## This dataset is used to predict whether or not a patient has diabetes, based on given features/diagnostic     ##
## measurements.                                                                                                 ##
## Several constraints were placed on the selection of these instances from a larger database. In particular, all## 
## patients here are females at least 21 years old of Pima Indian heritage.                                      ##
## - Input :                                                                                                     ##
##      - Pregnancies: Number of times pregnant                                                                  ##
##      - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test                      ##
##      - BloodPressure: Diastolic blood pressure (mm Hg)                                                        ##
##      - SkinThickness: Triceps skin fold thickness (mm)                                                        ##
##      - Insulin: 2-Hour serum insulin (mu U/ml)                                                                ##
##      - BMI: Body mass index (weight in kg/(height in m)^2)                                                    ##
##      - DiabetesPedigreeFunction: Diabetes pedigree function                                                   ##
##      - Age: Age (years)                                                                                       ##
## - Output :                                                                                                    ##
##      - Outcome: Class variable (0 or 1)                                                                       ##
###################################################################################################################



#import the libraries

from xml.dom import registerDOMImplementation
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

#import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


#import the dataset
path = "C:/Users/33646/Documents/Portfolio/Diabetes/Data/diabetes.csv"
print(path)
df = pd.read_csv(path)
df_raw = df.copy()




##################################### Data Treatment #####################################

###### Treatment 0 ######

#Calculating the number of 0

zero_features = ['Glucose','BloodPressure','SkinThickness',"Insulin",'BMI']
total_count = df_raw['Glucose'].count()

def numberof0():
    for feature in zero_features:
        zero_count = df_raw[df_raw[feature]==0][feature].count()
        st.markdown(f'{feature} has {zero_count} 0 values, which represent in percent {round(100*zero_count/total_count,2)} %')

#Plot histogram before treatment

plt.rcParams['figure.figsize'] = 40,60
sns.set(font_scale = 3)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.5)
        
fig_before_treat_0value, axs_before_0_treatment = plt.subplots(3, 2, sharey=True)
        
for i, _c in enumerate(df.loc[:,'Glucose':'BMI'].columns):
    ax = axs_before_0_treatment.flat[i]
    sns.histplot(data=df, x=_c, hue="Outcome",kde=True,palette="YlGnBu", ax=ax)
                
# Replacing the 0 by mean            

zero_features = ['Glucose','BloodPressure','SkinThickness',"Insulin",'BMI']
diabetes_mean = df[zero_features].mean()
df[zero_features] = df[zero_features].replace(0, diabetes_mean)

#Plot histogram after treatment

plt.rcParams['figure.figsize'] = 40,60
sns.set(font_scale = 3)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.5)
        
fig_after_treat_0value, axs_after_0_treatment = plt.subplots(4, 2, sharey=True)
        
for i, _c in enumerate(df.loc[:,:'Age'].columns):
    ax = axs_after_0_treatment.flat[i]
    sns.histplot(data=df, x=_c, hue="Outcome",kde=True,palette="YlGnBu", ax=ax)






###### Treatment Outliers ######

#Plot boxplot before treatment

fig_before_outliers_treatment , ax_before_outliers_treatment = plt.subplots(figsize = (20,20))
#rotate the xlabel
sns.boxplot(data = df, ax = ax_before_outliers_treatment)
ax_before_outliers_treatment.set_xticklabels(ax_before_outliers_treatment.get_xticklabels(),rotation=90)

#Outlier treatment

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        st.write(variable, "has outliers")
        
def replace_with_thresholds(dataframe, numeric_columns):
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, df.columns)

#Plot boxplot after treatment

fig_after_outliers_treatment , ax_after_outliers_treatment = plt.subplots(figsize = (20,20))
#rotate the xlabel
sns.boxplot(data = df, ax = ax_after_outliers_treatment)
ax_after_outliers_treatment.set_xticklabels(ax_after_outliers_treatment.get_xticklabels(),rotation=90)










##################################### Data Visualization #####################################

###### Repartition Outcome ######

fig_repartition_outcome , ax_repartition_outcome = plt.subplots(figsize=(12,7))
sns.countplot(x='Outcome', data=df, palette='YlGnBu', ax = ax_repartition_outcome)

###### Pairplot ######

plot_pairplot = sns.pairplot(df, hue='Outcome', palette='YlGnBu',)

for ax in plot_pairplot.axes.flatten():
    # rotate x axis labels
    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    # rotate y axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    # set y labels alignment
    ax.yaxis.get_label().set_horizontalalignment('right')
    
###### Correlation Matrix ######

corr=df.corr().round(2)

fig_heatmap, ax_heatmap = plt.subplots(figsize=(18,6))
sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.set_palette("bright")
sns.set_style("white")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, annot_kws={"size": 10}, cmap="YlGnBu",mask=mask,cbar=True, ax=ax_heatmap)
plt.title('Correlation Matrix')







##################################### Data Classification #####################################

X = df.drop(['Outcome'], axis=1)
X = np.array(X)
y = df['Outcome']
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)

cv = KFold(n_splits=10, random_state=1, shuffle=True)

###### function confusion matrix ######

def plot_confusion_matrix(y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7,7)) 
    sns.set(font_scale=3.0)
    sns.heatmap(cm, annot=True, fmt='d',annot_kws={'size': 30})
    plt.xlabel('y prediction')
    plt.ylabel('y')

###### Logistic Regression ######

LR = LogisticRegression(solver='liblinear', penalty="l2", C=0.001)
y_pred_LR = cross_val_predict(LR, X, y, cv=10)
matrix_LR = plot_confusion_matrix(y_pred_LR)
Results_LR = pd.DataFrame.from_dict({'Model': ['Logistic Regression'], 'Accuracy': [accuracy_score(y, y_pred_LR)], 'Precision': [precision_score(y, y_pred_LR)], 'Recall': [recall_score(y, y_pred_LR)], 'F1': [f1_score(y, y_pred_LR)], 'AUC': [roc_auc_score(y, y_pred_LR)]})

###### Linear Discriminant Analysis ######

LDA = LinearDiscriminantAnalysis(solver="svd")
y_pred_LDA = cross_val_predict(LDA, X, y, cv=10)
matrix_LDA = plot_confusion_matrix(y_pred_LDA)
Results_LDA = pd.DataFrame.from_dict({'Model': ['Linear Discriminant analysis'], 'Accuracy': [accuracy_score(y, y_pred_LDA)], 'Precision': [precision_score(y, y_pred_LDA)], 'Recall': [recall_score(y, y_pred_LDA)], 'F1': [f1_score(y, y_pred_LDA)], 'AUC': [roc_auc_score(y, y_pred_LDA)]})

###### KNeighbors Classifier ######

KNN = KNeighborsClassifier(n_neighbors=29)
y_pred_KNN = cross_val_predict(KNN, X, y, cv=10)
matrix_KNN = plot_confusion_matrix(y_pred_KNN)
Results_KNN = pd.DataFrame.from_dict({'Model': ['K-Neighbors Classifier'], 'Accuracy': [accuracy_score(y, y_pred_KNN)], 'Precision': [precision_score(y, y_pred_KNN)], 'Recall': [recall_score(y, y_pred_KNN)], 'F1': [f1_score(y, y_pred_KNN)], 'AUC': [roc_auc_score(y, y_pred_KNN)]})

###### Decision Tree Classifier ######

DTC = DecisionTreeClassifier(criterion='gini', max_depth=4)
y_pred_DTC = cross_val_predict(DTC, X, y, cv=10)
matrix_DTC = plot_confusion_matrix(y_pred_DTC)
Results_DTC = pd.DataFrame.from_dict({'Model': ['Decision Tree Classifier'], 'Accuracy': [accuracy_score(y, y_pred_DTC)], 'Precision': [precision_score(y, y_pred_DTC)], 'Recall': [recall_score(y, y_pred_DTC)], 'F1': [f1_score(y, y_pred_DTC)], 'AUC': [roc_auc_score(y, y_pred_DTC)]})

###### Random Forest Classifier ######

RFC = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=9)
y_pred_RFC = cross_val_predict(RFC, X, y, cv=10)
matrix_RFC = plot_confusion_matrix(y_pred_RFC)
Results_RFC = pd.DataFrame.from_dict({'Model': ['Random Forest Classifier'], 'Accuracy': [accuracy_score(y, y_pred_RFC)], 'Precision': [precision_score(y, y_pred_RFC)], 'Recall': [recall_score(y, y_pred_RFC)], 'F1': [f1_score(y, y_pred_RFC)], 'AUC': [roc_auc_score(y, y_pred_RFC)]})

###### Gaussian NB ######

GNB = GaussianNB(var_smoothing=0.12328467394420659)
y_pred_GNB = cross_val_predict(GNB, X, y, cv=10)
matrix_GNB = plot_confusion_matrix(y_pred_GNB)
Results_GNB = pd.DataFrame.from_dict({'Model': ['Gaussian NB'], 'Accuracy': [accuracy_score(y, y_pred_GNB)], 'Precision': [precision_score(y, y_pred_GNB)], 'Recall': [recall_score(y, y_pred_GNB)], 'F1': [f1_score(y, y_pred_GNB)], 'AUC': [roc_auc_score(y, y_pred_GNB)]})

###### Support Vector Machine ######

SVC_model = LinearSVC(max_iter=10000)
y_pred_SVC = cross_val_predict(SVC_model, X, y, cv=10)
matrix_SVC = plot_confusion_matrix(y_pred_SVC)
Results_SVC = pd.DataFrame.from_dict({'Model': ['Support Vector Machine'], 'Accuracy': [accuracy_score(y, y_pred_SVC)], 'Precision': [precision_score(y, y_pred_SVC)], 'Recall': [recall_score(y, y_pred_SVC)], 'F1': [f1_score(y, y_pred_SVC)], 'AUC': [roc_auc_score(y, y_pred_SVC)]})

###### Gradient Boosting ######

GBC = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,min_samples_split=5)
y_GBC = cross_val_predict(GBC, X, y, cv=10)
matrix_GBC = plot_confusion_matrix(y_GBC)
Results_GBC = pd.DataFrame.from_dict({'Model': ['Gradient Boosting Classifier'], 'Accuracy': [accuracy_score(y, y_GBC)], 'Precision': [precision_score(y, y_GBC)], 'Recall': [recall_score(y, y_GBC)], 'F1': [f1_score(y, y_GBC)], 'AUC': [roc_auc_score(y, y_GBC)]})

###### LightGBM ######

LGBMC = LGBMClassifier(colsample_bytree=0.7,max_depth=15,min_split_gain=0.4, n_estimators=400, num_leaves=50, reg_alpha=1.1, reg_lambda=1.2, subsample=0.9, subsample_freq=20)
y_LGBMC = cross_val_predict(LGBMC, X, y, cv=10)
matrix_LGBMC = plot_confusion_matrix(y_LGBMC)
Results_LGBMC = pd.DataFrame.from_dict({'Model': ['LightGBM Classifier'], 'Accuracy': [accuracy_score(y, y_LGBMC)], 'Precision': [precision_score(y, y_LGBMC)], 'Recall': [recall_score(y, y_LGBMC)], 'F1': [f1_score(y, y_LGBMC)], 'AUC': [roc_auc_score(y, y_LGBMC)]})

###### XG-Boost ######

XGB = XGBClassifier(colsample_bytree = 1.0, gamma = 2, max_depth=3,min_child_weight=10, subsample=1)
y_XGB = cross_val_predict(XGB, X, y, cv=10)
matrix_XGB = plot_confusion_matrix(y_XGB)
Results_XGB = pd.DataFrame.from_dict({'Model': ['XG-Boost Classifier'], 'Accuracy': [accuracy_score(y, y_XGB)], 'Precision': [precision_score(y, y_XGB)], 'Recall': [recall_score(y, y_XGB)], 'F1': [f1_score(y, y_XGB)], 'AUC': [roc_auc_score(y, y_XGB)]})

###### MLP classifier ######

MLP = MLPClassifier()
y_MLP = cross_val_predict(MLP, X, y, cv=10)
matrix_MLP = plot_confusion_matrix(y_MLP)
Results_MLP = pd.DataFrame.from_dict({'Model': ['MLP Classifier'], 'Accuracy': [accuracy_score(y, y_MLP)], 'Precision': [precision_score(y, y_MLP)], 'Recall': [recall_score(y, y_MLP)], 'F1': [f1_score(y, y_MLP)], 'AUC': [roc_auc_score(y, y_MLP)]})

###### Neural Network ######
ANN_model = tf.keras.models.Sequential()
ANN_model.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(8,)))
ANN_model.add(tf.keras.layers.Dropout(0.2))

ANN_model.add(tf.keras.layers.Dense(units=400, activation='relu'))
ANN_model.add(tf.keras.layers.Dropout(0.2))

ANN_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ANN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #We use binary_crossentropy because we have binary outcome

history = ANN_model.fit(X_train, y_train, epochs=200)

y_ANN = ANN_model.predict(X_test)
y_pred_ANN = (y_ANN > 0.5)

acc = round(accuracy_score(y_test, y_pred_ANN), 2)
cm_ann = confusion_matrix(y_test, y_pred_ANN)
matrix_ANN, ax = plt.subplots(figsize=(7,7)) 
sns.set(font_scale=3.0)
matrix_ANN = sns.heatmap(cm_ann, annot=True, fmt='d',annot_kws={'size': 30})
plt.xlabel('y prediction')
plt.ylabel('y')

Results_ANN = pd.DataFrame.from_dict({'Model': ['Neural Network'], 'Accuracy': [accuracy_score(y_test, y_pred_ANN)], 'Precision': [precision_score(y_test, y_pred_ANN)], 'Recall': [recall_score(y_test, y_pred_ANN)], 'F1': [f1_score(y_test, y_pred_ANN)], 'AUC': [roc_auc_score(y_test, y_pred_ANN)]})


###### merge of the conclusion dataframe ######

Results = pd.concat([Results_LR, Results_LDA, Results_KNN, Results_DTC,Results_RFC,Results_GNB,Results_SVC,Results_GBC,Results_LGBMC,Results_XGB,Results_MLP, Results_ANN], ignore_index=True)


###### Classification app ######

MLP.fit(X,y)















##################################### create a streamlit application #####################################

st.title("Diabete Disease App 🩺")

#Attention possibilité de disparition des graphs a caude de cela
st.set_option('deprecation.showPyplotGlobalUse', False)

pages = st.sidebar.selectbox('Select the page', ['Introduction','Data Treatment ⚒️​','Data visualization 📊', 'About the models 🧭', 'Classification 📈','Conclusion', 'Classification App 🚀'])

if pages == 'Introduction':
    
    st.header("Introduction to this classification application")
    st.markdown("This application will help you to classify if someone has diabete or not")
    st.markdown("We will first display some information about the dataset")
    
    #Display of some informations about the dataset
    st.dataframe(df_raw.head(), width=1000,)
    st.markdown(f"The dataframe contains {df_raw.shape[0]} rows and {df_raw.shape[1]} columns")
    st.markdown(f"There are {df_raw.isna().sum().sum()} missing values in the dataset.")


elif pages == 'Data Treatment ⚒️​':
    
    treatment = st.sidebar.selectbox('Select the data treatment', ['0 value','Outliers'])
    
    if treatment == '0 value':
        
        st.header("Treatment of the 0 values")

        value0 = st.sidebar.selectbox('Which phase of the treatment you want',["Whats the problem ?","Histograms before treatment","Histograms after treatment"])
        
        if value0 == 'Whats the problem ?' :
            
            st.dataframe(df_raw.describe(),width=1000,)
        
            st.markdown("We can see that some variabe has 0 as value which can be very strange for the glucose, bloodpressure, skinthickness, insulin and BMI. Let's delve deeper into these variables.")

            numberof0()
            
        if value0 == 'Histograms before treatment' :
                            
            st.pyplot(fig_before_treat_0value)
        
        if value0 == 'Histograms after treatment' :
            
            st.markdown("Let's treat the 0 value by replacing them by the mean of each column")
            st.pyplot(fig_after_treat_0value)
            

    elif treatment == 'Outliers':
        
        st.header("Outliers treatment")
        
        outliers = st.sidebar.selectbox("Which phase you want",["Boxplot before outliers treatment","Boxplot after outliers treatment"])
        
        if outliers == 'Boxplot before outliers treatment' :
        
            st.pyplot(fig_before_outliers_treatment)
            
            for col in df.columns:
                has_outliers(df, col)
        
        elif outliers == 'Boxplot after outliers treatment' :
            
            st.pyplot(fig_after_outliers_treatment)
            
            
elif pages == "Data visualization 📊" :

    st.header("Visualization")
    
    visualization = st.sidebar.selectbox("Choose a visualization", ["Repartition of the outcome","Pairplot","Correlation Matrix"])
        
    if visualization == "Repartition of the outcome" :
        
        st.pyplot(fig_repartition_outcome)
        
    elif visualization == "Pairplot" :
    
        st.pyplot(plot_pairplot)
        
    elif visualization == "Correlation Matrix" :
        
        st.pyplot(fig_heatmap)
        

elif pages == 'About the models 🧭' :
    
    model = st.sidebar.selectbox("Choose a model",["Logistic Regression", "Linear discriminant analysis", "Kneighbors classifier", "Decision tree", "Random Forest", "Gaussian NB", "Support Vector Machine", "Gradient Boosting", "LightGBM", "XG-Boost Classifier", "MLP", "Neural Network"])

    st.header("About the models 🧭")
    
    if model == "Logistic Regression" :
        
        st.markdown("In the Machine Learning world, Logistic Regression is a kind of parametric classification model, despite having the word ‘regression’ in its name.")
        st.markdown("Logistic regression is a statistical analysis method to predict a binary outcome, such as yes or no, based on prior observations of a data set.")
        st.markdown("A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables. For example, a logistic regression could be used to predict whether a political candidate will win or lose an election or whether a high school student will be admitted or not to a particular college. These binary outcomes allow straightforward decisions between two alternatives.")
        st.markdown("Statisticians and citizen data scientists must keep a few assumptions in mind when using logistic regression. For starters, the variables must be independent of one another. So, for example, zip code and gender could be used in a model, but zip code and state would not work.")
        st.markdown("The maths behind the model are :")
        st.latex(r'''P(x) = \frac{1}{1+e^{-x}}''')  
        st.markdown("where P(X) is the probability to be 1.")
        st.markdown("Other less transparent relationships between variables may get lost in the noise when logistic regression is used as a starting point for complex machine learning and data science applications. For example, data scientists may spend considerable effort to ensure that variables associated with discrimination, such as gender and ethnicity, are not included in the algorithm. However, these can sometimes get indirectly woven into the algorithm via variables that were not thought to be correlated, such as zip code, school or hobbies.")
        st.markdown("Another assumption is that the raw data should represent unrepeated or independent phenomena. For example, a survey of customer satisfaction should represent the opinions of separate people. But these results would be skewed if someone took the survey multiple times from different email addresses to qualify for a reward.")
        st.markdown("Advantages :")
        st.markdown("- Logistic regression is easier to implement, interpret, and very efficient to train.\n - It makes no assumptions about distributions of classes in feature space.\n - It can easily extend to multiple classes(multinomial regression) and a natural probabilistic view of class predictions.\n - It not only provides a measure of how appropriate a predictor(coefficient size)is, but also its direction of association (positive or negative).\n - It is very fast at classifying unknown records.\n - Good accuracy for many simple data sets and it performs well when the dataset is linearly separable.\n - It can interpret model coefficients as indicators of feature importance.\n - Logistic regression is less inclined to over-fitting but it can overfit in high dimensional datasets.One may consider Regularization (L1 and L2) techniques to avoid over-fittingin these scenarios.")
        st.markdown("Disadvantages :")
        st.markdown("- If the number of observations is lesser than the number of features, Logistic Regression should not be used, otherwise, it may lead to overfitting.\n - It constructs linear boundaries.\n - The major limitation of Logistic Regression is the assumption of linearity between the dependent variable and the independent variables.\n - It can only be used to predict discrete functions. Hence, the dependent variable of Logistic Regression is bound to the discrete number set.\n - Non-linear problems can’t be solved with logistic regression because it has a linear decision surface. Linearly separable data is rarely found in real-world scenarios.\n - Performs well when the dataset is linearly separable.	Logistic Regression requires average or no multicollinearity between independent variables.\n - It is tough to obtain complex relationships using logistic regression. More powerful and compact algorithms such as Neural Networks can easily outperform this algorithm.\n - In Linear Regression independent and dependent variables are related linearly. But Logistic Regression needs that independent variables are linearly related to the log odds (log(p/(1-p)).")
    
    elif model == "Linear discriminant analysis" :
        
        st.markdown("Linear Discriminant Analysis is a technique for classifying binary and non-binary features using and linear algorithm for learning the relationship between the dependent and independent features. It uses the Fischer formula to reduce the dimensionality of the data so as to fit in a linear dimension. LDA is a multi-functional algorithm, it is a classifier, dimensionality reducer and data visualizer. The aim of LDA is:")
        st.markdown("- To minimize the inter-class variability which refers to classifying as many similar points as possible in one class. This ensures fewer misclassifications.\n - To maximize the distance between the mean of classes, the mean is placed as far as possible to ensure high confidence during prediction.")
        st.markdown("LDA makes some assumptions about the data:")
        st.markdown("- Assumes the data to be distributed normally or Gaussian distribution of data points i.e. each feature must make a bell-shaped curve when plotted. \n - Each of the classes has identical covariance matrices.")
        st.markdown("However, it is worth mentioning that LDA performs quite well even if the assumptions are violated.")
        st.markdown("It works on a simple step-by-step basis. Here is an example. These are the three key steps.")
        st.latex(r'''S_b = \sum_{i=1}^{g}N_i(\bar{x}_i - \bar{x})(\bar{x}_i - \bar{x})^T''')
        st.latex(r'''S_W = \sum_{i=1}^g(N_i - 1)S_i =\sum_{i=1}^g\sum_{i=1}^{N_i}N_i(\bar{x}_i - \bar{x})(\bar{x}_i - \bar{x})^T''')
        st.latex(r'''P_{lda} = \argmax_{P} \frac{|P^TS_bP|}{|P^TS_wP|}''')
        st.markdown("where P is the lower-dimensional space projection. This is also known as Fisher’s criterion.")
        
    elif model == "Kneighbors classifier":
        
        st.markdown("In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric supervised learning method first developed by Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. It is used for classification and regression. In both cases, the input consists of the k closest training examples in a data set. The output depends on whether k-NN is used for classification or regression:")
        st.markdown("KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.")
        st.markdown("In the classification phase, k is a user-defined constant, and an unlabeled vector (a query or test point) is classified by assigning the label which is most frequent among the k training samples nearest to that query point.")
        st.markdown("The best choice of k depends upon the data; generally, larger values of k reduces effect of the noise on the classification, but make boundaries between classes less distinct. A good k can be selected by various heuristic techniques (see hyperparameter optimization). The special case where the class is predicted to be the class of the closest training sample (i.e. when k = 1) is called the nearest neighbor algorithm.")
        st.markdown("The accuracy of the k-NN algorithm can be severely degraded by the presence of noisy or irrelevant features, or if the feature scales are not consistent with their importance. Much research effort has been put into selecting or scaling features to improve classification. A particularly popular[citation needed] approach is the use of evolutionary algorithms to optimize feature scaling. Another popular approach is to scale features by the mutual information of the training data with the training classes.[citation needed]")
        st.markdown("In binary (two class) classification problems, it is helpful to choose k to be an odd number as this avoids tied votes. One popular way of choosing the empirically optimal k in this setting is via bootstrap method.")
        st.latex(r'''p(class=0) = \frac{count(class=0)}{(count(class=0) +count(class=1))}''')

    elif model == "Decision tree":
        
        st.markdown("Decision Tree is a Supervised Machine Learning Algorithm that uses a set of rules to make decisions, similarly to how humans make decisions.")
        st.markdown("Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.")
        st.markdown("The intuition behind Decision Trees is that you use the dataset features to create yes/no questions and continually split the dataset until you isolate all data points belonging to each class.")
        st.markdown("The ideal tree is the smallest tree possible, i.e. with fewer splits, that can accurately classify all data points.")
        st.markdown("On every split, the algorithm tries to divide the dataset into the smallest subset possible[2]. So, like any other Machine Learning algorithm, the goal is to minimize the loss function as much as possible.")
        st.markdown("But since you’re separating data points that belong to different classes, the loss function should evaluate a split based on the proportion of data points belonging to each class before and after the split.")
        st.markdown("Some advantages of decision trees are:")
        st.markdown("- Simple to understand and to interpret. Trees can be visualized.\n - Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.\n - The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.\n - Able to handle both numerical and categorical data. However, the scikit-learn implementation does not support categorical variables for now. Other techniques are usually specialized in analyzing datasets that have only one type of variable. See algorithms for more information.\n - Able to handle multi-output problems.\n - Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.\n - Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.\n - Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.")  
        st.markdown("The disadvantages of decision trees include:")
        st.markdown("- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.\n - Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.\n - Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations as seen in the above figure. Therefore, they are not good at extrapolation.\n - The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.\n - There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.\n - Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.")
        
    elif model == "Random Forest Classifier":
        
        st.markdown("Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.")
        st.markdown("One of the most important features of the Random Forest Algorithm is that it can handle the data set containing continuous variables as in the case of regression and categorical variables as in the case of classification. It performs better results for classification problems.")
        st.markdown("Ensemble uses two types of methods:")
        st.markdown("-  Bagging– It creates a different training subset from sample training data with replacement & the final output is based on majority voting. For example,  Random Forest.\n - Boosting– It combines weak learners into strong learners by creating sequential models such that the final model has the highest accuracy. For example,  ADA BOOST, XG BOOST")
        st.markdown("Bagging, also known as Bootstrap Aggregation is the ensemble technique used by random forest. Bagging chooses a random sample from the data set. Hence each model is generated from the samples (Bootstrap Samples) provided by the Original Data with replacement known as row sampling. This step of row sampling with replacement is called bootstrap. Now each model is trained independently which generates results. The final output is based on majority voting after combining the results of all models. This step which involves combining all the results and generating output based on majority voting is known as aggregation.")
        st.markdown("Advantages :")
        st.markdown("- One of the biggest advantages of random forest is its versatility. It can be used for both regression and classification tasks, and it’s also easy to view the relative importance it assigns to the input features.\n - Random forest is also a very handy algorithm because the default hyperparameters it uses often produce a good prediction result. Understanding the hyperparameters is pretty straightforward, and there’s also not that many of them. \n - One of the biggest problems in machine learning is overfitting, but most of the time this won’t happen thanks to the random forest classifier. If there are enough trees in the forest, the classifier won’t overfit the model.")
        st.markdown("Disadvantages :")
        st.markdown("- The main limitation of random forest is that a large number of trees can make the algorithm too slow and ineffective for real-time predictions. In general, these algorithms are fast to train, but quite slow to create predictions once they are trained. A more accurate prediction requires more trees, which results in a slower model. In most real-world applications, the random forest algorithm is fast enough but there can certainly be situations where run-time performance is important and other approaches would be preferred.\n - And, of course, random forest is a predictive modeling tool and not a descriptive tool, meaning if you’re looking for a description of the relationships in your data, other approaches would be better.")
        
    elif model == "Gaussian NB" :
        
        st.markdown("Naive Bayes are a group of supervised machine learning classification algorithms based on the Bayes theorem. It is a simple classification technique, but has high functionality. They find use when the dimensionality of the inputs is high. Complex classification problems can also be implemented by using Naive Bayes Classifier.")
        st.markdown("Naive Bayes Classifiers are based on the Bayes Theorem. One assumption taken is the strong independence assumptions between the features. These classifiers assume that the value of a particular feature is independent of the value of any other feature. In a supervised learning situation, Naive Bayes Classifiers are trained very efficiently. Naive Bayed classifiers need a small training data to estimate the parameters needed for classification. Naive Bayes Classifiers have simple design and implementation and they can applied to many real life situations.")
        
    elif model == "Suport Vector Machine":
        
        st.markdown("SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.")
        st.markdown("At first approximation what SVMs do is to find a separating line(or hyperplane) between data of two classes. SVM is an algorithm that takes the data as an input and outputs a line that separates those classes if possible.")
        st.markdown("According to the SVM algorithm we find the points closest to the line from both the classes.These points are called support vectors. Now, we compute the distance between the line and the support vectors. This distance is called the margin. Our goal is to maximize the margin. The hyperplane for which the margin is maximum is the optimal hyperplane.")
        st.markdown("Advantages :")
        st.markdown("- Accuracy\n - Works well on smaller cleaner datasets\n - It can be more efficient because it uses a subset of training points")
        st.markdown("Disadvantages :")
        st.markdown("- Isn’t suited to larger datasets as the training time with SVMs can be high\n - Less effective on noisier datasets with overlapping classes")
        
    elif model == "Gradient Boosting":
        
        st.markdown("Gradient boosting is a machine learning technique used in regression and classification tasks, among others. It gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually outperforms random forest. A gradient-boosted trees model is built in a stage-wise fashion as in other boosting methods, but it generalizes the other methods by allowing optimization of an arbitrary differentiable loss function.")
        st.markdown("Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.")
        st.markdown("There are three types of enhancements to basic gradient boosting that can improve performance:")
        st.markdown("- Tree Constraints: such as the depth of the trees and the number of trees used in the ensemble.\n - Weighted Updates: such as a learning rate used to limit how much each tree contributes to the ensemble.\n - Random sampling: such as fitting trees on random subsets of features and samples.")
        
    elif model == "LightGBM":
        
        st.markdown("Light GBM is a gradient boosting framework that uses tree based learning algorithm.")
        st.markdown("Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.")
        st.markdown("It has become difficult for the traditional algorithms to give results fast, as the size of the data is increasing rapidly day by day. LightGBM is called “Light” because of its computation power and giving results faster. It takes less memory to run and is able to deal with large amounts of data. Most widely used algorithm in Hackathons because the motive of the algorithm is to get good accuracy of results and also brace GPU leaning.")
        
    elif model == "XG-Boost Classifier":
        
        st.markdown("XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.")
        st.markdown("The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:")
        st.markdown("- Gradient Boosting algorithm also called gradient boosting machine including the learning rate.\n - Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.\n - Regularized Gradient Boosting with both L1 and L2 regularization")
        
    elif model == "MLP":
        
        st.markdown("The multilayer perceptron (MLP) is a feedforward artificial neural network model that maps input data sets to a set of appropriate outputs. An MLP consists of multiple layers and each layer is fully connected to the following one. The nodes of the layers are neurons with nonlinear activation functions, except for the nodes of the input layer. Between the input and the output layer there may be one or more nonlinear hidden layers.")
        
    elif model == "Neural Network":
        
        st.markdown("")
        

elif pages == "Classification 📈" :
    
    model = st.sidebar.selectbox("Choose a model",["Logistic Regression", "Linear discriminant analysis", "Kneighbors classifier", "Decision tree", "Random Forest", "Gaussian NB", "Support Vector Machine", "Gradient Boosting", "LightGBM", "XG-Boost Classifier", "MLP", "Neural Network"])

    if model == "Logistic Regression":
        
        st.header("Results Logistic Regression")
        st.pyplot(matrix_LR)
        st.dataframe(Results_LR,width=1000,)
    elif model == "Linear discriminant analysis":
        
        st.header("Results Linear discriminant analysis")
        st.pyplot(matrix_LDA)
        st.dataframe(Results_LDA,width=1000,)
    elif model == "Kneighbors classifier":
        
        st.header("Results Kneighbors classifier")
        st.pyplot(matrix_KNN)
        st.dataframe(Results_KNN,width=1000,)
    elif model == "Decision tree":
    
        st.header("Results Decision Tree Classifier")
        st.pyplot(matrix_DTC)
        st.dataframe(Results_DTC,width=1000,)
    elif model == "Random Forest":
        
        st.header("Results Random Forest")
        st.pyplot(matrix_RFC)
        st.dataframe(Results_RFC,width=1000,)
    elif model == "Gaussian NB":
        
        st.header("Results Gaussian NB")
        st.pyplot(matrix_GNB)
        st.dataframe(Results_GNB,width=1000,)
    elif model == "Support Vector Machine":
        
        st.header("Results Support vector Machine")
        st.pyplot(matrix_SVC)
        st.dataframe(Results_SVC,width=1000,)
    elif model == "Gradient Boosting" :
        
        st.header("Results Gradient Boosting")
        st.pyplot(matrix_GBC)
        st.dataframe(Results_GBC,width=1000,)
        
    elif model == "LightGBM":
        
        st.header("Results LightGBM")
        st.pyplot(matrix_LGBMC)
        st.dataframe(Results_LGBMC,width=1000,)

    elif model == "XG-Boost Classifier":
        
        st.header("Results XG-Boost")
        st.pyplot(matrix_XGB)
        st.dataframe(Results_XGB,width=1000,)
    elif model == "MLP":
        
        st.header("Results MLP")
        st.pyplot(matrix_MLP)
        st.dataframe(Results_MLP,width=1000,)
    elif model == "Neural Network":
        
        st.header("Results Neural Network")
        st.pyplot(matrix_ANN)
        st.dataframe(Results_ANN,width=1000,)
        
        
elif pages == "Conclusion":
    
    st.dataframe(Results,width=1000,)
    st.markdown("Looking at the results, we can assume that the MLP has the best score. We'll so use for the classification app.")
    
    
elif pages == "Classification App 🚀":
    
    col1, col2 = st.columns(2)
    
    Pregnancies = col1.slider("Select the number of pregnancies the women got", min_value=0, max_value=20, value=3, step=1)
    Glucose = col2.slider("Select the glucose concentration", min_value=44, max_value=200, value=120, step=1)
    BloodPressure = col1.slider("Select the BloodPressure", min_value=20, max_value=130, value=70, step=1)
    SkinThickNess = col2.slider("Select the triceps skin fold thickness", min_value=5, max_value=100, value=25, step=1)
    Insulin = col1.slider("Select the quantity of Serum Insulin", min_value=10, max_value=900, value=120, step=1)
    BMI = col2.slider("Select the BodyMass index", min_value=15, max_value=70, value=30, step=1)
    Diabete = col1.slider("Select the Diabete pedigree function", min_value=0.07, max_value=2.50, value=0.5, step=0.0001)
    Age = col2.slider("Select the Age", min_value=20, max_value=90, value=3, step=30)
    
    input_raw = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickNess,Insulin,BMI,Diabete,Age]])
    input = scaler.transform(input_raw)
    pred = MLP.predict(input)
    
    #create a boutton for prediction
    Predict = st.button("Predict")
    
    if Predict:
    #pred = model.predict(input_array)
        if pred < 0 or pred > 1:
            st.error("The input values must be irrelevant, try again by giving different values")
        pred = round(float(pred), 3)
        if pred == 0 :
            diagnostic = "The person does not have diabete"
            st.success(diagnostic)
        elif pred == 1 :
            diagnostic = "The person has diabete"
            st.error(diagnostic)
        st.balloons()
    
    



