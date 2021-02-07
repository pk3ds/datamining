from yellowbrick.cluster import silhouette_visualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import collections
import warnings
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st
# python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize': (11, 6)})


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
################################################

st.set_page_config(layout="centered")

st.title('Data Mining Project')
# Read Laundry Dataset
st.header("Read Laundry Dataset")
df = pd.read_csv('Laundry_Data.csv')
df = df.drop(columns=['No', 'Date', 'Time'], axis=1)
st.write(df)
st.write("The shape of this dataframe is: ", df.shape)

# Data Preprocessing
st.header("Data Preprocessing")
st.write('The number of missing values for each column in the dataframe')
st.write(df.isnull().sum())
st.write('The data type of each column or feature')
st.write(df.dtypes)

# Visualize missing values
st.header("Visualize missing values")
df.isnull().sum().plot(kind='bar')
st.pyplot()

# Data Cleaning and Transformation
st.header("Data Cleaning and Transformation")
df2 = df.copy(deep=True)
# Object Data Type
# Replacing NAN for Race
df2['Race'].fillna("Not_Available", inplace=True)

# Replacing NAN for Gender
df2['Gender'].fillna("Not_Available", inplace=True)

# Replacing NAN for Body_Size
df2['Body_Size'].fillna("Not_Available", inplace=True)

# Replacing NAN for With_Kids
df2['With_Kids'].fillna("Not_Available", inplace=True)

# Replacing NAN for Kids_Category
df2['Kids_Category'].fillna("Not_Available", inplace=True)

# Replacing NAN for Basket_Size
df2['Basket_Size'].fillna("Not_Available", inplace=True)

# Replacing NAN for Basket_colour
df2['Basket_colour'].fillna("Not_Available", inplace=True)

# Replacing NAN for Attire
df2['Attire'].fillna("Not_Available", inplace=True)

# Replacing NAN for Shirt_Colour
df2['Shirt_Colour'].fillna("Not_Available", inplace=True)

# Replacing NAN for shirt_type
df2['shirt_type'].fillna("Not_Available", inplace=True)

# Replacing NAN for Pants_Colour
df2['Pants_Colour'].fillna("Not_Available", inplace=True)

# Replacing NAN for pants_type
df2['pants_type'].fillna("Not_Available", inplace=True)

# Replacing NAN for Wash_Item
df2['Wash_Item'].fillna("Not_Available", inplace=True)

# Numerical Data Type
# Replacing NAN for Age_Range
df2['Age_Range'].fillna(df['Age_Range'].mean(), inplace=True)

# transform race
df2['Race'] = np.where(df2['Race'] == 'foreigner ', 'foreigner', df2['Race'])

# edit Kids_Category
df2['Kids_Category'] = np.where(
    df2['Kids_Category'] == 'toddler ', 'toddler', df2['Kids_Category'])

# edit Shirt_Colour
df2['Shirt_Colour'] = np.where(
    df2['Shirt_Colour'] == 'black ', 'black', df2['Shirt_Colour'])

# edit Pants_Colour
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'blue  ', 'blue', df2['Pants_Colour'])
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'blue ', 'blue', df2['Pants_Colour'])
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'black ', 'black', df2['Pants_Colour'])

df2['Pants_Colour'] = 'pants_colour_' + df2['Pants_Colour'].str[:]
df2['Shirt_Colour'] = 'shirt_colour_' + df2['Shirt_Colour'].str[:]
df2['Basket_colour'] = 'basket_colour_' + df2['Basket_colour'].str[:]

# pandas dummies
cat_vars = ['Race', 'Gender', 'Body_Size', 'With_Kids', 'Kids_Category', 'Basket_Size',
            'Basket_colour', 'Attire', 'Shirt_Colour', 'shirt_type', 'Pants_Colour',
            'pants_type', 'Wash_Item']
df3 = df2.copy(deep=True)

for var in cat_vars:
    cat_list = 'var_' + var
    cat_list = pd.get_dummies(df3[var], prefix=var)
    df4 = df3.join(cat_list)
    df3 = df4

df_vars = df3.columns.values.tolist()
to_keep = [i for i in df_vars if i not in cat_vars]

df_final = df3[to_keep]
# df_final.columns.values
st.write('This dataset has been undergone different preprocessing techniques. Since there are many missing values in different columns, therefore we have fill up the features with object data type with a new constant value which is Not_Available. ')
st.write('Meanwhile the missing values in Age_Range which is an integer data type column is filled with the mean value from the column.')
st.write('After the missing values has been filled up accordingly, the categorical features were transform using pandas function which is get dummies to transform the string value into numerical value')
st.write('The target column value which is Spectacle column was also been transform to numerical value by assigning Yes value to 0 and No value to 1')
specs_dict = {'no': 1, 'yes': 0}
df_final['Spectacles'] = df_final.Spectacles.map(specs_dict)
st.write(df_final)

# Association Rule Mining
st.header("Association Rule Mining")
arm_df = df2[['Attire', 'Shirt_Colour', 'shirt_type',
              'Pants_Colour', 'pants_type']].copy()
st.write("We would like to apply Association Rule Mining to find the correlation between the attire of each customer.")
st.write("In order to visualize better we have added prefix in front of shirt and pants colour to distinguish them better.")
st.write(arm_df)
st.write("The DataFrame use for Association Rule Mining shape: ", arm_df.shape)

records = []
for i in range(0, 807):
    records.append([str(arm_df.values[i, j]) for j in range(0, 5)])

association_rules = apriori(
    records, min_support=0.01, min_confidence=0.7, min_lift=3, min_length=2)
association_results = list(association_rules)

st.write("We have adjusted the parameter value for the Apriori which is :")
st.write("Minimum Support: 0.0.1")
st.write("Minimum Confidence: 0.7")
st.write("Minimum Lift: 3")
st.write("Minimum Length = 2")
st.write("From the parameter values stated above, we have produced output of 15 rules which can be seen below.")

cnt = 0
st.write("=====================================")
# for item in association_results:
#     cnt += 1
#     # first index of the inner list
#     # Contains base item and add item
#     pair = item[0]
#     items = [x for x in pair]
#     st.write("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])

#     # second index of the inner list
#     st.write("Support: " + str(round(item[1], 3)))

#     # third index of the list located at 0th
#     # of the third index of the inner list

#     st.write("Confidence: " + str(round(item[2][0][2], 4)))
#     st.write("Lift: " + str(round(item[2][0][3], 4)))
#     st.write("=====================================")

option = st.selectbox('From the parameter values stated above, we have produced output of 15 rules which can be seen below.',
                      ('1', '2', '3', '4', '5',
                       '6', '7', '8', '9', '10',
                       '11', '12', '13', '14', '15'))

item = association_results[int(option)]
pair = item[0]
items = [x for x in pair]
st.write("(Rule " + str(option) + ") " + items[0] + " -> " + items[1])
st.write("Support: " + str(round(item[1], 3)))
st.write("Confidence: " + str(round(item[2][0][2], 4)))
st.write("Lift: " + str(round(item[2][0][3], 4)))

# Dataset Visualization

st.header("Dataset Visualization")

st.write("Below the barchart for number of people wears spectacles based on")
option = st.selectbox('',
                      ('Race', 'Gender', 'Attire', 'Wash_Item'))

if option == "Race":
    # Race vs decision
    a = pd.crosstab(df.Race, df.Spectacles).plot(kind='bar')
    plt.title('Number of People Wears Spectacles based on Race')
    plt.xlabel('Race')
    plt.ylabel('Frequency')

    st.write("Below the barchart for number of people wears spectacles based on race.")
    st.text("Race vs decision")
    st.pyplot()
elif option == "Gender":
    a = pd.crosstab(df.Gender, df.Spectacles).plot(kind='bar')
    plt.title('Number of People Wears Spectacles based on Gender')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')

    st.write(
        "Below the barchart for number of people wears spectacles based on gender.")
    st.text("Gender vs decision")
    st.pyplot()
elif option == "Attire":
    a = pd.crosstab(df.Attire, df.Spectacles).plot(kind='bar')
    plt.title('Number of People Wears Spectacles based on Attire')
    plt.xlabel('Attire')
    plt.ylabel('Frequency')

    st.write(
        "Below the barchart for number of people wears spectacles based on attire.")
    st.text("Attire vs decision")
    st.pyplot()
else:
    a = pd.crosstab(df.Wash_Item, df.Spectacles).plot(kind='bar')
    plt.title('Number of People Wears Spectacles based on Wash_Item')
    plt.xlabel('Wash_Item')
    plt.ylabel('Frequency')

    st.write(
        "Below the barchart for number of people wears spectacles based on Wash_Item.")
    st.text("Wash_Item vs decision")
    st.pyplot()

st.text("Boxplot for Race against Age Range")
st.text("The boxplot shows the median for the age for all the races are between 40 and 45 years old.\nThere are no outliers in the age range for each race.\nIt can be concluded that the data are normally distributed among all the races.")
sns.catplot(x="Race", y="Age_Range", kind="box", data=df, aspect=1.5)
st.pyplot()

st.text("Density Plot for Age Range")
st.text("While the density plot shows the histogram distribution of the age range\nof the customer with the density plot.")
sns.distplot(df["Age_Range"], bins=10)
st.pyplot()
st.text("\nAlthough the graph has negative value for skewness which -0.06,\nbut we can assume that the graph is almost normally distributed.")
st.write("Skewness: ", df["Age_Range"].skew())

b = sns.countplot(x='Spectacles', data=df)

for p in b.patches:
    b.annotate("%.0f" % p.get_height(), (p.get_x() +
                                         p.get_width() / 2., p.get_height()),
               ha='center', va='center', rotation=0,
               xytext=(0, 18), textcoords='offset points')
st.text("Bar chart below shows people who wears and does not wear spectacles goes to the laundry.")
st.pyplot()


st.header("Oversampling using SMOTE")
X = df_final.drop('Spectacles', axis=1)
y = df_final["Spectacles"]

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

st.text("After we have done with pre-processed the dataset, we have found the dataset used in this\n" +
        "project is an imbalanced dataset. This can cause the machine learning model used to be\n" +
        "bias and ignoring the minority class entirely in the training and predicting the class label.\n" +
        "Therefore we have applied oversampling technique called SMOTE onto our dataset. First,\n" +
        "The dataset was split into training and test dataset and from there, the training dataset\n" +
        "was fitted into the SMOTE to synthesizes new examples for the minority class. Initially, the\n" +
        "minority class which is the 0 or yes has only 105 samples and then this calss was oversampled\n" +
        "to match the count for majority class. In the end, both class labels have 459 samples\n" +
        "in the training dataset. Towards the end, we will evaluate and compare whether there is\n" +
        "improvement or not in terms of performance for each classifier used between oversampling dataset\n" +
        "and non-oversampling dataset. Figure below shows the distribution of class label\n" +
        "after applying SMOTE.")

st.write("Length of oversampled data : ", len(os_data_X))

st.write("Number of People not wear spectacles in oversampled data : ",
         len(os_data_y[os_data_y['y'] == 1]))
st.write("Number of People wear spectacles in oversampled data : ",
         len(os_data_y[os_data_y['y'] == 0]))
st.write("Proportion of People not wear spectacles in oversample data : ", len(
    os_data_y[os_data_y['y'] == 1]) / len(os_data_X))
st.write("Proportion of People wear spectacles in oversample data : ",
         len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))

# after oversampling with SMOTE
b = sns.countplot(x='y', data=os_data_y)

for p in b.patches:
    b.annotate("%.0f" % p.get_height(), (p.get_x() +
                                         p.get_width() / 2., p.get_height()),
               ha='center', va='center', rotation=0,
               xytext=(0, 18), textcoords='offset points')

st.pyplot()

# counter

st.write(str(collections.Counter(y_train)))

# classification

X = df_final.drop('Spectacles', axis=1)
y = df_final["Spectacles"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=0)

st.title("Oversampling Dataset")
st.header("Logistic Regression")
st.text("We have assign our Logistic Regression model into the Recursive Feature Elimination with Cross-Validation(RFECV)\n" +
        "parameter including with the parameter.")
# Create RF classifier
modelLR = LogisticRegression(C=0.1, penalty='l2', solver='newton-cg')

# Train with oversampling data
modelLR.fit(os_data_X, os_data_y)

# Predict the result
y_pred = modelLR.predict(X_test)

# Model Accuracy
st.write()
st.write("Accuracy on training set: {:.3f}".format(
    modelLR.score(X_train, y_train)))
st.write("Accuracy on test set: {:.3f}".format(modelLR.score(X_test, y_test)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
st.write()
st.header("Classification Report")
st.write()
st.write(metrics.classification_report(y_test, y_pred))

st.header("Confusion Matrix")
st.write()

cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Logisic Regression Classifier Oversampling")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

st.header("AUC Score and ROC Curve")
st.write()
# Calculate AUC
prob_LR_os = modelLR.predict_proba(X_test)
prob_LR_os = prob_LR_os[:, 1]

auc_LR_os = roc_auc_score(y_test, prob_LR_os)
st.write('AUC: %.2f' % auc_LR_os)

# Plot ROC Curve

fpr_LR_os, tpr_LR_os, thresholds_LR_os = roc_curve(y_test, prob_LR_os)

plt.plot(fpr_LR_os, tpr_LR_os, color='blue',
         label='Logistic Regression Oversampling (area = %0.2f)' % auc_LR_os)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.header("Decision Tree Classifier")
st.write()
# Create DT classifier
modelDT = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, random_state=0)

# Train with oversampling data
modelDT.fit(os_data_X, os_data_y)

# Predict the result
y_pred = modelDT.predict(X_test)

# Model Accuracy
st.write("Accuracy on training set: {:.3f}".format(
    modelDT.score(X_train, y_train)))
st.write("Accuracy on test set: {:.3f}".format(modelDT.score(X_test, y_test)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

st.header("Classification Report")
st.write()
st.write(metrics.classification_report(y_test, y_pred))

# Confusion Matrix
st.header("Confusion Matrix")
st.write()
cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Decision Tree Classifier Oversampling")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

# AUC Score and ROC Curve
st.header("AUC Score and ROC Curve")
st.write()
# Calculate AUC
prob_DT_os = modelDT.predict_proba(X_test)
prob_DT_os = prob_DT_os[:, 1]

auc_DT_os = roc_auc_score(y_test, prob_DT_os)
st.write('AUC: %.2f' % auc_DT_os)

# Plot ROC Curve

fpr_DT_os, tpr_DT_os, thresholds_DT_os = roc_curve(y_test, prob_DT_os)

plt.plot(fpr_DT_os, tpr_DT_os, color='orange',
         label='Decision Tree Oversampling (area = %0.2f)' % auc_DT_os)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

# Preprocessed Dataset
st.title("Preprocessed Dataset")
st.header("Logistic Regression Classifier")
st.write()
# Train the model using training data
modelLR.fit(X_train, y_train)

# Predict the result
y_pred = modelLR.predict(X_test)

# Model Accuracy
st.write("Accuracy on training set: {:.3f}".format(
    modelLR.score(X_train, y_train)))
st.write("Accuracy on test set: {:.3f}".format(modelLR.score(X_test, y_test)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

st.header("Classification Report")

st.write(metrics.classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Logistic Regression Classifier Pre-processed Data")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

st.header("AUC Score and ROC Curve")
# Calculate AUC
prob_LR = modelLR.predict_proba(X_test)
prob_LR = prob_LR[:, 1]

auc_LR = roc_auc_score(y_test, prob_LR)
st.write('AUC: %.2f' % auc_LR)

st.write()
# Plot ROC Curve

fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, prob_LR)

plt.plot(fpr_LR, tpr_LR, color='blue',
         label='Logistic Regression (area = %0.2f)' % auc_LR)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.header("Decision Tree Classfier")
# Train the model using training data
modelDT = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, random_state=10)
modelDT.fit(X_train, y_train)

# Predict the result
y_pred = modelDT.predict(X_test)

# Model Accuracy
st.write("Accuracy on training set: {:.3f}".format(
    modelDT.score(X_train, y_train)))
st.write("Accuracy on test set: {:.3f}".format(modelDT.score(X_test, y_test)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

st.header("Classification Report")
st.write(metrics.classification_report(y_test, y_pred))

st.header("Confusion Matrix")

cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Decision Tree Classifier Pre-processed Data")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

st.header("AUC Score and ROC Curve")
# Calculate AUC
prob_DT = modelDT.predict_proba(X_test)
prob_DT = prob_DT[:, 1]

auc_DT = roc_auc_score(y_test, prob_DT)
st.write('AUC: %.2f' % auc_DT)
# Plot ROC Curve

fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT)

plt.plot(fpr_DT, tpr_DT, color='orange',
         label='Decision Tree (area = %0.2f)' % auc_DT)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.title("Feature Selection")


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


st.header("Logistic Regression")
modelLR.fit(X, y)
rfe = RFECV(modelLR, min_features_to_select=1,
            cv=StratifiedKFold(10), scoring='accuracy')
rfe.fit(X, y)

st.write('Optimal number of features: {}'.format(rfe.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfe.grid_scores_) + 1),
         rfe.grid_scores_, color='#303F9F', linewidth=3)

plt.show()
st.pyplot()

rfe_score = ranking(list(map(float, rfe.ranking_)), columns, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()),
                         columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending=False)

rfe_score = rfe_score.loc[rfe_score['Score'] == 1]

new_columns = rfe_score['Features'].tolist()

X_LR = X[new_columns]
# display(X_LR.head())
st.write(X_LR)

st.header("Train the model with feature selection")
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(
    X_LR, y.values.ravel(), test_size=0.3, random_state=0)
# Train the model using training data
modelLR.fit(X_train_LR, y_train_LR)

# Predict the result
y_pred_LR = modelLR.predict(X_test_LR)

# Model Accuracy
st.write("Accuracy on training set: {:.3f}".format(
    modelLR.score(X_train_LR, y_train_LR)))
st.write("Accuracy on test set: {:.3f}".format(
    modelLR.score(X_test_LR, y_test_LR)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test_LR, y_pred_LR)))
st.write('Recall= {:.2f}'. format(recall_score(y_test_LR, y_pred_LR)))
st.write('F1= {:.2f}'. format(f1_score(y_test_LR, y_pred_LR)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test_LR, y_pred_LR)))


st.write(metrics.classification_report(y_test_LR, y_pred_LR))


cf_matrix = confusion_matrix(y_test_LR, y_pred_LR)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Logistic Regression Classifier Feature Selection")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

# Calculate AUC
prob_LR_FS = modelLR.predict_proba(X_test_LR)
prob_LR_FS = prob_LR_FS[:, 1]

auc_LR_FS = roc_auc_score(y_test_LR, prob_LR_FS)
st.write('AUC: %.2f' % auc_LR_FS)

# Plot ROC Curve
fpr_LR_FS, tpr_LR_FS, thresholds_LR_FS = roc_curve(y_test_LR, prob_LR_FS)

plt.plot(fpr_LR_FS, tpr_LR_FS, color='blue',
         label='Logistic Regression FS (area = %0.2f)' % auc_LR_FS)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.header("Decision Tree")

modelDT.fit(X, y)
rfe = RFECV(modelDT, min_features_to_select=10,
            cv=StratifiedKFold(10), scoring='accuracy')
rfe.fit(X, y)
st.write('Optimal number of features: {}'.format(rfe.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfe.grid_scores_) + 1),
         rfe.grid_scores_, color='#303F9F', linewidth=3)

plt.show()
st.pyplot()

rfe_score = ranking(list(map(float, rfe.ranking_)), columns, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()),
                         columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending=False)

rfe_score = rfe_score.loc[rfe_score['Score'] == 1]

new_columns = rfe_score['Features'].tolist()

X_DT = X[new_columns]
# display(X_DT.head())
st.write(X_DT)

st.header("Train the model with feature selection")
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(
    X_DT, y.values.ravel(), test_size=0.3, random_state=0)
# Train the model using training data
modelDT.fit(X_train_DT, y_train_DT)

# Predict the result
y_pred_DT = modelDT.predict(X_test_DT)

# Model Accuracy
st.write("Accuracy on training set: {:.3f}".format(
    modelDT.score(X_train_DT, y_train_DT)))
st.write("Accuracy on test set: {:.3f}".format(
    modelDT.score(X_test_DT, y_test_DT)))
st.write()
st.write('Precision= {:.2f}'.format(precision_score(y_test_DT, y_pred_DT)))
st.write('Recall= {:.2f}'. format(recall_score(y_test_DT, y_pred_DT)))
st.write('F1= {:.2f}'. format(f1_score(y_test_DT, y_pred_DT)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test_DT, y_pred_DT)))

st.write(metrics.classification_report(y_test_DT, y_pred_DT))

st.header("Confusion Matrix")


cf_matrix = confusion_matrix(y_test_DT, y_pred_DT)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix Decision Tree Classifier Feature Selection")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot()

# Calculate AUC
prob_DT_FS = modelDT.predict_proba(X_test_DT)
prob_DT_FS = prob_DT_FS[:, 1]

auc_DT_FS = roc_auc_score(y_test_DT, prob_DT_FS)
st.write('AUC: %.2f' % auc_DT_FS)

# Plot ROC Curve
fpr_DT_FS, tpr_DT_FS, thresholds_DT_FS = roc_curve(y_test_DT, prob_DT_FS)

plt.plot(fpr_DT_FS, tpr_DT_FS, color='orange',
         label='Decision Tree FS (area = %0.2f)' % auc_DT_FS)
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.title("Comparing Results From Each Classifiers")
st.header("Performance Comparison")
st.header("Linear Regression")
plt.plot(fpr_LR_os, tpr_LR_os, color='blue',
         label='Logistic Regression Oversampling (area = %0.2f)' % auc_LR_os)
plt.plot(fpr_LR, tpr_LR, color='red',
         label='Logistic Regression Pre-processed (area = %0.2f)' % auc_LR)
plt.plot(fpr_LR_FS, tpr_LR_FS, color='orange',
         label='Logistic Regression FS (area = %0.2f)' % auc_LR_FS)

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.header("Decision Tree")
plt.plot(fpr_DT_os, tpr_DT_os, color='blue',
         label='Decision Tree Oversampling (area = %0.2f)' % auc_DT_os)
plt.plot(fpr_DT, tpr_DT, color='red',
         label='Decision Tree (area = %0.2f)' % auc_DT)
plt.plot(fpr_DT_FS, tpr_DT_FS, color='orange',
         label='Decision Tree FS (area = %0.2f)' % auc_DT_FS)

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

st.title("K-Means Clustering")
st.header("Applying the k-Means approach")
st.header("Identify the Optimal k value")

distortions = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300,
                tol=1e-04, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
st.pyplot()
# the elbow is better
st.header("Create a copy of df and merge the newly predicted labels back")
km = KMeans(n_clusters=2, random_state=1)
km.fit(X)
label = km.predict(X)
df2['label'] = label
df_new = df2.copy()
df_new = df_new.drop("Spectacles", axis=1)
df_new['Spectacles'] = km.labels_
st.write(df_new)

st.header("Create plots to visualise the difference")
fig, ax = plt.subplots(1, 2, figsize=(13, 7))
sns.stripplot(x="Washer_No", y="Age_Range",
              hue="Spectacles", data=df_final, ax=ax[0])
sns.stripplot(x="Washer_No", y="Age_Range",
              hue="Spectacles", data=df_new, ax=ax[1])
st.pyplot()

fig, ax = plt.subplots(1, 2, figsize=(13, 7))
sns.stripplot(x="Race", y="Age_Range", hue="Spectacles", data=df2, ax=ax[0])
sns.stripplot(x="Race", y="Age_Range", hue="Spectacles", data=df_new, ax=ax[1])
st.pyplot()

st.header("Retrieve the Silhouette Score for when K=2")
silhouette_visualizer(KMeans(2, random_state=12), X, colors='yellowbrick')
st.pyplot()
