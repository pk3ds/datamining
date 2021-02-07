import warnings
import streamlit as st
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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

df = pd.read_csv('Laundry_Data.csv')

df = df.drop(columns=['No', 'Date', 'Time'], axis=1)

df_na = df.copy(deep=True)
df_na.dropna(inplace=True)

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

st.write(df2.isnull().sum())

df2["Race"].unique()

# edit race
df2['Race'] = np.where(df2['Race'] == 'foreigner ', 'foreigner', df2['Race'])

df2["Race"].unique()

df2["Kids_Category"].unique()

# edit Kids_Category
df2['Kids_Category'] = np.where(
    df2['Kids_Category'] == 'toddler ', 'toddler', df2['Kids_Category'])

df2["Kids_Category"].unique()

df2["Shirt_Colour"].unique()
# edit Shirt_Colour
df2['Shirt_Colour'] = np.where(
    df2['Shirt_Colour'] == 'black ', 'black', df2['Shirt_Colour'])
df2["Shirt_Colour"].unique()
df2["Pants_Colour"].unique()
# edit Pants_Colour
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'blue  ', 'blue', df2['Pants_Colour'])
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'blue ', 'blue', df2['Pants_Colour'])
df2['Pants_Colour'] = np.where(
    df2['Pants_Colour'] == 'black ', 'black', df2['Pants_Colour'])
df2["Pants_Colour"].unique()
df = df2.copy(deep=True)
df2['Pants_Colour'] = 'pants_colour_' + df2['Pants_Colour'].str[:]
df2["Pants_Colour"].unique()
df2['Shirt_Colour'] = 'shirt_colour_' + df2['Shirt_Colour'].str[:]
df2["Shirt_Colour"].unique()
df2['shirt_type'].unique()
df2['shirt_type'] = np.where(
    df2['shirt_type'] == 'long sleeve', 'long_sleeve', df2['shirt_type'])
df2['shirt_type'] = df2['shirt_type'].str[:] + '_shirt'
df2['pants_type'].unique()
df2['pants_type'] = df2['pants_type'].str[:] + '_pants'
df2["pants_type"].unique()
cat_vars = ['Race', 'Gender', 'Body_Size', 'With_Kids', 'Kids_Category', 'Basket_Size',
            'Basket_colour', 'Attire', 'Shirt_Colour', 'shirt_type', 'Pants_Colour',
            'pants_type', 'Wash_Item']
df3 = df.copy(deep=True)

for var in cat_vars:
    cat_list = 'var_' + var
    cat_list = pd.get_dummies(df3[var], prefix=var)
    df4 = df3.join(cat_list)
    df3 = df4

df_vars = df3.columns.values.tolist()
to_keep = [i for i in df_vars if i not in cat_vars]
df_final = df3[to_keep]
st.write(df_final.isnull().sum())
specs_dict = {'no': 1, 'yes': 0}
df_final['Spectacles'] = df_final.Spectacles.map(specs_dict)
df_final.sample(10)
arm_df = df2[['Attire', 'Shirt_Colour', 'shirt_type',
              'Pants_Colour', 'pants_type']].copy()

records = []
for i in range(0, 807):
    records.append([str(arm_df.values[i, j]) for j in range(0, 5)])
association_rules = apriori(
    records, min_support=0.01, min_confidence=0.7, min_lift=3, min_length=2)
association_results = list(association_rules)
cnt = 0

for item in association_results:
    cnt += 1
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]

a = pd.crosstab(df.Race, df.Spectacles).plot(kind='bar')
#a.div(a.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Number of People Wears Spectacles based on Race')
plt.xlabel('Race')
plt.ylabel('Frequency')

a = pd.crosstab(df.Gender, df.Spectacles).plot(kind='bar')
#a.div(a.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Number of People Wears Spectacles based on Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')

a = pd.crosstab(df.Attire, df.Spectacles).plot(kind='bar')
#a.div(a.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Number of People Wears Spectacles based on Attire')
plt.xlabel('Attire')
plt.ylabel('Frequency')

a = pd.crosstab(df.Wash_Item, df.Spectacles).plot(kind='bar')
#a.div(a.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Number of People Wears Spectacles based on Wash_Item')
plt.xlabel('Wash_Item')
plt.ylabel('Frequency')

sns.catplot(x="Race", y="Age_Range", kind="box", data=df, aspect=1.5)

sns.distplot(df["Age_Range"], bins=10)

b = sns.countplot(x='Spectacles', data=df)

for p in b.patches:
    b.annotate("%.0f" % p.get_height(), (p.get_x() +
                                         p.get_width() / 2., p.get_height()),
               ha='center', va='center', rotation=0,
               xytext=(0, 18), textcoords='offset points')

X = df_final.drop('Spectacles', axis=1)
y = df_final["Spectacles"]

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

st.write("Length of oversampled data : ", len(os_data_X))
st.write("Number of People not wear spectacles in oversampled data : ",
         len(os_data_y[os_data_y['y'] == 1]))
st.write("Number of People wear spectacles in oversampled data : ",
         len(os_data_y[os_data_y['y'] == 0]))
st.write("Proportion of People not wear spectacles in oversample data : ", len(
    os_data_y[os_data_y['y'] == 1]) / len(os_data_X))
st.write("Proportion of People wear spectacles in oversample data : ",
         len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))
