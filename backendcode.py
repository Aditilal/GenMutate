
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # caring to care
from nltk.corpus import stopwords
from string import punctuation

# from scipy.misc import imresize (did not run)
from PIL import Image
import numpy as np


# # Word2Vec related libraries
# #from gensim.models import KeyedVectors
# from sklearn import model_selection

#Importing dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from joblib import dump,load

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
os.chdir('/Users/Aditi Lal/Project/Data')
source= '/Users/Aditi Lal/Project/Data'
train_variant = pd.read_csv('training_variants.txt')
test_variant = pd.read_csv('test_variants.txt')

print(test_variant.head())
print(train_variant.head())
print(train_variant.shape)
print(test_variant.shape)
train_m = train_variant.isnull().sum()
print(train_m)
test_m = test_variant.isnull().sum()
print(test_m)
train_data = train_variant.dropna(axis = 0, how = "any")
test_data = test_variant.dropna(axis = 0, how = "any")
print(train_data)
print(test_data)
train_data["Class"].unique()
train_data["Gene"].unique()
train_data["Gene"].value_counts()
train_data["Variation"].unique()
train_data["Variation"].value_counts()

#sns.countplot(x = train_data["Gene"], hue = train_data["Class"])
plt.show()
train_data.groupby(["Gene"])["Class"].value_counts()
train_data.groupby(["Variation"])["Class"].value_counts()
str_data = train_data.select_dtypes(include = ["object"])
str_dt = test_data.select_dtypes(include = ["object"])
int_data = train_data.select_dtypes(include = ["integer", "float"])
int_dt = test_data.select_dtypes(include = ["integer", "float"])

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
feature = str_data.apply(label.fit_transform)
feature = feature.join(int_data)
feature.head()

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Test = str_dt.apply(label.fit_transform)
Test = Test.join(int_dt)
print(Test.head())

from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder() 
data = onehotencoder.fit_transform(feature).toarray() 

!pip install imblearn

y_train = feature["Class"]
y_train.head()
x_train = feature.drop(["Class", "ID"] ,axis = 1)
print(x_train.head())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
print(x_train.head())
print(y_train.head())

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score

# Define the undersampling and oversampling methods
undersampler = RandomUnderSampler(sampling_strategy = {
    1: 770,   # the majority class, leave as is
    2: 770,   # oversample class 2 to match class 1
    3: 770,   # oversample class 3 to match class 1
    4: 770,   # oversample class 4 to match class 1
    5: 770,   # oversample class 5 to match class 1
    6: 770,   # oversample class 6 to match class 1
    7: 770,   # oversample class 7 to match class 1
    8: 770,   # oversample class 8 to match class 1
    9: 770    # oversample class 9 to match class 1
}, random_state=10)
oversampler = SMOTE(sampling_strategy = {
    1: 770,   # the majority class, leave as is
    2: 770,   # oversample class 2 to match class 1
    3: 770,   # oversample class 3 to match class 1
    4: 770,   # oversample class 4 to match class 1
    5: 770,   # oversample class 5 to match class 1
    6: 770,   # oversample class 6 to match class 1
    7: 770,   # oversample class 7 to match class 1
    8: 770,   # oversample class 8 to match class 1
    9: 770    # oversample class 9 to match class 1
}, random_state=10)

# Perform the undersampling and oversampling on the training data
x_train_undersample, y_train_undersample = undersampler.fit_resample(x_train_smote, y_train_smote)
x_train_oversample, y_train_oversample = oversampler.fit_resample(x_train_smote, y_train_smote)

# Train the classifiers using the balanced training data
classifier1 = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=10)
classifier1.fit(x_train_undersample, y_train_undersample)

classifier2 = KNeighborsClassifier(n_neighbors=5, weights='uniform')
classifier2.fit(x_train_oversample, y_train_oversample)

classifier3 = SVC(kernel='rbf', probability=True, random_state=10)
classifier3.fit(x_train_undersample, y_train_undersample)

classifier4 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
classifier4.fit(x_train_oversample, y_train_oversample)

# Evaluate the classifiers using the original test data
y_pred1 = classifier1.predict(x_test)
acc1 = accuracy_score(y_test, y_pred1)
prec1 = precision_score(y_test, y_pred1, average='macro')

y_pred2 = classifier2.predict(x_test)
acc2 = accuracy_score(y_test, y_pred2)
prec2 = precision_score(y_test, y_pred2, average='macro')

y_pred3 = classifier3.predict(x_test)
acc3 = accuracy_score(y_test, y_pred3)
prec3 = precision_score(y_test, y_pred3, average='macro')

y_pred4 = classifier4.predict(x_test)
acc4 = accuracy_score(y_test, y_pred4)
prec4 = precision_score(y_test, y_pred4, average='macro')

print("Accuracy scores:")
print("Random Forest Classifier (undersampling):", acc1)
print("K-Nearest Neighbors Classifier (oversampling):", acc2)
print("Support Vector Machine Classifier (undersampling):", acc3)
print("Logistic Regression Classifier (oversampling):", acc4)

print("\nPrecision scores:")
print("Random Forest Classifier (undersampling):", prec1)
print("K-Nearest Neighbors Classifier (oversampling):", prec2)
print("Support Vector Machine Classifier (undersampling):", prec3)
print("Logistic Regression Classifier (oversampling):", prec4)

ax = sns.countplot(x=train_data["Class"])

# Get the maximum count value
max_count = train_data["Class"].value_counts().max()

# Add count labels to each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x()+0.3, p.get_height()+50), ha='center', va='top', color='black', fontsize=12)

# Set y-axis limits to make sure all count labels are visible
plt.ylim(0, max_count+500)

# Set plot title and axis labels
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

# Show the plot
plt.show()
# 
from sklearn import metrics

a="TP53"
b="Truncating Mutations"
data = [[a, b]]
n1 = pd.DataFrame(data)
n2 = n1.select_dtypes(include=["object"])
n3 = n1.select_dtypes(include=["integer", "float"])
label = LabelEncoder()
n4 = n2.apply(label.fit_transform)
n4 = n4.join(n3)
print(n4.head())
new_input = n4.values
new_output = classifier1.predict(new_input)
print(new_output)
