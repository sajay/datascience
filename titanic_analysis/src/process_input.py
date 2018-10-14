#Combining 
#https://www.kaggle.com/bemmerdinger/titanic-data-science-solutions AND
#https://blog.patricktriest.com/titanic-machine-learning-in-python/

#No ML code

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

# preview the data
train_df.head()
train_df.tail()


train_df.info()
print('_'*40)
test_df.info()

#get the means

print("Survived Mean:")
print(train_df['Survived'].mean())
train_df.groupby('Pclass').mean()
class_sex_grouping = train_df.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()

