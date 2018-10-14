#########################################################################################
##                   		                                                           ##  
##                   		                                                           ##  
##                     Hands On Machine Learning with Python                           ##
##                   			                                                       ##  
##                   		                                                           ## 
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		                                                           ##  
##                   		           John Anderson                                   ##   
#########################################################################################




# Everything after the hashtag in this line is a comment.
# This is to keep your sanity.

#########################################################################################
#########################          Python Crash Course          ######################### 
#########################################################################################

#Strings, Lists, Functions, and More
#We can assign values to variables. Here’s an example:

yourName = ‘Felicity’
print(yourName)
Returns ‘Felicity’
print(yourName[0])
#This prints ‘F’, the first letter of ‘Felicity.’ 

print(yourName[0:3])
#Result would be ‘Fel’. 

print(len(yourName))
#Counts the number of letters or finds out the “length” of the string. So the result here would be 8.

yourAge = 22
print(yourAge)  #This is a comment. Anyway, this prints ‘22’
print(yourAge * 2)   #Prints 22*2 which is 44
yourAge = 25   #We can also reassign values to variables. 

#Let’s now talk about Lists one of the most useful and popular data structures in Python:
myList = [1,2,3,4]
myList.append(5)   #myList then becomes [1,2,3,4,5]
myList[0]   #this returns the first value in the list, which is 1. Remember indexing in Python starts at zero.
print(len(myList))   #Prints the “length” of the list, which in this case 5.

#Let’s get into Functions and Flow Control (If, Elif, and Else statements):
def add_numbers(a,b):
	return a + b
print(add_numbers(3,4))
#First we define a function and include an instruction. 
#Then we call or test it but this time include the numbers.
#The function add_numbers will add 3 and 4, and print 7.
def if_even(num):
	if num % 2 == 0:
		print(“Number is even.”)
	else:
		print(“It’s odd.”)
if_even(24)   #This returns “Number is even.”
if_even(25)   #This returns “It’s odd.”


#Numpy, Pandas, Scipy , Matplotlib, Scikit-learn
#Here are examples of how to import them:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Numerical Python (Numpy)
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])   # create a rank 2 array
print(type(a))
print(a.shape)

b = np.random.random((2,2))  # create an array filled with random values
print(b)
print(b.shape)

#Let us look at an example of matrix product using Numpy.
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

# matrix product
print(np.dot(x, y))


#Pandas
#The code below shows how to create a Series object in Pandas.
import pandas as pd

s = pd.Series([1,3,5,np.nan,6,8])

print(s)

#To create a dataframe, we can run the following code.

df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))

print(df)


#Scientific Python (Scipy)

#Here is a simple usage of scipy that finds the inverse of a matrix.
from scipy import linalg
z = np.array([[1,2],[3,4]])

print(linalg.inv(z))

 
#Matplotlib

#Here is an example that uses Matplotlib to plot a sine waveform.

# magic command for Jupyter notebooks
%matplotlib inline
import matplotlib.pyplot as plt

# compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# plot the points using matplotlib
plt.plot(x, y)
plt.show()  # Show plot by calling plt.show()

 
#Scikit-Learn

# sample decision tree classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#########################################################################################
###################        Your First Machine Learning Project      ##################### 
#########################################################################################

#Source: https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook

#To access the data, we start by using pandas:
import pandas as pd
#Next step is to access the data and get it ready for later processing and analyzing:
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

#Once our data’s ready, let’s take a peek:
train_df.head(10)
 
print(train_df.columns.values)
 
#We should correct them first before the machine learning proper:
train_df.info()
print('_'*40)
test_df.info()

#Good news is we can actually test those assumptions and see early if they make sense:
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
#Notice that Pclass = 1 (upper class passengers) had the highest survival rate. Let’s look at Gender next:
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
#What about passengers with Siblings and/or Spouses with them:
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
#Let’s also look at the survival rate of passengers with Parents and/or Children with them:
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
#let’s visualize the data (this could be much easier to interpret). First, let’s import the necessary libraries and packages:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline   #so that plots will show in our notebook inline

#Then, we set up the visuals (starting with Age):
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
 
#Let’s also visualize Pclass (Passenger class) and whether they survived (1) or not (0):
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
 
#Let’s explore further. If a passenger embarked in a different location, does it affect his/her chances? It’s quite tricky but let’s explore anyway:
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
 
Embarked (S = Southampton; C = Cherbourg; Q = Queenstown)

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
 
#The next step then is to exclude the irrelevant features and include the relevant ones.
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
 

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
 
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
((891, 9), (418, 9))
train_df.head()
 

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
 
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
 
import numpy as np
guess_ages = np.zeros((2,3))
guess_ages
 
#We can do that through the following code
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
 
#Now we’ve taken care of missing values. Next is we create Age bands (range of age) and put them side by side with Survived:
 
#For convenience and neatness, we can then transform those Age Bands into ordinals:
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
 

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
 
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
#Let’s just discard Parch, SibSp, and FamilySize and focus on the isAlone:
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
 

#Let’s move forward.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
 
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
 
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
 
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
 
#Then we create a FareBand for simplicity:
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
 
#We then convert it into ordinal values (similar to what we did in AgeBand).
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
 
#That’s the test set. Let’s also take a peek at the test dataset:
test_df.head(10)
 

#First, we define the independent variables (X_train) and the target (Y_train) in our training set (we similarly do this in the test set):
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


#Let’s then train our model using the first option (Logistic Regression):
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#Let’s try another model (Random Forest Classifier):
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


#########################################################################################
##############                            Regression               ###################### 
######################################################################################### 

#The dataset can be downloaded from this URL https://forge.scilab.org/index.php/p/rdataset/source/file/master/csv/MASS/Boston.csv
 
#First we import relevant libraries and load the dataset using Pandas.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib magic command for Jupyter notebook
%matplotlib inline

dataset = pd.read_csv('Boston.csv')
dataset.head()


#Let us plot the relationship between one of the predictors and the price of a house 
plt.scatter(dataset['crim'], dataset['medv'])
plt.xlabel('Per capita crime rate by town')
plt.ylabel('Price')
plt.title("Prices vs Crime rate")

#Next we split our dataset into predictors and targets. Then we create a training and test set.

X = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Having fit the classifier, we can use it to predict house prices using features in the test set.

y_pred = regressor.predict(x_test)

#The next step is to evaluate the classifier using metrics such as the mean square error and the coefficient of determination  R square.

from sklearn.metrics import mean_squared_error, r2_score

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test, y_pred)))


#Finally, we can plot the predicted prices from the model against the ground truth (actual prices).

plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

 

#The scatter plot above shows a positive relationship between the predicted prices and actual prices. 
#Here is the code in its entirety.

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# load dataset
dataset = pd.read_csv('Boston.csv')
dataset.head()

# plot crime vs price
plt.scatter(dataset['crim'], dataset['medv'])
plt.xlabel('Per capita crime rate by town')
plt.ylabel('Price')
plt.title("Prices vs Crime rate")

# separate predictors and targets
X = dataset.drop(['Unnamed: 0', 'medv'], axis=1)
y = dataset['medv']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test, y_pred)))

# plot predicted prices vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

#Logistic Regression

#The dataset can be downloaded at: https://www.kaggle.com/uciml/pima-indians-diabetes-database/data

#Let us import relevant libraries and load the dataset to have a sense of what it contains.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')
dataset.head(5)

#Next we separate the columns in the dataset into features and labels. 
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

# Training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(features_train, labels_train)

The trained model can now be evaluated on the test set.

pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))

#Generalized Linear Models
 
#First we import the Statsmodels package as shown below.
import statsmodels.api as sm

#Next we load the dataset and extract the explanatory variable (X).
data = sm.datasets.scotland.load()
# data.exog is the independent variable X
data.exog = sm.add_constant(data.exog)

# we import the appropriate model and instantiate an object from it. 
# Instantiate a poisson family model with the default link function.
poisson_model = sm.GLM(data.endog, data.exog, family=sm.families.Poisson())

#We then fit the model on the data.
poisson_results = poisson_model.fit()

#We can now print a summary of results to better understand the trained model.
print(poisson_results.summary())


###########################   Application 2   #################################
#The dataset can be downloaded at https://www.kaggle.com/uciml/sms-spam-collection-dataset/downloads/spam.csv/1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comment the magic command below if not running in Jupyter notebook

#Next we load the dataset using Pandas and display the first 5 rows.
data = pd.read_csv('spam.csv', encoding='latin-1')
data.head(5)

#Let us plot a bar chart to visualize the distribution of legitimate and spam messages.

count_class = pd.value_counts(data['v1'], sort= True)
count_class.plot(kind='bar', color=[['blue', 'red']])
plt.title('Bar chart')
plt.show()

 
#We have to vectorize them to create new features. 
from sklearn.feature_extraction.text import CountVectorizer

f = CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
print(np.shape(X))


#Next we map our target variables into categories and split the dataset into train and test sets.
from sklearn.model_selection import train_test_split

data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = train_test_split(X, data['v1'], test_size=0.25, random_state=42)

#The next step involves initializing the Naive Bayes model and training it on the data.
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

#Finally, we gauge the model performance on the test set.

score = clf.score(X_test, y_test)
print('Accuracy: {}'.format(score))


#########################################################################################
##############                      K-Nearest Neighbors            ###################### 
######################################################################################### 

# The dataset can be downloaded from Kaggle https://www.kaggle.com/saurabh00007/iriscsv/downloads/Iris.csv/1.


#To begin let’s import all relevant libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

#Next we use Pandas to load the dataset which is contained in a CSV file 
dataset = pd.read_csv('Iris.csv')
dataset.head(5)

#In line with our observations, we separate the columns into features (X) and targets (y).

X = dataset.iloc[:, 1:5].values # select features ignoring non-informative column Id
y = dataset.iloc[:, 5].values # Species contains targets for our model

#To do this we leverage Scikit-Learn label encoder.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) # transform species names into categorical values

#Next we split our dataset into a training set and a test set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#We can implement L2 distance in Python using Numpy as shown below.

def euclidean_distance(training_set, test_instance):
    # number of samples inside training set
    n_samples = training_set.shape[0]
    
    # create array for distances
    distances = np.empty(n_samples, dtype=np.float64)
    
    # euclidean distance calculation
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum(np.square(test_instance - training_set[i])))
        
    return distances

#Locating Neighbors

class MyKNeighborsClassifier():
    """
    Vanilla implementation of KNN algorithm.
    """
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    def fit(self, X, y):
        """
        Fit the model using X as array of features and y as array of labels.
        """
        n_samples = X.shape[0]
        # number of neighbors can't be larger then number of samples
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y
        
    def pred_from_neighbors(self, training_set, labels, test_instance, k):
        distances = euclidean_distance(training_set, test_instance)
        
        # combining arrays as columns
        distances = sp.c_[distances, labels]
        # sorting array by value of first column
        sorted_distances = distances[distances[:,0].argsort()]
        # selecting labels associeted with k smallest distances
        targets = sorted_distances[0:k,1]

        unique, counts = np.unique(targets, return_counts=True)
        return(unique[np.argmax(counts)])
        
        
    def predict(self, X_test):
        
        # number of predictions to make and number of features inside single sample
        n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
        for i in range(n_predictions):
            # calculation of single prediction
            predictions[i] = self.pred_from_neighbors(self.X, self.y, X_test[i, :], self.n_neighbors)

        return(predictions)


# instantiate learning model (k = 3)
my_classifier = MyKNeighborsClassifier(n_neighbors=3)

# fitting the model
my_classifier.fit(X_train, y_train)

# predicting the test set results
my_y_pred = my_classifier.predict(X_test)

#We then check the predicted classes against the ground truth labels 
#and use Scikit-Learn accuracy module to calculate the accuracy of our classifier.

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy: ' + str(round(accuracy, 2)) + ' %.')

#########################################################################################
##############                       Naive Bayes                   ###################### 
######################################################################################### 

#The dataset can be downloaded from the following URL https://www.kaggle.com/uciml/sms-spam-collection-dataset/downloads/spam.csv/1.

#As always, we begin by importing the libraries we would utilize.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comment the magic command below if not running in Jupyter notebook
%matplotlib inline

#Next we load the dataset using Pandas and display the first 5 rows.
data = pd.read_csv('spam.csv', encoding='latin-1')
data.head(5)

#Let us plot a bar chart to visualize the distribution of legitimate and spam messages.
count_class = pd.value_counts(data['v1'], sort= True)
count_class.plot(kind='bar', color=[['blue', 'red']])
plt.title('Bar chart')
plt.show()

#vectorization
from sklearn.feature_extraction.text import CountVectorizer
f = CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
print(np.shape(X))

 

#Next we map our target variables into categories and split the dataset into train and test sets.
from sklearn.model_selection import train_test_split

data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = train_test_split(X, data['v1'], test_size=0.25, random_state=42)

#The next step involves initializing the Naive Bayes model and training it on the data.
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

#Finally, we gauge the model performance on the test set.
score = clf.score(X_test, y_test)
print('Accuracy: {}'.format(score))



#########################################################################################
##############            Decision Trees and Random Forest         ###################### 
######################################################################################### 

#dataset at https://gist.github.com/tijptjik/9408623/archive/b237fa5848349a14a14e5d4107dc7897c21951f5.zip

# First, lets load the dataset and use Pandas head method to have a look at it.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comment the magic command below if not running in Jupyter notebook
%matplotlib inline

dataset = pd.read_csv('wine.csv')
dataset.head(5)

#The next thing we do is split the dataset into predictors and targets, sometimes referred to as features and labels respectively.
features = dataset.drop(['Wine'], axis=1)
labels = dataset['Wine']

#we divide the dataset into a train and test split.
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

#All that is left is for us to import the decision tree classifier and fit it to our data.
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

classifier.fit(features_train, labels_train)

#We can now evaluate the trained model on the test set and print out the accuracy.

pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))

#Random Forests

import numpy as np
import pandas as pd

# load dataset
dataset = pd.read_csv('wine.csv')

# separate features and labels
features = dataset.drop(['Wine'], axis=1)
labels = dataset['Wine']

# split dataset into train and test sets
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

# import random forest classifier from sklearn
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# fit classifier on data
classifier.fit(features_train, labels_train)

# predict classes of test set samples
pred = classifier.predict(features_test)

# evaluate classifier performance using accuracy metric
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {:.2f}'.format(accuracy))



#########################################################################################
##############                   Neural Networks                   ###################### 
######################################################################################### 

#As before, let’s import the necessary library/libraries so that we can work on the data:
import pandas as pd
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
print (results_list)


#########################################################################################
##############                   Clustering                        ###################### 
######################################################################################### 

##dataset at https://www.kaggle.com/saurabh00007/iriscsv/downloads/Iris.csv/1

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset from csv file
dataset = pd.read_csv('Iris.csv')

# display first five observations
dataset.head(5)

x = dataset.drop(['Id', 'Species'], axis=1)
x = x.values # select values and convert dataframe to numpy array

# finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = [] # array to hold sum of squared distances within clusters

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares') # within cluster sum of squares
plt.show()

# creating the kmeans object
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

# Next we use the fit_predict method on our object. 
y_kmeans = kmeans.fit_predict(x)

# visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


#########################################################################################
##############       Support Vector Machine (SVM) classifier       ###################### 
######################################################################################### 

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset from csv file
dataset = pd.read_csv('diabetes.csv')


# create features and labels
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

# split dataset into training set and test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

# import support vector machine classifier
from sklearn.svm import SVC
classifier = SVC()

# fit data
classifier.fit(features_train, labels_train)

# get predicted class labels
pred = classifier.predict(features_test)

# get accuracy of model on test set
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy: {}'.format(accuracy))


from sklearn.svm import SVC
classifier = SVC(kernel='linear')



#########################################################################################
##############              Deep Learning Case Studies             ###################### 
######################################################################################### 

#The churn modelling dataset can be downloaded at: https://www.kaggle.com/aakash50897/churn-modellingcsv/data

# import all relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tflearn

# load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# get first five rows (observations)
dataset.head()

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train = np.reshape(y_train, (-1, 1)) # reshape y_train to [None, 1]
y_test = np.reshape(y_test, (-1, 1)) # reshape y_test to [None, 1]

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# build the neural network
net = tflearn.input_data(shape=[None, 11])
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 1, activation='tanh')
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net)

# we start training by applying gradient descent algorithm
model.fit(X_train, y_train, n_epoch=10, batch_size=16, validation_set=(X_test, y_test),
          show_metric=True, run_id="dense_model")

 
#################################
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# load IMDB dataset
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# data preprocessing
# sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# building the convolutional network
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)

# import tflearn, layers and data utilties
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# load IMDB dataset
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# data preprocessing
# sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# building the convolutional network
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)


