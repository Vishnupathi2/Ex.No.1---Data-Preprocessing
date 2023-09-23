# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1.Importing the libraries

2.Importing the dataset

3.Taking care of missing data

4.Encoding categorical data

5.Normalizing the data

6.Splitting the data into test and train

## PROGRAM:
```
Developed by:Vishnupathi A
Register no:212221223004
```
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```
## OUTPUT:
# df.head():
![1](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/e872f897-7999-41a9-9a9e-f43c2e9423dd)

# df.isnull().sum():
![2](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/8f5a4748-6c6e-4d45-ba73-9ab03a157e00)

# df value:
![3](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/5f76c64c-3665-48d2-bccf-4af9d51b5871)

# VALUES OF INPUT AND OUTPUT DATA ON VAR X AND Y:
![4](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/6b1e12f5-9d06-4fb3-94d3-9c53955e11bb)
![5](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/5fc3fde4-3d8b-4f31-8ce4-e766899a477f)

# NORMALIZING DATA:
![6](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/764bc78b-1438-445b-8da7-13e8fbddb27b)

# X_TRAIN AND Y_TRAIN VALUES:
![7](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/8d6d64e7-bafc-419d-8658-2b297f3df1e2)

# X AND Y VALUES:
![8](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/bbc76739-b36d-48e6-8c46-6cd7be9b7ee8)

# X_TEST AND Y_TEST VALUES:
![9](https://github.com/Vishnupathi2/Ex.No.1---Data-Preprocessing/assets/145830753/5b7a23ea-6487-4b09-9b67-ce475167c34a)

## RESULT:
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully .
