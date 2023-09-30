import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def DiabetesPredictor(filePath):
    diabetes = pd.read_csv(filePath)
    print("Columns of dataset:")
    print(diabetes.columns)

    print("First 5 records of the dataset: ")
    print(diabetes.head())

    print("Dimension of Diabetes dataset: ",diabetes.shape)

    data_train,data_test,label_train,label_test = train_test_split( diabetes.loc[:,diabetes.columns!='Outcome'], diabetes['Outcome'], stratify= diabetes['Outcome'], random_state=66)

    dectree = DecisionTreeClassifier(random_state=0)

    dectree.fit(data_train,label_train)

    print("Accuracy on training dataset: ",dectree.score(data_train,label_train))
    print("Accuracy on testing dataset: ", dectree.score(data_test, label_test))

    #Increase depth = 3
    print("Add MaxDepth= 3")

    dectree = DecisionTreeClassifier(random_state=0, max_depth=3)

    dectree.fit(data_train,label_train)

    print("Accuracy on training dataset: ",dectree.score(data_train,label_train))
    print("Accuracy on testing dataset: ", dectree.score(data_test, label_test))

    print("Feature Importance: ",dectree.feature_importances_)

    plot_feature_importance_diabetes(dectree,diabetes)

def plot_feature_importance_diabetes(model,diabetes):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features),model.feature_importances_,align='center')
    diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)
    plt.show()


def main():
    print("-------Code by Anchal-------")
    print("-------Machine Learning Application-------")
    DiabetesPredictor("diabetes_data.csv")

if __name__=="__main__":
    main()
