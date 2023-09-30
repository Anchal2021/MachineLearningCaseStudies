from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def winepredictor():
    #Load dataset
    wine=datasets.load_wine()

    #print the names of the features
    print(wine.feature_names)

    #print the label species(class_0,class_1,class_2)
    print(wine.target_names)

    #print the wine data(top 5 records)
    print(wine.data[0:5])

    #printthe wine labels
    print(wine.target)

    #split datasets into training set and test set
    x_train,x_test,Y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)

    #create KNN classifier
    knn=KNeighborsClassifier(n_neighbors=3)

    # Train the model using the tarining sets
    knn.fit(x_train, Y_train)

    #predict the response for test datasets
    y_pred =knn.predict(x_test)
    print("=======================================")
    print("Result:",y_pred)
    print("=======================================")

    #model Accuracy how often is the classifier correct
    print("Accuracy",metrics.accuracy_score(y_test,y_pred))

def main():
    print("-------Code by Anchal-------")
    print("-------Machine Learning Application-------")
    winepredictor()

if __name__=="__main__":
    main()
