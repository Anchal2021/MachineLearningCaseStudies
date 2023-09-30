import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def PlayPredictorModel(data_path):
    # Loading data
    data = pd.read_csv(data_path,index_col=0)
    # print("Size of Actual Dataset: ",len(data))

    # Clean, Prepare and manipulate data
    feature_names = ["Whether", "Temperature"]
    # print("Names of features: ", feature_names)

    whether = data.Whether
    temperature = data.Temperature
    play = data.Play

    # creating labelEncoder
    le=preprocessing.LabelEncoder()

    # converting string labels into numbers

    weather_encoded=le.fit_transform(whether)
    # print("weather",weather_encoded)

    temp_encoded=le.fit_transform(temperature)
    label=le.fit_transform(play)

    # combinig weather and temp into single listof tuples
    features=list(zip(weather_encoded,temp_encoded))

    # step3:train Data
    model=KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(features,label)

    return model


def prediction(weather,temperature):
    model = PlayPredictorModel("PlayPredictor.csv")
    # step4:Test data
    predicted = model.predict([[weather, temperature]])  # 0:Overcast  ,2:Mild
    return predicted


def main():
    print("----------Code by Anchal------------")
    print("Machine Learning Application")
    print("Play predictor application using K Nearest Knighbor algorithm")
    w = input("Enter a whether [Overcast,Sunny,Rainy]:")
    T=input("Enter a temperature [Hot,Mild,Cool]:")

    w_dict = { "Overcast": 0, "Sunny":1, "Rainy":2}
    T_dict={"Hot":1,"Mild":0,"Cool":2}

    res = prediction(w_dict[w],T_dict[T])
    print("Result:",res)
    if res[0] == 1:
        print("Its a wonderfull day you can play outside.")
    else:
        print("Its a Bad weather you cannot play Outside. ")


if __name__=="__main__":
    main()


















