import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier

def predictClass(input_map):
    # Preprocess data

    ## Read from csv
    ciData = pd.read_csv('CencusIncome.data.csv', encoding='utf-8')
    ciData.drop(['income'], axis=1, inplace=True)

    ## Split categorical and non-categorical data
    categorical_columns = ciData.drop(['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week'], axis=1)
    non_categorical_columns = ciData.drop(categorical_columns, axis=1)

    ## Encode categorical data, scale/standardization non-categorical data
    ciData = pd.concat([ciData.drop(categorical_columns, axis=1), pd.get_dummies(categorical_columns)], axis=1)
 
    input_frame = pd.DataFrame(columns=list(ciData))
    input_frame.loc[0] = ['0' for i in range(len(list(input_frame)))]
        
    ### --- Test Prediction with MLP ---
    saved_classifier = joblib.load('income_mlp.model')

    print '\nInput : ', input_map

    # Label jenis hasil
    labels = ['<=50K', '>50k']

    # Pengisian categorical data pada dataframe
    for key,val in input_map.items():
        for col in list(input_frame):
            splitted_col = col.split(' ')
            if (key == col) :
                input_frame[col][0] = val
            elif (len(splitted_col) == 2 and val == splitted_col[1]):
                input_frame[col][0] = '1'
     
    input_frame.append(ciData)
 
    ## Standardization is used because currently it gives the best accuracy 
    input_frame = pd.concat([pd.DataFrame(data=scale(non_categorical_columns), columns=list(non_categorical_columns)), input_frame.drop(non_categorical_columns, axis=1)], axis=1)
 
    input_frame.drop(input_frame.index[1:], inplace=True)
    
    # Predict input menggunakan model
    predicted = saved_classifier.predict(input_frame)
    result = labels[predicted[0]]

    print '\nResult : ', result, '\n\n'
    return result