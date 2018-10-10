import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(data_path):
    rides = pd.read_csv(data_path)

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for dummy_field in dummy_fields:
        dummies = pd.get_dummies(rides[dummy_field], prefix=dummy_field, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                    'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)

    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

    scaled_features = {}
    for quant_feature in quant_features:
        mean, std = data[quant_feature].mean(), data[quant_feature].std()
        scaled_features[quant_feature] = [mean, std]
        data[quant_feature] = (data[quant_feature] - mean) / std

    test_data = data[-21*24:]

    data = data[:-21*24]

    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    return features, targets, test_features, test_targets, scaled_features, rides['dteday']
