#!/usr/bin/env python3

import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
# Prediction model dependencies
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

new_cnn_model = tf.keras.models.load_model('cnn_model.h5')

print('Model Summary')
new_cnn_model.summary()


def xdata_preprocess(dat_arr):
    dat_arr = np.array(dat_arr)
    mean = dat_arr.mean(axis=0)
    std = dat_arr.std(axis=0)
    test_data_pp = (dat_arr - mean) / std
    # print('test_data_pp.shape', test_data_pp.shape)
    return test_data_pp


def ydata_preprocess(dat_arr):
    dat_arr = np.array(dat_arr)
    mean = dat_arr.mean(axis=0)
    std = dat_arr.std(axis=0)
    y_test_data_pp = (dat_arr - mean) / std
    # print('y_test_data_pp.shape', y_test_data_pp.shape)
    # print('mean:', mean, 'std:', std)
    return y_test_data_pp, mean, std


def ydata_revpreprocess(dat_arr, mean, std):
    dat_arr = np.array(dat_arr)
    new_pred = (dat_arr * std) + mean
    return np.array(new_pred)


def prediction(dat):
    # load data
    dat_file = np.load(dat)

    # test data
    test_data = np.array(dat_file[:, :-6])

    # test label data
    y_test_data = np.array(dat_file[:, -2:])

    # preprocess test label data
    y_test_data_reshaped, mean, std = ydata_preprocess(y_test_data)

    # preprocess test data
    test_data_pp = xdata_preprocess(test_data)
    input_dimension = 1
    test_data_reshaped = test_data_pp.reshape(test_data_pp.shape[0], test_data_pp.shape[1], input_dimension)

    # print("After reshape test data set shape:\n", test_data_reshaped.shape)
    # print("1 Sample shape:\n", test_data_reshaped[0].shape)

    # Model prediction
    test_predictions = new_cnn_model.predict(test_data_reshaped)

    # Reverse scaling test label data predictions
    new_predictions = ydata_revpreprocess(test_predictions, mean, std)

    # Calculating systolic MAE
    mae_tst0 = mean_absolute_error(y_test_data[:, 0], new_predictions[:, 0])
    # print('systolic mae', mae_tst0)

    # Calculating diastolic MAE
    mae_tst1 = mean_absolute_error(y_test_data[:, 1], new_predictions[:, 1])
    # print('diastolic mae', mae_tst1)

    # sp_prediction, sp_MAE, dp_prediction, dp_MAE
    sp_prediction = np.array(new_predictions[:, 0])
    sp_MAE = mae_tst0
    dp_prediction = np.array(new_predictions[:, 1])
    dp_MAE = mae_tst1

    return sp_prediction, sp_MAE, dp_prediction, dp_MAE


# get_ipython().system(' pip install tsfel # installing TSFEL for feature extraction')
def str2bool(v):
    return v.lower() in ("true", "1", "https", "load")


def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)


def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))] * 2)])


# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch


# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time


# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time / 1000


def influx_time_epoch(time):
    return time / 10e8


def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set


def calc_mae(gt, pred):
    return np.mean(abs(np.array(gt) - np.array(pred)))


# list1: label; list2: prediction
def plot_2vectors(label, pred, name):
    list1 = label
    list2 = np.array(pred)
    if len(list2.shape) == 2:
        mae = calc_mae(list1, list2[:, 0])
    else:
        mae = calc_mae(list1, list2)

    # zipped_lists = zip(list1, list2)
    # sorted_pairs = sorted(zipped_lists)

    # tuples = zip(*sorted_pairs)
    # list1, list2 = np.array([ list(tuple) for tuple in  tuples])

    # print(list1.shape)
    # print(list2.shape)

    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0, np.min(list2), f'MAE={mae}')

    # plt.plot(range(num_rows), list2, label=name + ' prediction')
    plt.scatter(np.arange(list2.shape[0]), list2[sorted_id], s=1, alpha=0.5, label=f'{name} prediction', color='blue')

    plt.scatter(np.arange(list1.shape[0]), list1[sorted_id], s=1, alpha=0.5, label=f'{name} label', color='red')

    # plt.plot(range(num_rows), list1, 'r.', label=name + ' label')

    plt.legend()
    plt.savefig(f'{name}.png')
    print(f'Saved plot to {name}.png')
    plt.show()
