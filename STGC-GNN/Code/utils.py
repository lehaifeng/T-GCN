import datetime
from statistics import mean

import pandas as pd
import numpy as np
import h5py
import deepdish
import pickle
import statsmodels as sm

file_path = "Data/METR-LA/metr-la.h5"  ## ['df']


# load time-series data
def load_h5_data(file_path):
    file = deepdish.io.load(file_path)
    return file['df']  # LA
    # return file['df'].values  #ndarray(34272, 207)
    # return file['df'].T  # dataframe(207,34727)


# fill missing data with average value of same moment in other days
def data_prepocess(data):
    date_index = data.index  # DatetimeIndex
    datetime_index = data.index.date  # datetime ndarray "2012-03-01"
    datetime_unique = np.unique(datetime_index)  # datetime list
    day_num = len(datetime_unique)  # total days
    data = data.T  # row is sensor  col is date
    sensor_num, time_num = data.shape
    for n in range(sensor_num):
        series = data.iloc[n]  # for a sensor
        series_value = series.values  # ndarray value
        for i in range(time_num):
            if series_value[i] == 0:
                valid_list = []
                null_list = []
                ave_value = 0
                j = 1
                while i + 288 * j < time_num:
                    if series_value[i + 288 * j] == 0:
                        null_list.append(j)
                        j += 1
                    else:
                        valid_list.append(j)
                        j += 1
                for m in valid_list:
                    ave_value += series_value[i + 288 * m]
                if len(valid_list) != 0:
                    ave_value = ave_value / len(valid_list)
                    series_value[i] = ave_value
                    # series.loc[m] =ave_value
                    for m in null_list:
                        series_value[i + 288 * m] = ave_value
    return data


# calculate the average value of each moment in a period of week
def data_average(data):
    data = data.T  # row is sensor  col is date
    sensor_num, time_num = data.shape
    data_week = np.zeros((sensor_num, 288 * 7))
    for n in range(sensor_num):
        series = data.iloc[n]  # for a sensor
        series_value = series.values  # ndarray value
        for i in range(7):
            for j in range(288):
                values = []
                k = 0
                while 288 * i + j + 288 * 7 * k < time_num:
                    if series_value[288 * i + j + 288 * 7 * k] != 0:
                        values.append(series_value[288 * i + j + 288 * 7 * k])
                    k += 1
                value_mean = mean(values)
                data_week[n][288 * i + j] = value_mean
    return data_week


def get_same_moment_list(timestamp, daycount):
    return pd.date_range(timestamp, freq='D', periods=daycount)  # DatetimeIndex objet


def load_pkl(pkl_file):
    f = open(pkl_file, 'rb')
    data = pickle.load(f, encoding='bytes')
    return data


def stationary_test(data):
    sensors_num = data.shape[0]
    for i in range(sensors_num):
        is_stationary = "no"
        result = sm.tsa.stattools.adfuller(data[i])
        if result[0] < result[4]["1%"]:
            is_stationary = "yes"
        print(
            "no.{}'s p-value is{},test statistic is{},critical level at 1% is {},Stationary is {}".format(i, result[1],
                                                                                                          result[0],
                                                                                                          result[4][
                                                                                                              "1%"],
                                                                                                          is_stationary))


# calculate the spatial weighting matrix
def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx

