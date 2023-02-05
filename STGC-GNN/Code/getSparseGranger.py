import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
import pickle
import csv
import seaborn as sns
from pandas import DataFrame
import utils
import networkx as nx
from causallearn.search.Granger.Granger import Granger
import heapq

# for METR_LA
pre_pkl_path = "Data/METR-LA/adj_mx.pkl"
pre_matrix = utils.load_pkl(pre_pkl_path)[2]
num_node = pre_matrix.shape[0]
sensor_ids_filename = "Data/METR-LA/graph_sensor_ids.txt"
distances_filename = "Data/METR-LA/distances_la_2012.csv"
file_path = "Data/METR-LA/week_average.npy"
save_path = "Output"


def get_average_week():
    data = utils.load_h5_data(file_path)
    data = utils.data_average(data)
    np.save("Output/week_average.npy", data)


def get_shortest_path_length(adjacency):
    G = nx.from_numpy_matrix(adjacency)
    cost_mx = np.zeros((num_node, num_node))
    length = dict(nx.all_pairs_dijkstra_path_length(G))
    path = dict(nx.all_pairs_dijkstra_path(G))
    for i in range(num_node):
        for j in range(num_node):
            cost_mx[i][j] = length[i][j]
    np.savetxt("Output/Hop_matrix.txt", cost_mx, fmt="%1.5f")
    return cost_mx, path


def get_spatial_distance():
    distance_mx = np.zeros((num_node, num_node), dtype=np.float32)
    distance_mx[:] = np.inf
    # METR_LA
    with open(sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(distances_filename, dtype={'from': 'str', 'to': 'str'})
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        distance_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    np.savetxt("Output/Cost_matrix.txt", distance_mx, fmt="%1.5f")
    return distance_mx


# using avrage value of weekly observed data to test GDTi
def get_ave_speed():
    ave_speed = np.zeros(num_node)
    data = np.load('Data/METR-LA/week_average.npy')
    for i in range(num_node):
        ave_speed[i] = np.mean(data[i])
    return ave_speed


# calculate the spatial-temporal lag
def get_sp_lag(cost_mx):
    sp_lag = np.zeros((num_node, num_node), dtype=int)
    speed = get_ave_speed()
    for i in range(num_node):
        for j in range(num_node):
            if cost_mx[i][j] != 0 and cost_mx[i][j] != np.inf:
                sp_lag[i][j] = math.ceil((cost_mx[i][j] / (1609 * speed[i])) * 12)  # 1609*v m/h
            else:
                sp_lag[i][j] = 0
    np.savetxt("Output/sp_lag.txt", sp_lag, fmt="%d")
    return sp_lag


def get_stgc_granger(data, sp_mx):
    G = Granger(4)
    Granger_matrix = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i != j:
                X_week = DataFrame(data[[i, j]].T, columns=['effect', 'cause'])
                # spatial-temporal aglignment
                X_week['cause'] = X_week['cause'].shift(sp_mx[i][j])
                X_week = X_week.iloc[sp_mx[i][j]:, :].values
                # granger causality test
                p_value_matrix = G.granger_test_2d(X_week)
                Granger_matrix[i][j] = p_value_matrix[1, 0]
    np.savetxt("Output/sparse_granger_4_directed.csv", Granger_matrix)
    return Granger_matrix


def get_by_significance(mx, sig):
    sparse_stgc_mx = np.identity(num_node, dtype=np.float32)
    hop_mx = np.loadtxt('Output/Hop_matrix.txt')
    for i in range(num_node):
        for j in range(num_node):
            if 0 < mx[i][j] < sig and hop_mx[i][j] == 1:
                sparse_stgc_mx[i][j] = 1
    return sparse_stgc_mx


def get_shorest_hop(distance, path):
    hop_mx = np.zeros((num_node, num_node), dtype=int)
    for i in range(num_node):
        for j in range(num_node):
            hop_mx[i][j] = len(path[i][j]) - 1
    np.savetxt("Output/Hop_matrix.txt", hop_mx, fmt="%1.5f")
    return hop_mx


def get_sparse_stgc(p_value_mx):
    distance_matrix = get_spatial_distance()
    # p_value_mx = pd.read_csv('Output/sparse_granger_4_directed.csv', header=None).values
    sparse_stgc_mx = get_by_significance(p_value_mx, 0.01)
    for i in range(num_node):
        for j in range(num_node):
            if sparse_stgc_mx[i][j] == 1 and distance_matrix[i][j] != np.inf:
                sparse_stgc_mx[i][j] = 1
            else:
                sparse_stgc_mx[i][j] = 0
    return sparse_stgc_mx


if __name__ == '__main__':
    get_average_week()
    cost_mx = get_spatial_distance()
    sp_lag = get_sp_lag(cost_mx)
    data = np.load("Output/week_average.npy")
    # spatial-temporal granger causality test
    stgc_mx = get_stgc_granger(data, sp_lag)
    # spatial-temporal granger causality graph
    get_sparse_stgc(stgc_mx)
