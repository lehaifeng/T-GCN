This is a TensorFlow implementation of T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction.

## Requirements:
* tensorflow
* scipy
* numpy
* matplotlib
* pandas
* math

## Run the demo
Python main.py

Our baselines included: <br>
(1) History Average model (HA)<br>
(2) Autoregressive Integrated Moving Average model (ARIMA)<br>
(3) Support Vector Regression model (SVR)<br>
(4) Graph Convolutional Network model (GCN)<br>
(5) Gated Recurrent Unit model (GRU)<br>

The python implementations of HA/ARIMA/SVR models were in the baselines.py; The GCN and GRU models were in gcn.py and gru.py respective.


The T-GCN model was in the tgcn.py


## Implement
In this paper, we set time interval as 15 minutes, 30 minutes, 45 minutes and 60 minutes.

In the sz_taxi dataset, we set the parameters seq_len to 4 and pre_len to 1, 2, 3, 4; In the los_loop dataset, we set the parameters seq_len 12 and the pre_len to 3, 6, 9, 12 respectively.

## Data Description
There are two datasets in the data fold.<br>
(1) SZ-taxi. This dataset was the taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015. We selected 156 major roads of Luohu District as the study area.<br>
(2) Los-loop. This dataset was collected in the highway of Los Angeles County in real time by loop detectors. We selected 207 sensors and its traffic speed from Mar.1 to Mar.7, 2012

In order to use the model, we need
* A N by N adjacency matrix, which describes the spatial relationship between roads, 
* A N by D feature matrix, which describes the speed change over time on the roads.

