This is a TensorFlow implementation of Temporal Graph Convolutional Network for the task of traffic flow prediction.


# Manuscript
## Temporal Graph Convolutional Network for Urban Traffic Flow Prediction Method

Accurate and real-time traffic forecasting plays an important role in the Intelligent Traffic System (ITS), it is of great significance for urban traffic planning, traffic management, and traffic control. However, traffic forecasting has always been a concerned “open” scientific issue, owing to the constraint of urban road network topological structure and the law of dynamic change with time, namely spatial dependence and temporal dependence. In order to capture the spatial and temporal dependence simultaneously, we propose a novel neural network-based traffic forecasting method, temporal graph convolutional network (T-GCN) model, which is in combination with the graph convolutional network (GCN) and gated recurrent unit (GRU). Specifically, the graph convolutional network is used to learn the complex topological structure to capture the spatial dependence and the gated recurrent unit is used to learn the dynamic change of traffic flow to capture the temporal dependence. And then, the T-GCN model is employed to realize the traffic forecasting task based on urban road network. Experiments demonstrate that our T-GCN model can obtain the spatio-temporal correlation from traffic data and the prediction effects outperform state-of-art baselines on real-world traffic datasets.

<img src="pics/gcn.png" width="400px" hight="400px" />
<img src="pics/Cell.png" width="400px" hight="400px" />
<img src="pics/arc.png" width="400px" hight="400px" />

The code of this paper can be downloaded at https://github.com/lehaifeng/T-GCN

# Code
## Requirements
* tensorflow
* scipy
* numpy
* matplotlib
* pandas
* math

## Run the demo
Python main.py

## Data Description
In order to use the model, we need
* An N by N adjacency matrix, which describes the spatial relationship between roads, 
* An N by D feature matrix, which describes the speed change over time on the roads.


