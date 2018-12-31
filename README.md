This is a TensorFlow implementation of T-GCN: A Temporal Graph ConvolutionalNetwork for Traffic Prediction.


# Manuscript
## T-GCN: A Temporal Graph ConvolutionalNetwork for Traffic Prediction

Accurate and real-time traffic forecasting plays an important role in the Intelligent Traffic System and is of great significance for urban traffic planning, traffic management, and traffic control. However, traffic forecasting has always been considered an “open” scientific issue, owing to the constraints of urban road network topological structure and the law of dynamic change with time, namely, spatial dependence and temporal dependence. To capture the spatial and temporal dependence simultaneously, we propose a novel neural network-based traffic forecasting method, the temporal graph convolutional network (T-GCN) model, which is in combination with the graph convolutional network (GCN) and gated recurrent unit (GRU). Specifically, the GCN is used to learn complex topological structures to capture spatial dependence and the gated recurrent unit is used to learn dynamic changes of traffic data to capture temporal dependence. Then, the T-GCN model is employed to traffic forecasting based on the urban road network. Experiments demonstrate that our T-GCN model can obtain the spatio-temporal correlation from traffic data and the predictions outperform state-of-art baselines on real-world traffic datasets.

The manuscript can be visited at https://arxiv.org/abs/1811.05320

<img src="pics/gcn.png" width="400px" hight="400px" />
GCN module
<img src="pics/arc.png" width="400px" hight="400px" />
GRU module
<img src="pics/Cell.png" width="400px" hight="400px" />
T-GCN Cell


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
* A N by N adjacency matrix, which describes the spatial relationship between roads, 
* A N by D feature matrix, which describes the speed change over time on the roads.


