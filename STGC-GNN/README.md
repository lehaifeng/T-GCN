# STGC-GNNs: A GNN-based traffic prediction framework with a spatial-temporal Granger causality graph

## Introduction
We proposed a Spatial-temporal Granger causality(STGC) graph to model the spatial dependence of traffic road network.
This graph is pluggbable and can be input to GNN-based traffic prediction models which only using spatial graph to capture spatial dependence.
The manuscript can be visited at https://arxiv.org/abs/2210.16789 or https://www.sciencedirect.com/science/article/abs/pii/S0378437123004685.

## Data
We used METR_LA dataset, which is a Los Angeles highway dataset in the United States and one of the benchmark datasets for traffic prediction. 
Existing methods model spatial dependence as spatial weighting graph by converting the spatial distance between nodes into weights using Gaussian filtering.
This part of code can be found in the Python file named utils.The datasets is as following:

- **adj_mx.pkl/dj_mx_0.1k.pkl**: the well-calculated spatial weighting graph with sparsity of 0.1.
- **distance_la_2012.csv**:the spatial cost of travelling from one node to another.
- **graph_sensor_ids.txt**:id for each node.
- **graph_sensor_locations.csv**: the latitude and longitude of each node.
- **metr-la.h5**: observed time-series of traffic flow velocity of each node.

## Code
It is noting that we don't show the code of backbone models, which can be found and reproduced.

- **utils.py**: code for data preprocessing and calculating spatial weighting graph.
- **getSparseGranger.py**: code for calculating STGC graph.

## Requirments
Our code is based on Python3 (>= 3.6). The major libraries are listed as follows:

- **causal-learn(>0.1.2.0)** 
- **networkx(>2.6.3)** 

## Citation information
If our repo is useful to you, please cite our published paper as follows:
``` 
Bibtex
@article{li2023iGCL,
    title={STGC-GNNs: A GNN-based traffic prediction framework with a spatial–temporal Granger causality graph},
    author={He, Silu; Luo, Qinyao; Du, Ronghua; Zhao, Ling; He, Guangjun; Fu, Han; Li, Haifeng},
    journal={Physica A: Statistical Mechanics and its Applications},
    DOI = {10.1016/j.physa.2023.128913},
    year={2023},
    type = {Journal Article}
}
  
Endnote
%0 Journal Article
%A He, Silu
%A Luo, Qinyao
%A Du, Ronghua
%A Zhao, Ling
%A He, Guangjun
%A Fu, Han
%A Li, Haifeng
%D 2023
%T STGC-GNNs: A GNN-based traffic prediction framework with a spatial–temporal Granger causality graph
%B Physica A: Statistical Mechanics and its Applications
%R 10.1016/j.physa.2023.128913
%! STGC-GNNs: A GNN-based traffic prediction framework with a spatial–temporal Granger causality graph
``` 
