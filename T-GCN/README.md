# The manuscript
## T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction

Accurate and real-time traffic forecasting plays an important role in the Intelligent Traffic System and is of great significance for urban traffic planning, traffic management, and traffic control. However, traffic forecasting has always been considered an open scientific issue, owing to the constraints of urban road network topological structure and the law of dynamic change with time, namely, spatial dependence and temporal dependence. To capture the spatial and temporal dependence simultaneously, we propose a novel neural network-based traffic forecasting method, the temporal graph convolutional network (T-GCN) model, which is in combination with the graph convolutional network (GCN) and gated recurrent unit (GRU). Specifically, the GCN is used to learn complex topological structures to capture spatial dependence and the gated recurrent unit is used to learn dynamic changes of traffic data to capture temporal dependence. Then, the T-GCN model is employed to traffic forecasting based on the urban road network. Experiments demonstrate that our T-GCN model can obtain the spatio-temporal correlation from traffic data and the predictions outperform state-of-art baselines on real-world traffic datasets. 

The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901   or  https://arxiv.org/abs/1811.05320 

If this repo is useful in your research, please kindly consider citing our paper as follow.   
```
Bibtex
@article{zhao2019tgcn,
    title={T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction},
    author={Zhao, Ling and Song, Yujiao and Zhang, Chao and Liu, Yu and Wang, Pu and Lin, Tao and Deng, Min and Li, Haifeng},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    DOI = {10.1109/TITS.2019.2935152},
    year={2019},
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Zhao, Ling
%A Song, Yujiao
%A Zhang, Chao
%A Liu, Yu
%A Wang, Pu
%A Lin, Tao
%A Deng, Min
%A Li, Haifeng
%D 2019
%T T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
%B IEEE Transactions on Intelligent Transportation Systems
%R DOI:10.1109/TITS.2019.2935152
%! T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
```

GCN module:<br>
<img src="pics/gcn.png" width="400px" hight="400px" />


GRU module:<br>
<img src="pics/arc.png" width="400px" hight="400px" />


T-GCN Cell:<br>
<img src="pics/Cell.png" width="400px" hight="400px" />
