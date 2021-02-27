This is a TensorFlow implementation of AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting.

# The manuscript
## AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting

Traffic forecasting is a fundamental and challenging task in the field of intelligent transportation. Accurate forecasting not only depends on the historical traffic flow information but also needs to consider the influence of a variety of external factors, such as weather conditions and surrounding POI distribution. Recently, spatiotemporal models integrating graph convolutional networks and recurrent neural networks have become traffic forecasting research hotspots and have made significant progress. However, few works integrate external factors. Therefore, based on the assumption that introducing external factors can enhance the spatiotemporal accuracy in predicting traffic and improving interpretability, we propose an attribute-augmented spatiotemporal graph convolutional network (AST-GCN). We model the external factors as dynamic attributes and static attributes and design an attribute-augmented unit to encode and integrate those factors into the spatiotemporal graph convolution model. Experiments on real datasets show the effectiveness of considering external information on traffic speed forecasting tasks when compared with traditional traffic prediction methods. Moreover, under different attribute-augmented schemes and prediction horizon settings, the forecasting accuracy of the AST-GCN is higher than that of the baselines. 

## Data Description
There are two datasets in the data fold.<br>
(1) SZ-taxi. This dataset was the taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015. We selected 156 major roads of Luohu District as the study area.<br>
(2) SZ-POI: This dataset provides information about POIs surrounding selected road sections. The POI categories can be divided into nine types: catering services, enterprises, shopping services, transportation facilities, education services, living services, medical services, accommodations, and others.<br>
(3) SZ-Weather: This auxiliary information contains the weather conditions about the study area recorded every 15 minutes in January 2015. The weather conditions are divided into five categories: sunny, cloudy, fog, light rain, and heavy rain. With the information of time-varying weather conditions.<br>

The manuscript can be visited at https://ieeexplore.ieee.org/document/9363197 or https://arxiv.org/abs/2011.11004.


