# T-GCN-PyTorch

This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320). 

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

Notice that [the original implementation](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning
* lightning-bolts

## Model Training

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# T-GCN
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.
