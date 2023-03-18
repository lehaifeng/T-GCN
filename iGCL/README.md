Invariant-Discriminative Graph Contrastive Learning (iGCL)
==========================================
This repository provides a PyTorch implementation of iGCL as described in the paper:
> [Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations](https://arxiv.org/abs/2210.08345)
---------------------------------------------------

### Abstract
<p align="justify">
Graph contrastive learning is a promising direction toward alleviating the label dependence, poor generalization and weak robustness of graph neural networks, learning representations with invariance, and discriminability by solving pretasks. The pretasks are mainly built on mutual information estimation, which requires data augmentation to construct positive samples with similar semantics to learn invariant signals and negative samples with dissimilar semantics in order to empower representation discriminability. However, an appropriate data augmentation configuration depends heavily on lots of empirical trials such as choosing the compositions of data augmentation techniques and the corresponding hyperparameter settings. We propose an augmentation-free graph contrastive learning method, invariant-discriminative graph contrastive learning (iGCL), that does not intrinsically require negative samples. iGCL designs the invariant-discriminative loss (ID loss) to learn invariant and discriminative representations. On the one hand, ID loss learns invariant signals by directly minimizing the mean square error between the target samples and positive samples in the representation space. On the other hand, ID loss ensures that the representations are discriminative by an orthonormal constraint forcing the different dimensions of representations to be independent of each other. This prevents representations from collapsing to a point or subspace. Our theoretical analysis explains the effectiveness of ID loss from the perspectives of the redundancy reduction criterion, canonical correlation analysis, and information bottleneck principle. The experimental results demonstrate that iGCL outperforms all baselines on 5 node classification benchmark datasets. iGCL also shows superior performance for different label ratios and is capable of resisting graph attacks, which indicates that iGCL has excellent generalization and robustness.  


### Datasets
All datasets are loaded and processed by [Pytorch-Geometric](https://github.com/pyg-team/pytorch_geometric). Note that the version of Pytorch-Geometric is `1.7.0`, which may be a slight difference with the latest version on loading these datasets.  


### Options

Training an iGCL is handled by the `main_graph_ssl.py` script which provides the following command line arguments.  

```
  --root          STRING    Path of saved processed data files.     Required is False.    Default is ./Data.
  --dataset       STRING    Name of the datasets.                   Required is False.    Default is WikiCS.
  --dim           INT       Dimension of representations.           Required is False.    Default is 1024.
  --num_layers    INT       Number of gnn layers.                   Required is False.    Default is 2.
  --lambd         FLOAT     Weight of normalization loss.           Required is False.    Default is 0.005.
  --topk          INT       Number of positive samples.             Required is False.    Default is 6.
  --epochs        INT       The maximum iterations of training.     Required is False     Default is 1000.
```

### Examples
The following command trains an iGCL on WikiCS.
```commandline
python main.py --root ./Data --dataset WikiCS --dim 1024 --num_layers 2 --lambd 0.005 --topk 6 --epochs 1000
```  
For more details about the hyperparameter setting, please refer to the paper.

### Acknowledge
The code of this repository references [CCA-SSG](https://github.com/hengruizhang98/CCA-SSG) and [AFGRL](https://github.com/Namkyeong/AFGRL). Thanks for their excellent work.

### Citation information
If our repo is useful to you, please cite our published paper as follows:
```
Bibtex
@article{li2023iGCL,
    title={Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations},
    author={Li, Haifeng; Cao, Jun; Zhu*, Jiawei; Luo, Qinyao; He, Silu; Wang, Xuying},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    DOI = {10.1109/TNNLS.2023.3248871},
    year={2023},
    type = {Journal Article}
}
  
Endnote
%0 Journal Article
%A Li, Haifeng
%A Cao, Jun
%A Zhu, Jiawei
%A Luo, Qinyao
%A He, Silu
%A Wang, Xuying
%D 2023
%T Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations
%B IEEE Transactions on Neural Networks and Learning Systems
%R 10.1109/TNNLS.2023.3248871
%! Augmentation-Free Graph Contrastive Learning of Invariant-Discriminative Representations
```
