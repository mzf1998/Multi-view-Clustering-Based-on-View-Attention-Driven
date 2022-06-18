# Multi-view Clustering Based on View-Attention Driven
This repo contains the code and data of the following paper Multi-view Clustering Based on View-Attention Driven

## Requirements

pytorch==1.7.0 

numpy>=1.19.1

scikit-learn>=0.23.2

munkres>=1.1.4

## Configuration

The hyper-parameters, the training options  are defined in configure.py.

## Datasets

The Caltech101-20, LandUse-21, and Scene-15 datasets are placed in "data" folder. The NoisyMNIST dataset could be downloaded from [cloud](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing).

## Usage

The code includes:

- an example implementation of the model

```bash
python run.py --dataset 0 --devices 0 --print_num 100 --test_time 5
```

You can get the following output:

```bash
Epoch : 100/500 ===> Reconstruction loss = 0.5299===> Reconstruction loss = 0.0453 ===>  ===> Contrastive loss = -8.9548e+02 ===> Loss = -8.9543e+02
view_concat {'kmeans': {'AMI': 0.7201, 'NMI': 0.7298, 'ARI': 0.906, 'accuracy': 0.7682, 'precision': 0.4756, 'recall': 0.4681, 'f_measure': 0.4483}}
Epoch : 200/500 ===> Reconstruction loss = 0.4642===> Reconstruction loss = 0.0401 ===>  ===> Contrastive loss = -8.9676e+02 ===> Loss = -8.9671e+02
view_concat {'kmeans': {'AMI': 0.7158, 'NMI': 0.7254, 'ARI': 0.8922, 'accuracy': 0.7569, 'precision': 0.4733, 'recall': 0.4495, 'f_measure': 0.4213}}
Epoch : 300/500 ===> Reconstruction loss = 0.4459===> Reconstruction loss = 0.0395 ===>  ===> Contrastive loss = -8.9402e+02 ===> Loss = -8.9397e+02
view_concat {'kmeans': {'AMI': 0.7158, 'NMI': 0.725, 'ARI': 0.8799, 'accuracy': 0.7557, 'precision': 0.4655, 'recall': 0.4441, 'f_measure': 0.4143}}
Epoch : 400/500 ===> Reconstruction loss = 0.4083===> Reconstruction loss = 0.0376 ===>  ===> Contrastive loss = -8.9606e+02 ===> Loss = -8.9602e+02
view_concat {'kmeans': {'AMI': 0.7119, 'NMI': 0.7212, 'ARI': 0.8714, 'accuracy': 0.7515, 'precision': 0.4592, 'recall': 0.443, 'f_measure': 0.4074}}
Epoch : 500/500 ===> Reconstruction loss = 0.3625===> Reconstruction loss = 0.0362 ===>  ===> Contrastive loss = -8.9688e+02 ===> Loss = -8.9684e+02
view_concat {'kmeans': {'AMI': 0.7094, 'NMI': 0.719, 'ARI': 0.8701, 'accuracy': 0.7536, 'precision': 0.4656, 'recall': 0.4457, 'f_measure': 0.4209}}
```


