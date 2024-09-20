# Lagrangian Learning Framework

This repository provides a Tensorflow as well as a PyTorch implementation of the Lagrangian learning framework for neuronal and synaptic dynamics as described in  [A neuronal least-action principle for real-time learning in cortical circuits](https://www.biorxiv.org/content/10.1101/2023.03.25.534198v3).


Repository Organization
------------

    ├── README.md		<- The top-level README for researchers using this project.
    ├── experiments        	<- Folder containing all experiments mentioned in the paper
    │   ├── MNIST			<- MNIST experiments
    │   	├── supervised 		<- regular, supervised MNIST digit recognition
    │   └── iEEG           	<- time series experiments using iEEG signals
    │   	├── layered     	<- Learn to reproduce timeseries with a layered-recurrent architecture
    │   	└── fully-recurrent	<- Learn to reproduce timeseries with a fully-recurrent architecture
    │
    ├── model			<- Lagrangian model implementations in numpy (for reference), tensorflow and pytorch
    │
    ├── requirements.txt   <- The requirements file for reproducing the recommended python environment,i.e.
    │                          manually defined python package definitions suitable for running the code
    │
    └──frozen_requirements.txt   <- The requirements file for reproducing the exact python environment, i.e.
                              generated with `pip freeze > requirements.txt`



## Lagrangian learning framework

The Lagrangian learning framework introduces a novel model that derives real-time error-backpropagation in multi-areal cortical circuits from a least-action principle. In this model, ongoing synaptic plasticity minimizes an output error along the gradient with respect to the hidden synaptic strengths. The instantaneous downstream activity backpropagates upstream where it is compared with its driving activity to form a causal prediction error. The key feature is that the backpropagated activity is predictive for the upstream activity that caused the errors in the downstream neurons. A temporal delay ensuing from somatic filtering is counterbalanced by an advancement achieved through a Hodgkin-Huxley-type neuronal activation mechanism. Moreover, dynamic short-term synaptic depression in the backprojecting pathway is correcting for the distortion of downstream errors by the neuronal transfer functions. To calculate a prediction error in the upstream area, local interneurons – driven by lateral connections from the pyramidal neurons within that area – learn to cancel the backpropagated activity. What these interneurons cannot explain away remains as a neuron-specific real-time prediction error that induces plasticity in the synapses projecting forward to these pyramidal neurons. The presented theory suggests how prospective coding and error-correcting plasticity can work together in recurrent cortical circuits to learn time-continuous responses to ongoing stimulus streams.

## Experiments

The above described lagrangian framework is tested on various basic benchmarks which are described below in detail.

### MNIST experiments

In this section, we explore supervised experiments on the popular MNIST dataset. The supervised experiment reproduces the classical benchmark of classifying MNIST digits using a lagrangian network.

#### Supervised experiment

The supervised experiment entails the classical approach to train a network to classify MNIST digits from images out of the MNIST dataset. The network is trained in a supervised manner to output the correct label when exposed to a certain image of the corresponding digit.

To run this experiment:
```bash
python experiments/MNIST/supervised/lagrange_mnist_training.py
```

The results show up in `experiments/MNIST/supervised/output`.

### iEEG experiment

The iEEG experiment basically learns to pattern-complete timeseries in a similar fashion to the unsupervised MNIST experiment. The iEEG data is kept in its raw time series form and is sliced in time such that one sample contains the current voltage of all recorded iEEG electrodes at one specific point in time. During training, the lagrangian network is provided with these time-slices on all neurons of the equally sized output layer to learn to correlate the voltages of the signal, similar to the MNIST image above in the unsupervised experiment. After training, only a part of the iEEG slice is applied to some the output layer neurons, which makes the network respond with the missing iEEG voltages on the remaining output layer neurons.

#### Layered-recurrent network

The iEEG experiment is performed with a layered-recurrent network.

To run this experiment:

```bash
cd experiments/iEEG/pattern-completion/layered_0_40_100_56/;python train_predict.py
```

The results show up in `experiments/iEEG/pattern-completion/layered_0_40_100_56/outputs`.

#### Fully-recurrent network

The iEEG experiment is performed with a fully-recurrent network.

To run this experiment:

```bash
cd experiments/iEEG/pattern-completion/fully_recurrent_40+56/;python train_predict.py
```

The results show up in `experiments/iEEG/pattern-completion/fully_recurrent_40+56/outputs`.