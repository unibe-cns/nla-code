# Basic pattern completion with the Lagrangian network to predict the voltage of an iEEG recording based on recordings of the same point in time



Lagrangian neural networks are governed by a set of ordinary differential equations, that define the the process of learning output patterns from input patterns. 
The dynamics of the equations can be tuned through the weights of the network such that it favors certain fixed points of network voltages over others. 
These fixed points are favored as the network dynamics tries to lower the energy state according to a loss function that keeps the network in a self-prediction state as well as keep the output of some neurons near to some predefined targets.
Due to how the dynamics are defined, the network will always favor fixed points and is thus not directly suitable to be applied to time series. 

However, as a first approach, we consider the iEEG signals as a set of time series of which we can choose a subset as input and the remaining as target to make the network learn the input-target relationship. 
In this case, the model merely predicts the iEEG voltages of some electrodes based on others at the same specific slice in time without including the notion of time into the modelling itself. 
In later experiments, we will include part of the history in order to predict the future iEEG state.

* Input iEEG: 48 electrodes
* Output iEEG: 8 electrodes
* iEEG sample frequency: 512 Hz
* Training time (50 epochs, cpu): 4.5 hours

![Trained network example outputs](./layered_0_40_100_56/outputs/post_train_example.png)