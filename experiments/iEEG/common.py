
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from pathlib import Path


def resample_dataset(images, dt):
    """
    Resample time series data if needed (iEEG is given with dt of 2ms).
    Args:
        images: time series data.
        dt: Desired timestep of iEEG.

    Returns:

    """
    new_images = []
    for i in range(len(images)):
        new_images.append(np.interp(np.arange(0, len(images[i]), dt / 2.), np.arange(0, len(images[i])), images[i]))
    return np.asarray(new_images)


# functions for training, testing and plotting
def test_series(index, signals, network, duration, unclamped_neurons, base_output_folder='./outputs'):
    print("Testing series...")

    if not os.path.exists(base_output_folder):
        os.mkdir(base_output_folder)

    # beta is an array for the last layer describing which output neurons are nudged. Entries that are zero are not nudged (e.g. for testing -> pattern compl.)
    beta_nudging_strength = 10.0
    beta_vector = np.zeros(sum(network.layers))
    beta_vector[-network.layers[-1]:] = beta_nudging_strength
    beta_vector[-unclamped_neurons:] = 0
    network.set_beta(beta_vector)

    # beta_vector = np.ones(network.layers[-1]) * 10
    # beta_vector[-unclamped_neurons:] = 0
    # network.set_beta(beta_vector)

    volts = []
    for i in tqdm(range(duration)):
        network.update_network(np.array([]), signals[:, i], train_W=False, train_B=False, train_PI=False)
        volts.append(network.get_voltages()) #[-unclamped_neurons:])
    volts = np.asarray(volts)
    plot_output(index, signals, volts, duration, unclamped_neurons, base_output_folder)
    np.save(base_output_folder + '/volts_' + str(index), volts)
    print("...Done.")
    return signals, volts


def plot_output(index, signals, volts, duration, unclamped_neurons, base_output_folder='./outputs'):

    if not os.path.exists(base_output_folder):
        os.mkdir(base_output_folder)

    plt.close('all')
    fig, ax = plt.subplots(unclamped_neurons, 1)
    for i in range(unclamped_neurons):
        ax[i].plot(volts[:, -unclamped_neurons + i], color='darkred')
        ax[i].plot(signals[-unclamped_neurons + i][:duration], color='k', alpha=0.5, linestyle='--')
    plt.savefig(base_output_folder + '/series_' + str(index) + '.png')


def train_series(images, network, beta, duration):
    print("Training series...")
    network.set_beta(beta)
    repetitions = 20
    t = tqdm(total=repetitions * duration)
    for j in range(repetitions):
        for i in range(duration):
            network.update_network(np.array([]), images[:, i], train_W=True, train_B=False, train_PI=False)
            t.update(1)
    t.close()
    print("...Done.")


def save_params(network, epoch_i, base_save_path="./checkpoints"):
    """
    Save current params of the network.
    Args:
        network: the lagrangian network.
        epoch_i: the current training epoch.
        base_save_path: base save path for saving the parameters.

    """
    save_path = Path(base_save_path + "/network_params_e" + str(epoch_i))
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    network.save(save_path)


def load_params(network, epoch_i, base_load_path="./checkpoints", dry_run=False):
    """
    Load current params of the network from file.
    Args:
        network:
        epoch_i:
        base_load_path:
        dry_run:

    Returns:

    """

    load_path = Path(base_load_path + "/network_params_e" + str(epoch_i))
    if not load_path.exists():
        raise Exception("{0} does not exist. Loading network aborted.".format(load_path))
    elif dry_run:
        return True

    return network.load(load_path)
