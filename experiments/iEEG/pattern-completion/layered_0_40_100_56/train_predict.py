

import argparse
import matplotlib

matplotlib.use('Agg')
import numpy as np

import random
import sys

sys.path.append("/media/benelot/SPACE/loci/IDSC/Physiology/projects/lagrangian-learning-the-final-nla/")
random.seed(1238349532)
from model.network_params import NetworkParams
from model.network_params import ComputationBackendType
from model.lagrange_model_torch import LagrangeNetwork
import json

from experiments.iEEG.common import resample_dataset, train_series, test_series, save_params, load_params


if __name__ == "__main__":
    # get args from outside
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train network to reproduce iEEG data using a layered-recurrent network.')
    #parser.add_argument('--framework', default=ComputationBackendType.PYTORCH.value, type=str, help='The framework to be used for training: tf or torch')

    args, unknown = parser.parse_known_args()

    # experiment params
    dataset_name = '../../data/raw/EEG_set.json'
    dataset_id = 1
    experiment_duration = 4000 # ms
    unclamped_neurons = 10

    network_params = NetworkParams()
    network_params.load_params("layered_params.json")

    experiment_duration = int(experiment_duration * 2 / network_params.arg_dt)

    # load dataset
    with open(dataset_name, 'r') as infile:
        data = json.load(infile)['patient_' + str(dataset_id)]
    input_data = np.array(data['input']) / 2.0
    print(f"Input iEEG signals: {input_data.shape}")

    target_data = np.array(data['target']) / 2.0
    print(f"Target iEEG signals: {target_data.shape}")

    dataset = np.append(input_data, target_data, axis=0)

    network = LagrangeNetwork(network_params)

    if network_params.arg_dt != 2.0:
        print('Upsampling iEEG...')
        dataset = resample_dataset(dataset, network_params.arg_dt)
        print('...Done.')

    print('Start training...')  # start training
    is_epoch_known = False
    for epoch in range(2000):

        # find last already trained epoch
        if not is_epoch_known:
            try:
                load_params(network, epoch, dry_run=True)  # test if network params could be loaded
                print('Epoch {0} already done.'.format(epoch))
                continue

            except Exception as e:
                if epoch != 0:
                    print('Epoch {0} not done.'.format(epoch))
                    print('Loading weights of epoch {0} to continue...'.format(epoch - 1))
                    load_params(network, epoch - 1, dry_run=False)
                    print('...Done.')
                is_epoch_known = True

        if epoch != 0:
            # start/continue training at epoch
            print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
            train_series(dataset, network, network_params.arg_beta, experiment_duration)

        print('Test epoch {0}... | '.format(epoch), end='')
        test_series(epoch, dataset, network, experiment_duration, unclamped_neurons)
        print('Save epoch {0}... | '.format(epoch), end='')
        save_params(network, epoch)
        print('...Done. |')

    print('...Done.')
