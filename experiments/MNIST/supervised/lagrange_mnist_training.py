

import random
import numpy as np
import datetime
import argparse

import sys

sys.path.append("/media/benelot/SPACE/loci/IDSC/Physiology/projects/lagrangian-learning-the-final-nla/")
from model.network_params import ArchType, ActivationFunction, NetworkParams, IntegrationMethod


# plotting
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# mnist dataset and metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from experiments.MNIST.common import get_digits_from_class, load_params, save_params

from model.network_params import ComputationBackendType
from model.lagrange_model import LagrangeNetwork
from tqdm import tqdm


def get_prediction_accuracy(y, y_hat):
    """
    Test accuracy of predictions (average number of correct labels).

    Args:
        y: true labels.
        y_hat: predicted labels.

    Returns:
    """

    class_rates = []
    for (predicted_label, true_label) in zip(y_hat, y):  # argmax of true labels should be equal to argmax of distorted/decayed labels
        class_rates.append(np.argmax(true_label) == np.argmax(predicted_label))

    return np.mean(class_rates)


def get_confusion_matrix(y_hat, y):
    """
    Get confusion matrix to understand problematic confusions.
    """
    c = confusion_matrix(y_hat, y)
    return c


def save_accuracies(accuracies, prefix=datetime.datetime.now().strftime("%Y%m%d%H%M%S")):
    """
    Plot accuracy and save values to text.
    """
    plt.close()
    plt.plot(accuracies, linewidth=2.)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    try:
        plt.savefig("./output/" + prefix + '_accuracies.pdf')
        np.savetxt("./output/" + prefix + '_accuracies.txt', accuracies)
    except (PermissionError, FileNotFoundError) as e:
        print("Could not save accuracies due to missing permissions (They should be printed above).\n Please fix permissions, I will try again after next epoch...")


class MnistTrainer:

    def __init__(self, params: NetworkParams, classes: int=10, train_samples: int=2000, test_samples: int=100, epoch_hook=None):
        """ Initialize the mnist trainer.

        Args:
            params: Network parameters for a new network.
            classes: Number of MNIST classes for labels.
            train_samples: Number of training samples
            test_samples: Number of test samples
            epoch_hook: Function hook executed after every epoch
        """
        self.params = params
        self.epoch_hook = epoch_hook

        # load mnist's official training and test data
        # Load data from https://www.openml.org/d/554
        training_set_size = train_samples  # default 2000
        test_set_size = test_samples  # default 100
        digit_classes = classes  # default 10

        print("Loading MNIST...", end=" ")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='.', as_frame=False)
        print("...Done.")

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        # get mnist training images from official training set of size training_set_size
        print("Preparing MNIST train images...", end=" ")
        train_dataset = []
        for digit_class in range(digit_classes):
            print(digit_class, end=" ")
            train_dataset += get_digits_from_class(x_train, y_train, self.params.layers[-1], digit_class, training_set_size, params.activation_function == ActivationFunction.SIGMOID)

        self.train_dataset = train_dataset
        print("...Done.")

        # get mnist test images from official training set of size test_set_size
        print("Preparing MNIST test images...", end=" ")
        test_dataset = []
        for digit_class in range(digit_classes):
            print(digit_class, end=" ")
            test_dataset += get_digits_from_class(x_test, y_test, self.params.layers[-1], digit_class, test_set_size, params.activation_function == ActivationFunction.SIGMOID)
        self.test_dataset = test_dataset
        print("...Done.")

        # shuffle both training sets
        print("Shuffling MNIST train/test images...", end=" ")
        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)
        print("...Done.")

    def train(self, network, batch_size=1, epoch_qty: int=10, skip_initial_test=False, verbose=3):

        # create a prefix for the file containing a unique timestamp
        accuracy_filename_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # append timestamp

        test_x, test_y = [x for x, _ in self.test_dataset], [y for _, y in self.test_dataset]
        accuracies = []

        # start training the network with the training set and periodically test prediction with the test set
        print("Start training network...")

        is_epoch_known = False
        for epoch in tqdm(range(epoch_qty), desc="Epochs"):

            # find last already trained epoch
            if not is_epoch_known:
                try:
                    load_params(network, epoch, dry_run=True)  # test if network params could be loaded
                    print('Epoch {0} already done.'.format(epoch))
                    continue

                except Exception as e:
                    print('Epoch {0} not done.'.format(epoch))
                    if epoch != 0:
                        print('Loading weights of epoch {0} to continue...'.format(epoch - 1))
                        load_params(network, epoch - 1, dry_run=False)
                        print('...Done.')

                    is_epoch_known = True

            if epoch != 0:
                print('\r' + 'Train epoch {0}... | '.format(epoch), end='')
                random.shuffle(self.train_dataset)
                train_x, train_y = [x for x, _ in self.train_dataset], [y for _, y in self.train_dataset]
                network.fit(train_x, train_y, batch_size=batch_size, verbose=verbose if verbose >= 3 else 0)
                skip_initial_test = False

            if not skip_initial_test:
                print('Test epoch {0}... | '.format(epoch), end='')
                y_hat = network.predict(test_x, batch_size=batch_size, verbose=verbose if verbose >= 3 else 0)
                accuracies.append(get_prediction_accuracy(test_y, y_hat))
                print(accuracies)
                save_accuracies(accuracies, accuracy_filename_prefix)
                print('Save epoch {0}... | '.format(epoch), end='')
                save_params(network, epoch)

                print("Confusion matrix after epoch ", epoch, ": ")
                print(get_confusion_matrix(np.argmax(y_hat, axis=1), np.argmax(test_y[:len(y_hat)], axis=1)))

            if self.epoch_hook is not None:
                self.epoch_hook(network)

        return network, accuracies


if __name__ == "__main__":
    # get args from outside
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train lagrange mnist on all 10 classes and check accuracy.')
    parser.add_argument('--framework', default='torch', type=str, help='The framework to be used for training: tf or torch')
    parser.add_argument('--classes', default=10, type=int, help='Number of classes to distinguish.')
    parser.add_argument('--train_samples', default=2000, type=int, help='Number of training samples per class.')
    parser.add_argument('--test_samples', default=100, type=int, help='Number of test samples per class.')

    args, unknown = parser.parse_known_args()

    # setup network parameters
    params = NetworkParams()
    params.layers = [28 * 28, 300, 100, args.classes]  # layer structure
    params.learning_rate_factors = [1, 0.2, 0.1]  # learning rate factor for different layers
    params.arg_tau = 10.0  # membrane time constant
    params.arg_dt = 1.0  # integration step
    params.arg_beta = 0.0  # nudging param. beta

    params.arg_lrate_biases = 0.0001  # learning rate of the biases
    params.arg_lrate_biases_I = 0.1  # learning rate of interneuron biases
    params.arg_lrate_W = 0.0001  # learning rate of the network weights
    params.arg_lrate_W_B = 0.0  # learning rate of feedback weights B
    params.arg_lrate_PI = 0.0  # learning rate of interneuron to principal weights W_PI
    params.arg_w_init_params = {'mean': 0, 'std': 0.1, 'clip': 0.3}  # weight initialization params for normal distribution sampling (mean, std, clip)
    params.arg_bias_init_params = {'mean': 0, 'std': 0.1, 'clip': 0.3}  # bias initialization params for normal distribution sampling(mean, std, clip)
    params.arg_interneuron_b_scale = 1  # interneuron bias scaling

    params.use_sparse_mult = False  # use sparse multiplication when calculating matrix product of two matrices
    params.integration_method = IntegrationMethod.CHOLESKY_SOLVER  # use Cholesky decomposition (faster, might fail due to numerical issues) or LU decomposition (slower, but works always)
    params.dtype = np.float32  # data type used in calculations

    params.use_biases = True
    params.only_discriminative = True
    params.with_input_dynamics = False

    params.network_architecture = ArchType.LAYERED_FEEDFORWARD  # network architecture used (defines the connectivity pattern / shape of weight matrix)
    #                                                    currently implemented: LAYERED_FORWARD (FF network),
    #                                                                           LAYERED_RECURRENT (every layer is recurrent)
    #                                                                           FULLY_RECURRENT (completely recurrent),
    #                                                                           IN_RECURRENT_OUT (input layer - layers of recurr. networks - output layer)
    params.use_interneurons = False  # use either interneuron circuit (True) or weight transport (False)
    params.dynamic_interneurons = False  # either interneuron voltage is integrated dynamically (True) or set to the stat. value instantly (False = ideal theory)
    params.activation_function = ActivationFunction.HARD_SIGMOID  # change activation function

    params.check_with_profiler = False  # run graph once with profiler on
    params.rnd_seed = 2354323495  # numpy random seed

    random.seed(1238349532)

    network = LagrangeNetwork(ComputationBackendType(args.framework), params)

    # test batch sizes
    # pred_times = []
    # train_times = []
    # for batch_size in tqdm([1, 32, 64, 128]):
    #     time_per_prediction_step, time_per_train_step = network.network._test_simulation_run(batch_size)
    #     pred_times.append((batch_size, time_per_prediction_step))
    #     train_times.append((batch_size, time_per_train_step))
    #     print(pred_times)
    #     print(train_times)
    #
    # print(pred_times)
    # print(train_times)

    batch_size = 180

    trainer = MnistTrainer(params, classes=args.classes, train_samples=args.train_samples, test_samples=args.test_samples)
    trainer.train(network, batch_size=batch_size)
