from enum import Enum
import numpy
import json
import re


class ComputationBackendType(str, Enum):
    """
    Type of computational backend to use for the lagrangian.
    """

    TENSORFLOW = 'tf'
    PYTORCH = 'torch'


class IntegrationMethod(str, Enum):
    """
    Type of method to get the voltage gradients.
    """

    CHOLESKY_SOLVER = 'Cholesky-based Solver'                                   # CHOLESKY solver (faster, might fail due to numerical issues)
    LU_SOLVER = 'LU-based Solver'                                               # LU solver (slower, but works always)
    LS_SOLVER = 'LS-based Solver'                                               # Least squares solver (iterative gradient solver0
    HESSIAN_SOLVER = 'Hessian Solver'                                           # Hessian Solver
    ROOT_SOLVER = 'Root Finding Based Solver'                                   # root finding algorithm based Solver
    EULER_METHOD = 'Euler Method'                                               # method based on Eq. 14.0 (the dot u equation), approximating dot u(t) with dot u(t-1)
    FORWARD_INTEGRATION = 'Forward Integration'                                 # method based on Eq. 14.0, with a scheme approximating \\tau \\bar \\dot r and e
    RICHARDSON_1ST_METHOD = 'Richardson 1st Order Method'                       # method based on Eq 14.0, approximating dot u(t-1) with dot u(t-2) and dot u (t-3)
    RICHARDSON_2ND_METHOD = 'Richardson 2nd Order Method'                       # method based on Eq. 14.0, approximating dot u(t-1) with (t-2), (t-3) and (t-5)
    COMPARE_CHOLESKY_SOLVER = 'Compare Cholesky Solver'                         # comparative mode, integrate with cholesky, but calculate all other updates
    COMPARE_ROOT_SOLVER = 'Compare Root Finding Solver'                         # comparative mode, integrate with root solver, but calculate all other updates
    COMPARE_EULER_METHOD = 'Compare Euler Method'                               # comparative mode, integrate with euler method, but calculate all other updates
    COMPARE_RICHARDSON_1ST_METHOD = 'Compare Richardson (1st order) Method'     # comparative mode, integrate with richardson 1st, but calculate all other updates
    COMPARE_RICHARDSON_2ND_METHOD = 'Compare Richardson (2st order) Method'     # comparative mode, integrate with richardson 2nd, but calculate all other updates


class ArchType(str, Enum):
    """
    Type of connection architecture for the network.
    """

    LAYERED_FEEDFORWARD = 'layered-feedforward'                                 # make fully-connected layers from input neurons to output neurons (classical feedforward NN)
    LAYERED_FEEDBACKWARD = 'layered-feedbackward'                               # make fully-connected layers from output neurons to input neurons (special case FFNN)
    LAYERED_RECURRENT = 'layered-recurrent'                                     # make fully-connected layers from input neurons to output neurons, and inverse
    LAYERED_RECURRENT_RECURRENT = 'layered-recurrent-recurrent'                 # make fully-connected layers from input neurons to output neurons, each layer recurrent
    FULLY_RECURRENT = 'fully-recurrent'                                         # make fully-connected network of neurons, except of self-connections
    IN_RECURRENT_OUT = 'in-recurrent-out'                                       # make input layer, recurrent hidden layers and output layer


class ActivationFunction(str, Enum):
    """
    Activation function for the neurons.
    """

    SIGMOID = 'sigmoid'                                                         # sigmoidal activation function
    RELU = 'relu'                                                               # ReLU activation function
    HARD_SIGMOID = 'hard-sigmoid'                                               # Hard Sigmoid activation function
    SIGMOID_INTEGRAL = 'sigmoid-integral'                                       # Sigmoid Integral activation function


class NetworkParams:
    """
    Lagrangian network parameters class. It allows to save to and load from json.
    """

    layers = [2, 10, 10, 1]                                         # layer structure
    learning_rate_factors = [1, 1, 1]                               # learning rate factor to scale learning rates for each layer
    arg_tau = 10.0                                                  # membrane time constant
    arg_dt = 0.1                                                    # integration step
    arg_beta = 0.1                                                  # nudging parameter beta

    arg_lrate_W = 0.1                                               # learning rate of the network weights
    arg_lrate_W_B = 0.0                                               # learning rate of feedback weights B
    arg_lrate_PI = 0.0                                              # learning rate of interneuron to pyramidal neuron weights W_PI
    arg_lrate_biases = 0.1                                          # learning rate of the biases
    arg_lrate_biases_I = 0.1                                        # learning rate of interneuron biases

    arg_w_init_params = {'mean': 0, 'std': 1.0, 'clip': 3.0}        # weight initialization params for normal distribution sampling (mean, std, clip)
    arg_clip_weight = False                                         # clip the weights and biases back to their initialized range
    arg_clip_weight_deriv = False                                   # weight derivative update clipping
    arg_weight_deriv_clip_value = 0.05                              # weight derivative update clipping
    arg_bias_init_params = {'mean': 0, 'std': 0.1, 'clip': 0.3}     # bias initialization params for normal distribution sampling(mean, std, clip)
    arg_interneuron_b_scale = 1                                     # interneuron backward weight scaling
    with_afferent_input_weights = True;                             """if the input neurons have incoming synaptic weight connections"""
    use_input_neurons = True;                                       """if the first layer should be considered input neurons and is clamped to input"""
    use_input_as_rates = False;                                     """if the inputs are interpreted as rates instead of voltages"""
    is_weight_constant_for_u = False;                               """if the weights are considered constant (in derivatives) during the calculation of dot u"""

    integration_method = IntegrationMethod.EULER_METHOD             # network integration method
    integration_method_iterations = 1                               # number of iterations for interated integration methods
    dtype = numpy.float32                                           # data type used in calculations

    use_biases = True                                               # False: use model neurons with weights only, True: use biases+weights
    only_discriminative = False                                     # if the network is not used as a generative model, then the input layers needs no biases and no incoming connections
    allow_selfloops = False                                         # if neurons are allowed to have a connection to themselves
    with_input_dynamics = False                                     # if input neurons receive any inputs through synaptic connections
    network_architecture = ArchType.LAYERED_FEEDFORWARD             # network architecture used (defines the connectivity pattern / shape of weight matrix)
    #        currently implemented: LAYERED_FORWARD (FF network),
    #                               LAYERED_RECURRENT (every layer is recurrent)
    #                               FULLY_RECURRENT (completely recurrent),
    #                               IN_RECURRENT_OUT (input layer - layers of recurr. networks - output layer)
    use_interneurons = False                                        # train interneuron circuit (True) or use weight transport (False)
    dynamic_interneurons = False                                    # either interneuron voltage is integrated dynamically (True) or set to the stat. value instantly (False, =ideal theory)
    activation_function = ActivationFunction.SIGMOID                # change activation function, currently implemented: sigmoid, ReLU, capped ReLU

    check_with_profiler = False                                     # run graph once with profiler on to reveal bottlenecks
    write_tensorboard = False                                       # write outputs to be analysed with tensorboard
    rnd_seed = 2354323495                                           # random seed

    def __init__(self, file_name=None):
        if file_name is not None:
            self.load_params(file_name)

    def load_params(self, file_name):
        """
        Load parameters from json file.
        """
        with open(file_name, 'r') as file:
            deserialize_dict = json.load(file)
            for key, value in deserialize_dict.items():
                if isinstance(value, str) and 'numpy' in value:
                    value = eval(value)
                elif isinstance(value, str) and getattr(self, key).__class__ is not str:
                    key_class = getattr(self, key).__class__
                    value = key_class(value)

                setattr(self, key, value)

    def save_params(self, file_name):
        """
        Save parameters to json file.
        """
        with open(file_name, 'w') as file:
            file.write(self.to_json())

    def to_json(self):
        """
        Turn network params into json.
        """
        serialize_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if callable(value):
                    if value is numpy.float32 or value is numpy.float64:
                        value = re.search('\'(.+?)\'', str(value)).group(1)
                    else:
                        break
                serialize_dict[key] = value

        return json.dumps(serialize_dict, indent=4)

    def __str__(self):
        """
        Return string representation.
        Returns:

        """
        return self.to_json()
