# Pytorch implementation of the Lagrange model. Trainable only with single samples. Just for reference.
# This is the version that is easier to read as it is without matrix-batch notation.
#
# Authors: Benjamin Ellenberger (benelot@github) and Nicolas Deperrois (nicozenith@github)


import time

import numpy as np
import torch
import torch.nn as nn
# utils
from torch.utils.data import DataLoader

# network parameters
from model.network_params import ArchType, IntegrationMethod, ActivationFunction, NetworkParams
from utils.torch_bottleneck import bottleneck
from utils.torch_utils import get_torch_dtype, SimpleDataset


class LagrangeNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Lagrange model.
    Prefixes: st = state, pl = placeholder, obs = observable, arg = argument.
    """

    # INIT
    # TODO: Look into RtDeel formulas to properly use backward etc weights instead of W^T

    def __init__(self, params: NetworkParams):
        """
        Initialize network with network parameters.

        Args:
            params: network parameters for initialization.
        """
        # device execution context
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');  """device for torch"""
        self.dtype = get_torch_dtype(params.dtype);                                  """torch data type"""

        # network and simulation parameters and flags
        self.params = params;                                                        """store parameters of network (for deepcopies, saving and loading)"""
        self.rnd_seed = params.rnd_seed;                                             """random seed for reproducibility"""
        self._set_random_seed(params.rnd_seed)                                       # set random seed in all frameworks
        self.tau = params.arg_tau;                                                   """[ms] differential equation time constant"""
        self.dt = params.arg_dt;                                                     """[ms] dt timestep of numerical integration"""

        # # network simulation parameters
        self.use_input_neurons = params.use_input_neurons;                           """if the first layer is considered input neurons and pulled towards input by the derivative"""
        self.use_input_as_rates = params.use_input_as_rates;                         """use input as rates"""
        self.input_size = params.layers[0];                                          """number of input neurons"""
        self.neuron_qty = sum(params.layers);                                        """number of neurons"""
        self.layers = params.layers;                                                 """the network's layer of neurons definition"""
        self.with_input_dynamics = params.with_input_dynamics;                       """if the network's input neurons gets input from incoming synaptic connections"""
        self.set_beta(params.arg_beta);                                              """nudging parameter (pushing the output neurons towards a target)"""

        # # weight and bias parameters
        self.arg_w_init_params = params.arg_w_init_params;                           """the mean, stddev and clip of the weight sampling normal distribution"""
        self.arg_clip_weights = params.arg_clip_weight;                              """if the weights should be clipped to keep them within the original, sampled range"""
        self.arg_clip_weight_derivs = params.arg_clip_weight_deriv;                  """if the weight derivatives should be clipped to keep them within a certain range"""
        self.arg_weight_deriv_clip_value = params.arg_weight_deriv_clip_value;       """the value of min and max around zero above which weigh derivatives are clipped"""
        self.with_afferent_input_weights = params.with_afferent_input_weights;       """if the input neurons have incoming synaptic weight connections"""

        self.use_biases = params.use_biases;                                         """if the network's neurons should have biases"""
        self.arg_bias_init_params = params.arg_bias_init_params;                     """the mean, stddev and clip of the bias sampling normal distribution"""

        # # learning rates
        self.arg_lrate_W = params.arg_lrate_W;                                       """the learning rate to scale \dot W when added to W"""
        self.arg_lrate_W_B = params.arg_lrate_W_B;                                   """the learning rate to scale \dot B when added to B"""
        self.arg_lrate_PI = params.arg_lrate_PI;                                     """the learning rate to scale \dot W^{PI} when added to W^{PI}"""

        self.arg_lrate_biases = params.arg_lrate_biases;                             """the learning rate to scale \dot biases when added to biases"""
        self.arg_lrate_biases_I = params.arg_lrate_biases_I;                         """the learning rate to scale \dot biases^I when added to biases^I"""

        # setup network input and target voltages
        self.st_inputs = torch.zeros(self.input_size, dtype=self.dtype, device=self.device);          """voltages to which the input neurons are set (+ input dynamics)"""
        self.st_old_inputs = torch.zeros(self.input_size, dtype=self.dtype, device=self.device);      """previous input voltages"""
        self.st_targets = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device);         """voltages towards which the output neurons are nudged (see beta)"""
        self.st_old_targets = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device);     """previous target voltages"""

        # fit/predict variables to generate exponentially decaying/smooth input transients (for non-timeseries inputs to avoid large steps)
        self.decay_steps = torch.linspace(-20, 90, 100, dtype=self.dtype, device=self.device);                   """steps of exponential decay while learning"""
        self.old_decay_x = torch.zeros(self.input_size, dtype=self.dtype, device=self.device);                   """previous input of fit/predict"""
        self.old_decay_y = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device);                  """previous output of fit/predict"""
        self.dummy_label = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

        # integration methods supported by this implementation
        self.INTEGRATION_METHODS = [IntegrationMethod.CHOLESKY_SOLVER,                  # linear equation solver based on a cholesky-decomposition
                                    IntegrationMethod.LU_SOLVER,                        # linear equation solver based on a LU-decomposition
                                    IntegrationMethod.EULER_METHOD,                     # lagrange framework implementing euler method
                                    IntegrationMethod.RICHARDSON_1ST_METHOD,            # lagrange framework implementing euler method, improved with a 1st order extrapolation
                                    IntegrationMethod.RICHARDSON_2ND_METHOD]            # lagrange framework implementing euler method, improved with a 1st order extrapolation

        # setup activation function and solver
        self.act_function, self.act_func_deriv, self.act_func_second_deriv, self.inverse_act_func = self._generate_activation_function(params.activation_function)     # set functions from type
        self.integration_method = None
        self.set_integration_method(params.integration_method)                                                                                  # set integration method

        # setup neuron connection weight and bias masks (to initialize and learn only synaptic connections according to network architecture)
        self.weight_mask = self._make_connection_weight_mask(params.network_architecture, params.layers, params.learning_rate_factors, params.with_afferent_input_weights,
                                                             params.only_discriminative,
                                                             params.allow_selfloops);    """(adjacency matrix) synaptic weight mask encoding the network structure"""
        self.bias_mask = self._make_neuron_bias_mask(params.layers, self.input_size, params.learning_rate_factors, params.only_discriminative); """vector mask of enabled biases"""

        # initialize network
        self._initialize_network(params.arg_w_init_params, params.use_biases, params.arg_bias_init_params)

        # perform basic profiling
        if params.check_with_profiler:
            print("\nProfile model for single sample:\n--------------------------------")
            self._profile_model()

        # perform single sample test run to get network stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # reinitialize network variables after testing simulation
        self._initialize_network(params.arg_w_init_params, params.use_biases, params.arg_bias_init_params)

    # NETWORK SETUP & INITIALIZATION METHODS

    def _make_connection_weight_mask(self, network_architecture, layers, learning_rate_factors, with_afferent_input_weights=True, only_discriminative=True, allow_selfloops=False):
        """
        Create matrix weight mask encoding of the network structure based on the network architecture.
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.

        Args:
            network_architecture: the architecture of synaptic connections of the network
            layers: layer definition of network
            learning_rate_factors: learning rate scalars for each layer
            only_discriminative: if only discriminative, the network's input neurons have no biases and no incoming synaptic connections
            allow_selfloops: If a neuron can have a synaptic connection to itself

        Returns:
            weight mask encoding of the network structure.
        """
        if network_architecture == ArchType.LAYERED_FEEDFORWARD:                        # feed forward mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)   # # make a feed forward mask

        elif network_architecture == ArchType.LAYERED_FEEDBACKWARD:                     # feed backward mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask = weight_mask.T

        elif network_architecture == ArchType.LAYERED_RECURRENT:                        # recurrence through layered forward and backward connections
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)   # # make a feed forward mask
            weight_mask += weight_mask.T                                                # # make a feed backward mask

        elif network_architecture == ArchType.LAYERED_RECURRENT_RECURRENT:              # recurrent layers, forward and backward connections
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)   # # make a feed forward mask
            weight_mask += weight_mask.T                                                # # make a feed backward mask
            weight_mask = self._add_intralayer_recurrent_connections(weight_mask, layers, learning_rate_factors)  # make each layer recurrent except input layer

        elif network_architecture == ArchType.IN_RECURRENT_OUT:                         # input layer, recurrent hidden layers, output layer
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)   # # make a feed forward mask
            weight_mask = self._add_intralayer_recurrent_connections(weight_mask, layers, learning_rate_factors)  # make each layer recurrent except input layer

        elif network_architecture == ArchType.FULLY_RECURRENT:                          # ALL to ALL connection mask (with or without self loops)
            weight_mask = np.ones((sum(layers), sum(layers)))                           # #  create a complete graph mask of weights

            if not allow_selfloops:                                                     # if self loops are not allowed
                np.fill_diagonal(weight_mask, 0)                                        # #  drop loops

        else:
            raise NotImplementedError("Mask type ", network_architecture.name, " not implemented.")

            # only discriminative = no connections projecting back to the input layer and no biases in the input layer
        if only_discriminative or not with_afferent_input_weights:
            weight_mask[:layers[0], :] *= 0

        weight_mask = torch.tensor(weight_mask, dtype=self.dtype, device=self.device)
        return weight_mask

    @staticmethod
    def _make_feed_forward_mask(layers, learning_rate_factors):
        """
        Returns a mask for a feedforward architecture.

        Args:
            layers: a list containing the number of neurons per layer
            learning_rate_factors: contains learning rate multipliers for each layer

        Returns:
            weight mask of a feedforward structure

        Adapted from Jonathan Binas (@MILA)

        Weight mask encoding of the network structure
        ---------------------------------------------
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.
        """
        neuron_qty = int(np.sum(layers))                                                # total quantity of neurons
        layer_qty = len(layers)                                                         # total quantity of layers
        mask = np.zeros((neuron_qty, neuron_qty))                                       # create adjacency matrix of neuron connections
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]   # calculate start of layers, the postsynaptic neurons to each layer
        for i in range(len(learning_rate_factors)):
            mask[layer_offsets[i + 1]:layer_offsets[i + 2], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]  # connect all of layer i (i to i+1) with layer i + 1

        return mask

    @staticmethod
    def _add_intralayer_recurrent_connections(weight_mask, layers, learning_rate_factors):
        """
        Adds recurrent structure to every layer of network except input layer.

        Args:
            weight_mask: weight mask encoding the synaptic connection network structure
            layers: a list containing the number of neurons per layer.
            learning_rate_factors: contains learning rate multipliers for each layer.

        Returns:
            weight mask with added recurrent connections

        """
        layer_qty = len(layers)                                                                                                     # total quantity of layers
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]                                               # calculate start of layer offsets

        for i in range(1, len(learning_rate_factors)):                                                                              # exclude input layer
            weight_mask[layer_offsets[i]:layer_offsets[i + 1], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]    # connect all of layer i with itself
        np.fill_diagonal(weight_mask, 0)                                                                                            # drop loops (autosynapses)
        return weight_mask

    def _make_neuron_bias_mask(self, layers, input_size, learning_rate_factors, only_discriminative):
        """
        Creates vector mask for biases (similar to mask for weights).

        Args:
            layers: layer structure of network
            input_size: size of input
            learning_rate_factors: factors that scale the learning rate for neurons in a layer
            only_discriminative: if the network is only discriminative, the input layer has no biases

        Returns:
            bias mask of network structure
        """
        neuron_qty = int(np.sum(layers))                                                    # total quantity of neurons
        layer_qty = len(layers)                                                             # total quantity of layers
        bias_mask = np.ones(neuron_qty)
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]       # calculate start of layers
        for i in range(len(learning_rate_factors)):
            bias_mask[layer_offsets[i]:layer_offsets[i + 1]] *= learning_rate_factors[i]    # set learning rate factors for biases of each layer

            # only discriminative = no connections projecting back to the input layer and no biases in the input layer
        if only_discriminative:
            bias_mask[:input_size] *= 0

        bias_mask = torch.tensor(bias_mask, dtype=self.dtype, device=self.device)
        return bias_mask

    @staticmethod
    def _generate_activation_function(activation_function):
        """
        Implementation of different activation functions.

        :math:`r(t) = \\rho(u(t)), \, r'(t) = \\rho'(u(t)), \, r''(t) = \\rho''(u(t))`

        Args:
            activation_function: type of activation function

        Returns:
            activation function, 1st and 2nd derivative
        """
        if activation_function == ActivationFunction.SIGMOID:                                                       # if the activation is a sigmoid
            act_function = nn.Sigmoid()                                                                             # define the activation function as a sigmoid of voltages
            act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))                 # function of the 1st derivative
            act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))    # function of the 2nd derivative
            inv_act_func = lambda rates: torch.log(rates/(1-rates)).clamp(-1000, 1000)                              # inverse in the sense that applying the act function leads again to the same rate
        elif activation_function == ActivationFunction.RELU:                                                        # regular ReLU unit
            act_function = nn.ReLU()
            act_func_deriv = lambda voltages: ((voltages > 0) * 1.0).clone().detach()                               # 1st derivative of the ReLU function
            act_func_second_deriv = lambda voltages: 0.0                                                            # 2nd derivative of the ReLU function
            inv_act_func = lambda rates: nn.ReLU                                                                    # inverse in the sense that applying the act function leads again to the same rate
        elif activation_function == ActivationFunction.HARD_SIGMOID:                                                # ReLU which is clipped to 0-1
            act_function = lambda voltages: voltages.clamp(0, 1)                                                    # Hard ReLU
            act_func_deriv = lambda voltages: ((voltages >= 0) * (voltages <= 1) * 1.0).float().clone().detach()    # 1st derivative of Hard ReLU
            act_func_second_deriv = lambda voltages: voltages * 0.0                                                 # 2nd derivative of Hard ReLU
            inv_act_func = lambda rates: rates.clamp(0, 1)                                                          # inverse in the sense that applying the act function leads again to the same rate
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv, inv_act_func

    def _create_observables(self):
        """
        Make variables to inspect extended network state (but are not needed otherwise).
        """
        # observables
        self.obs_errors = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)            # observe neural somato-dendritic mismatch errors
        self.obs_error_lookaheads = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # observe neural somato-dendritic mismatch error look aheads

    def _initialize_network(self, weight_init_params, use_biases, bias_init_params):
        """
        Set up voltages, weights, and their derivatives.

        :math:`u(t), \\dot u(t), u^I(t), \\dot u^I(t), W, \\dot W, b, \\dot{b}, b^I, \\dot b^I, W^{IP}, \\dot W^{IP}, W^{PI}, \\dot W^{PI}, W^{B}, \\dot W^{B}`

        Args:
            weight_init_params: mean, stddev and clip of weight sampling random normal distribution

            use_biases: if the network uses biases
            bias_init_params: mean, stddev and clip of bias sampling random normal distribution

        """

        # states

        self.initialize_voltages()

        self.initialize_weights_and_biases(weight_init_params, use_biases, bias_init_params)

        # observables
        self._create_observables()

        # traces
        # Fill dot_voltage_trace and dot_error_trace with 4 constant data points each
        self.dot_voltages_trace = [self.st_dot_voltages,
                                   self.st_dot_voltages,
                                   self.st_dot_voltages,
                                   self.st_dot_voltages]

    def initialize_voltages(self):
        """
        Initialize the voltages (:math:`u(t), \\dot u(t), u^I(t), \\dot u^I(t)`) to zero.
        """
        # setup voltage variable for neurons
        self.st_voltages = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)       # initialize all voltages to the resting membrane potential, 0
        self.st_dot_voltages = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)   # initialize all voltage derivatives to 0

        # setup voltage variable for inter-neurons
        self.st_voltages_I = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)     # voltages of interneurons to resting membrane potential

    def initialize_weights_and_biases(self, weight_init_params, use_biases, bias_init_params):
        """
        Initialize the weights and biases to their respective ranges.

        :math:`W, \\dot W, b, \\dot{b}, b^I, \\dot b^I, W^{IP}, \\dot W^{IP}, W^{PI}, \\dot W^{PI}, W^{B}, \\dot W^{B}`

        Args:
            weight_init_params: mean, stddev and clip of weight sampling random normal distribution

            use_biases: if the network uses biases
            bias_init_params: mean, stddev and clip of bias sampling random normal distribution

        """
        # setup weights variable for pyramidal neurons
        self.st_weights = torch.tensor(self._create_initial_weights(self.weight_mask, **weight_init_params), dtype=self.dtype, device=self.device)
        self.st_dot_weights = self.weight_mask * 0.0  # initialize the weight derivatives to zero

        # setup bias variables for neurons
        self.st_biases = torch.tensor(self._create_initial_biases(use_biases, self.bias_mask, **bias_init_params), dtype=self.dtype, device=self.device)
        self.st_dot_biases = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Initialize weight matrix sampled from normal distribution.

        Args:
            weight_mask: Mask of synaptic connections based on network architecture
            mean: mean of normal distribution
            std: standard deviation of normal distribution
            clip: normal distribution clipped to (min, max) around mean

        Returns:
            Weight matrix randomly sampled from a normal distribution with :mean:, :std: and :clip:
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), mean - clip, mean + clip) * (weight_mask.cpu().numpy() > 0)  # initialize weights with normal sample where mask is larger 0

    @staticmethod
    def _create_initial_biases(use_biases, bias_mask, mean, std, clip):
        """
        Initialize bias matrix sampled from normal distribution (or set biases to zero if only weights are used).

        Args:
            use_biases: if biases are used
            bias_mask: which neurons have biases to learn
            mean: mean of normal distribution
            std: standard deviation of normal distribution
            clip: normal distribution clipped to (min, max) around mean

        Returns:
            Bias matrix randomly sampled from a normal distribution with :mean:, :std: and :clip:

        """
        neuron_qty = bias_mask.shape[0]
        if use_biases:
            return np.clip(np.random.normal(mean, std, size=neuron_qty), mean - clip, mean + clip) * (bias_mask.cpu().numpy() > 0)  # initialize biases with normal sample where mask is larger 0
        else:
            return np.zeros(neuron_qty)  # set biases to zero

    # BASIC PROFILE AND TEST METHODS

    def _profile_model(self):
        """
        Profiling of the network algorithm/computational graph. Performs one network update step and saves the profile data.
        The data can be loaded in Chrome at chrome://tracing/.
        """
        sample_input = np.ones(self.input_size)  # input voltages set to 1
        sample_output = np.ones(self.neuron_qty)  # output voltages set to 1

        _, cpu_profiler, cuda_profiler = bottleneck(lambda: self.update_network(sample_input, sample_output, True))

        if cpu_profiler is not None:
            cpu_profiler.export_chrome_trace('torch_cpu_timeline.json')

        if cuda_profiler is not None:
            cuda_profiler.export_chrome_trace('torch_cuda_timeline.json')

    def _test_simulation_run(self):
        """
        Test network run. Estimates the average time used to perform a time/integration step.

        Returns:
            time average for prediction step, time average for training step
        """
        sample_size = 50.0  # number of samples to calculate the average integration time of

        sample_input_voltages = np.ones(self.input_size)  # input voltages set to 1
        sample_output_voltages = np.ones(self.neuron_qty)  # output voltages set to 1

        # test prediction
        init_time = time.time()  # initial time
        for i in range(int(sample_size)):
            self.update_network(sample_input_voltages, sample_output_voltages, False)  # run sample_size prediction updates of the network

        time_per_prediction_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        # test training
        init_time = time.time()
        for i in range(int(sample_size)):
            self.update_network(sample_input_voltages, sample_output_voltages, True)  # run sample_size training updates of the network

        time_per_train_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        return time_per_prediction_step, time_per_train_step

    def report_simulation_params(self):
        """
        Print report of simulation setup.
        """

        print('------------------')
        print('SUMMARY')
        print('------------------')
        print('Integration method:', self.integration_method.name)
        print('Total number neurons: ', self.neuron_qty)
        print('Total number syn. connections: ', int(torch.sum(self.weight_mask).item()))
        print('Layer structure: ', self.layers)
        print('Network architecture: ', self.params.network_architecture.name)
        print('Activation function: ', self.params.activation_function.name)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Bias initial distribution: ', self.arg_bias_init_params)
        print('Learning rate: ', self.arg_lrate_W)
        print('Beta (nudging parameter): ', self.beta)
        print('Membrane time constant (Tau): ', self.tau)
        print('Time step: ', self.dt)
        print('Time per prediction step in test run: ', self.time_per_prediction_step, 's')
        print('Time per training step in test run: ', self.time_per_train_step, 's')
        print('------------------')
        print("Simulation framework: Torch ", torch.__version__)
        print('Simulation running on :', self.device)
        print('------------------')

    # PERFORM UPDATE STEP OF NETWORK DYNAMICS

    def _perform_update_step(self, train_W):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        :math:`(2.0)\\: \\tau \\dot u = - u + W r +  e \\\\`
        :math:`(2.1)\\:             r = \\bar r + \\tau \\dot \\bar r \\\\`
        :math:`(2.2)\\:             e = \\bar e + \\tau \\dot \\bar e \\\\`

        :math:`(14.3)\\:      \\bar e  =  \\bar r' \\odot [ W^T  (u - W \\bar r)] + \\beta \\bar e_{trg} \\\\`
        :math:`(14.4)\\: \\bar e_{trg} = \\beta (u^{trg}(t) - u(t)) \\\\`

        Args:
            train_W: If forward weights should be trained.
        """

        r_bar, r_bar_deriv, r_bar_second_deriv, basal_input = self._get_activity_and_basal_input(self.st_voltages, self.st_weights, self.st_biases)  # get current activities and derivatives + basal input

        self._update_interneuron_voltages(r_bar)                                                            # update interneuron voltage either as dynamic interneurons or instantaneous interneurons
        # CALCULATE WEIGHT AND BIAS DERIVATIVES
        self._update_dot_weights(train_W, r_bar, r_bar_deriv, basal_input)      # calculate dot weight derivatives

        self._update_dot_bias(train_W, basal_input)                                        # calculate dot bias derivatives

        self._calculate_dot_voltages(r_bar, r_bar_deriv, r_bar_second_deriv, basal_input)                   # CALCULATE DOT VOLTAGE DERIVATIVES

        if self.integration_method == IntegrationMethod.EULER_METHOD or self.integration_method == IntegrationMethod.RICHARDSON_1ST_METHOD or self.integration_method == IntegrationMethod.RICHARDSON_2ND_METHOD:
            # calculate current somato-dendritic mismatch error, error lookaheads and derivative (observables)
            error, error_lookaheads, dot_errors_bar = self._calculate_mismatch_error(r_bar, r_bar_deriv, r_bar_second_deriv,
                                                                                     self.st_voltages, self.st_dot_voltages,
                                                                                     self.st_weights, self.st_biases)

            self._keep_values(error, error_lookaheads)                                                              # keep values of error and error lookaheads

            self._update_traces(self.st_dot_voltages, 10)                                                           # keep traces of dot voltages (for extrapolations)

            # UPDATE NETWORK STATE
        self._update_volts_weights()                                                                            # update volts and weights

        self._update_biases()                                                                                   # (if biases used) update biases

    # ALL METHODS USED TO PERFORM AN UPDATE STEP OF NETWORK DYNAMICS

    def _update_interneuron_voltages(self, r_bar):
        """
        Update interneuron voltages.

        if interneurons are dynamic:
        :math:`u^I(t) = u^I(t-1) + dt \\cdot \\dot u^I(t)`

        if interneurons are instantaneous:
        :math:`u^I(t) = W^{IP} \\bar r + b^I`

        Args:
            r_bar: activity/rate
        """
        # instantaneous inter-neurons get voltage from weighted activity
        self.st_voltages_I = self._calculate_basal_inputs(r_bar, self.st_weights, self.st_biases, self.use_input_neurons, self.input_size, self.st_inputs)  # i_voltage = activity * weight + bias (Eq. 33)

    def _update_dot_weights(self, train_W, r_bar, r_bar_deriv, basal_input):
        """
        Calculate weight derivatives.

        classical post-pre synaptic difference times the activities :math:`\\cdot \\eta` (weight_mask to keep existing connections)

        :math:`\\dot W = ((u - basal\\_input) \\otimes \\bar r(u)) \\cdot weight\_mask  \;(18)`

        For RtDeel:

        :math:`\\dot{B} = (\\bar r - B \, u) \,u^T`

        :math:`\\dot{W}^{PI} = ( B \, u - W^{PI} r^I ) \, (u^I)^T`

        :math:`\\dot{W}^{IP} = (u^I - W^{IP} \\bar r)\, \\bar r^T` or with weight alignment (as used here)

        Args:
            train_W: If pyramidal-pyramidal W is trained
            r_bar: activity/rate
            r_bar_deriv: derivative of activity/rate
            basal_input: Basal dendrite input from other neurons (defines somatic voltage)

        C: train_W: O(neuron_qty^3) (matrix-matrix multiplication), testing: O(neuron_qty^2) (matrix-vector multiplication)

        """
        # The weight update depends only on the current weights and voltages and can be updated first.
        if train_W:
            self.st_dot_weights = self._update_dot_weight(self.st_voltages, r_bar, basal_input)
            if self.arg_clip_weight_derivs:
                self.st_dot_weights = torch.clamp(self.st_dot_weights, -self.arg_weight_deriv_clip_value, self.arg_weight_deriv_clip_value)
        else:
            self.st_dot_weights = torch.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype, device=self.device)

        # calculate weight derivatives
        self.weights_r_bar_deriv = self._calculate_basal_input_deriv(self.st_weights, r_bar_deriv)  # get the weights multiplied by r bar', as written in Eq. 17.

    def _calculate_basal_input_deriv(self, weights, r_bar_deriv):
        if self.integration_method in {IntegrationMethod.EULER_METHOD, IntegrationMethod.RICHARDSON_1ST_METHOD,
                                       IntegrationMethod.RICHARDSON_2ND_METHOD}:
            return torch.mv(weights, r_bar_deriv)
        else:
            return weights * r_bar_deriv  # required for the linear equation solver methods

    def _update_dot_bias(self, train_W, basal_input):
        """
        Calculate the biases (:math;``bias, bias^{I}`) derivatives.

        :math:`\\dot b = (u(t) - W r) \cdot \eta_{biases} \cdot bias\\_mask`

        if interneurons:
        :math:`\\dot b^I = (W^B u(t) - W^{PI} u^I(t)) \cdot bias\\_mask \\cdot \\eta_{biases I}`

        Args:
            train_W: If pyramidal-pyramidal W is trained.
            basal_input: Basal inputs for each neuron (defines the somatic voltage).

        C: train_W: O(neuron_qty) (vector-scalar product), testing: O(1)

        """
        if self.use_biases:  # if we use biases
            if train_W:
                self.st_dot_biases = self._update_dot_biases(self.st_voltages, basal_input)  # do bias update
            else:
                self.st_dot_biases = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

    def _calculate_dot_voltages(self, r_bar, r_bar_deriv, r_bar_second_deriv,
                                basal_input):
        """
        Calculate voltage time derivative via:
        :math:`(2)\, \\dot u = 1 / \\tau \\cdot (-u + W \cdot r  + e + \\dot u_{input} )`

        Args:
            r_bar: activity/rate
            r_bar_deriv: derivative of activity/rate
            r_bar_second_deriv: 2nd derivative of activity/rate
            basal_input: basal dendrite input from other neurons (defines the somatic voltage)
        """
        # NUMERICAL METHODS
        if self.integration_method == IntegrationMethod.EULER_METHOD:
            # Calculate voltages deriv in an euler step using the st_voltages_deriv and st_weights_deriv from the previous update
            self.st_dot_voltages = self._calculate_dot_voltages_euler(r_bar, r_bar_deriv, r_bar_second_deriv,
                                                                      basal_input, self.st_dot_voltages)

        elif self.integration_method == IntegrationMethod.RICHARDSON_1ST_METHOD:
            extrapolated_dot_voltages = self._calculate_voltages_deriv_t_dt_richardson(self.dot_voltages_trace[-1], self.dot_voltages_trace[-2], self.dot_voltages_trace[-4], order=1)
            self.st_dot_voltages = self._calculate_dot_voltages_euler(r_bar, r_bar_deriv, r_bar_second_deriv,
                                                                      basal_input, extrapolated_dot_voltages)

        elif self.integration_method == IntegrationMethod.RICHARDSON_2ND_METHOD:
            extrapolated_dot_voltages = self._calculate_voltages_deriv_t_dt_richardson(self.dot_voltages_trace[-1], self.dot_voltages_trace[-2], self.dot_voltages_trace[-4], order=2)
            self.st_dot_voltages = self._calculate_dot_voltages_euler(r_bar, r_bar_deriv, r_bar_second_deriv,
                                                                      basal_input, extrapolated_dot_voltages)

        elif self.integration_method == IntegrationMethod.HESSIAN_SOLVER:
            self.st_dot_voltages = LagrangeNetwork._calculate_dot_voltages_hessian(self.st_voltages, r_bar, r_bar_deriv, r_bar_second_deriv, basal_input, self.st_inputs, self.st_old_inputs,
                                                                                   self.st_targets, self.st_old_targets, self.beta,
                                                                                   self.tau, self.dt, self.solve, self.st_weights, self.neuron_qty)

            # DEFAULT: LINEAR EQUATION SOLVERS
        else:
            self.st_dot_voltages = LagrangeNetwork._calculate_dot_voltages_cholesky(r_bar, r_bar_deriv, r_bar_second_deriv,
                                                                                    self.weights_r_bar_deriv, basal_input, self.st_voltages, self.st_voltages_I, self.layers[-1], self.tau, self.neuron_qty, self.beta, self.dt, self.st_inputs, self.st_old_inputs, self.input_size,
                                                                                    self.st_weights, self.use_input_neurons, self.st_targets, self.st_old_targets, self.st_dot_weights,
                                                                                    self.st_dot_biases, self.with_input_dynamics, self.solve)

    def _calculate_mismatch_error(self, r_bar, r_bar_deriv, r_bar_scnd_deriv,
                                  voltages, dot_voltages,
                                  weights, biases):
        """
        (OBSERVABLES) Calculate error and error lookaheads. Used as observables.

        Args:
            voltages:
            dot_voltages:
            weights:
            biases:

        C: O(neuron_qty^2) (matrix-vector multiplication)
        """
        # calculate basal inputs from other neurons for each neuron
        basal_inputs = self._calculate_basal_inputs(r_bar, weights, biases, self.use_input_neurons, self.input_size, self.st_inputs)  # (W_i * r) + u_input

        # calculate lookaheads
        r_lookaheads = self._get_r_lookaheads(r_bar, r_bar_deriv, dot_voltages)

        # calculate observables
        errors = LagrangeNetwork._get_bar_e(voltages, r_bar_deriv, basal_inputs, self.st_targets, self.st_weights, self.beta)

        dot_errors_bar = self._get_dot_bar_e(voltages,
                                             r_bar_deriv, r_bar_scnd_deriv,
                                             basal_inputs, dot_voltages)

        error_lookaheads = self._get_e_lookaheads(voltages,
                                                  r_bar_deriv, r_bar_scnd_deriv,
                                                  self.st_targets, basal_inputs, dot_voltages)

        return errors, error_lookaheads, dot_errors_bar

    def _keep_values(self, error, error_lookaheads):
        """
        Keep values from the previous run.

        Args:
            error: mismatch errors (observable)
            error_lookaheads: mismatch error lookaheads (observable)
        """
        self.obs_errors = error                       # mismatch errors (observable)
        self.obs_error_lookaheads = error_lookaheads  # mismatch error lookaheads (observable)

    def _update_traces(self, dot_voltages, window_size=10):
        """
        Update the value traces required for extrapolation.

        Args:
            dot_voltages: pyramidal neuron voltage time derivative
            window_size: number of values to keep
        """
        self.dot_voltages_trace.append(dot_voltages)             # voltage derivatives (used for Richardson exptrapolation!)
        self.dot_voltages_trace = self.dot_voltages_trace[-window_size:]

    def _update_volts_weights(self):
        """
        Update voltages and weights with the derivatives calculated earlier (euler integration).

        :math:`u(t) = u(t-1) + dt \\cdot \\dot u(t)`

        :math:`W = W + dt \\cdot \\dot W`

        C: O(neuron_qty) (vector-scalar product)

        """

        # EULER EQUATIONS
        #self.st_dot_voltages = torch.clamp(self.st_dot_voltages, -0.1, 0.1)
        self.st_voltages = self.st_voltages + self.dt * self.st_dot_voltages  # voltages += dt * voltages_deriv (EULER)
        self.st_weights = self.st_weights + self.dt * self.st_dot_weights  # weights += dt * weights_deriv (EULER)

        if self.arg_clip_weights:
            self.st_weights = torch.clamp(self.st_weights, -self.arg_w_init_params['clip'], self.arg_w_init_params['clip'])

    def _update_biases(self):
        """
        Update biases.

        :math:`b = b + dt \\cdot \\dot b`

        if interneurons:
        :math:`b^I = b^I + dt \\cdot \\dot b^I`

        else:
        :math:`b^I = b`
        """
        if self.use_biases:
            self.st_biases = self.st_biases + self.dt * self.st_dot_biases  # biases += dt * dot biases

            if self.arg_clip_weights:
                self.st_biases = torch.clamp(self.st_biases, -self.arg_bias_init_params['clip'], self.arg_bias_init_params['clip'])

    # FUNCTIONS OF CALCULATIONS NEEDED TO SOLVE ODE

    def _get_activity_and_basal_input(self, voltages, weights, biases):
        """
        Return neural activity + derivatives thereof as well as the basal input of each neuron.

        :math:`r(t), r'(t), r''(t), W \\bar r + b \; (\\text{"basal dendrite input"})`

        C: O(neuron_qty^2) (matrix-vector multiplication)

        Args:
            voltages: pyramidal neuron voltages
            weights: synaptic weights
            biases: neural biases

        Returns:
            r_bar, r_bar 1st and 2nd derivative, basal inputs

        """

        r_bar = self.act_function(voltages)                                  # neural activity
        r_bar_deriv = self.act_func_deriv(voltages)                          # derivative of neural activity
        r_bar_second_deriv = self.act_func_second_deriv(voltages)            # second derivative of neural activity

        basal_inputs = self._calculate_basal_inputs(r_bar, weights, biases, self.use_input_neurons, self.input_size, self.st_inputs)  # all the basal inputs to the all neurons
        return r_bar, r_bar_deriv, r_bar_second_deriv, basal_inputs

    def _set_input_and_target_volts(self, inputs, target_volts):
        """
        Set input and target of the network.
        Note: We keep the input of the previous time step to approximate the derivatives of the inputs.

        Args:
            inputs: The voltages/rates to which the input neurons should be clamped (depends on input neuron dynamics/feedback loops).
            target_volts: The voltages towards which the output voltages should be nudged.

        C: O(1)
        """
        # assert that input and output voltages are tensors
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=self.dtype, device=self.device)

        if isinstance(target_volts, np.ndarray):
            target_volts = torch.tensor(target_volts, dtype=self.dtype, device=self.device)

        if target_volts.shape[-1] != self.neuron_qty:
            target_volts = torch.cat([torch.zeros(self.neuron_qty-target_volts.shape[-1], dtype=self.dtype, device=self.device), target_volts], dim=0)

        if self.use_input_as_rates:
            inputs = self.inverse_act_func(inputs)

        # set (formerly) current as old and new input as current
        self.st_old_inputs = self.st_inputs
        self.st_inputs = inputs

        self.st_old_targets = self.st_targets
        self.st_targets = target_volts

    # DERIVATIVES FOR WEIGHT & BIAS UPDATE

    def _update_dot_weight(self, voltages, r_bar, basal_inputs):
        """
        Update of the network weight derivatives. Classical pre-post voltages * activities rule. See Eq. 18 in older manuscript.
        :math:`(18) \: \\dot W = ((u - u_{basal})  \\bar r^T) \\cdot weight\\_mask`

        C: O(voltage_qty^3) (matrix-matrix multiplication)

        Args:
            voltages: pyramidal neuron voltages
            r_bar: activity/rate of neurons
            basal_inputs: basal dendrite inputs of other neurons for each neuron (defines the somatic voltage)

        Returns:
            weight derivative based on voltages, r_bar and basal inputs
        """
        # multiply by weight_mask to keep only existing connections
        return torch.ger(voltages - basal_inputs, r_bar  # (voltages - layer input mismatch) * activities
                         ) * self.weight_mask * self.arg_lrate_W  # * weight mask * learning rate

    def _update_dot_biases(self, voltages, basal_inputs):
        """
        Update of the network bias time derivatives. Same as weights_update but without pre-synaptic activities (Eq. 18)

        :math:`(18) \: \\dot b = (u(t) - W r) \cdot \eta_{biases}`

        C: O(neuron_qty) (vector-scalar product)

        Args:
            voltages: pyramidal neuron voltages
            basal_inputs: basal dendrite input from other neurons for each neuron (defines somatic voltage)

        Returns:
            bias derivatives
        """
        return (voltages - basal_inputs) * self.arg_lrate_biases * self.bias_mask

    # SOLVE ODE WITH LINEAR EQUATION SOLVERS

    @staticmethod
    def _calculate_dot_voltages_hessian(voltages, r_bar, r_bar_deriv, r_bar_second_deriv, basal_input, input_voltage, old_input_voltage, target_voltage, old_target_voltage,
                                        beta, tau, dt, solve, weights, neuron_qty):
        """
        The original Euler-Lagrange Equation:

        :math:`(1 + \\tau \\frac{d}{dt}) f = 0`

        can be written as:

        :math:`\\tau (\\frac{\\partial f}{\\partial u} \\dot u + \\frac{\\partial f}{\\partial t}) = -f`

        We can transform this into:

        :math:`\\tau H(u) \\dot u = - f(u) - \\tau\\frac{\\partial f}{\\partial t}`

        :math:`f(u) = u - W \\bar r - \\bar e`

        :math:`H(u) = \\frac{\\partial f}{\\partial u} =  \\frac{\\partial^2 E}{\\partial^2 u}`

        Returns:

        """

        # calculate hessian matrix (partial^2 E / partial^2 u) of energy E
        hessian_matrix = LagrangeNetwork._calculate_2nd_order_u_deriv_of_energy(r_bar, r_bar_deriv, r_bar_second_deriv, voltages, weights,
                                                                                beta, neuron_qty)

        # calculate jacobian (partial E/ partial u) of energy E (also called f)
        jacobian_vector = LagrangeNetwork._calculate_1st_order_u_deriv_of_energy(r_bar, voltages, weights, r_bar_deriv, basal_input, target_voltage,
                                                                                 beta)
        # the unexplainables
        remainder_matrix = -r_bar_second_deriv * (weights.t() * input_voltage)
        remainder_vector = (-torch.mv(weights * r_bar_deriv,
                                      torch.cat((tau / dt * (input_voltage - old_input_voltage), input_voltage.new_zeros(neuron_qty - input_voltage.shape[0])), dim=0))
                            # - r_bar_deriv * weights.t() * (weights * r_bar- weights * r_bar) # seems irrelevant
                            + beta * tau * (target_voltage - old_target_voltage)/dt)  # beta lookahead
        # solve H(u) dot u = (- f(u) - tau partial f/partial t)/tau
        #print(hessian_matrix.cpu().detach().numpy())
        st_dot_voltages = solve(hessian_matrix + remainder_matrix, (-jacobian_vector - tau * -torch.diag(beta) - remainder_vector)/tau)

        return st_dot_voltages

    @staticmethod
    def _calculate_2nd_order_u_deriv_of_energy(r_bar, r_bar_deriv, r_bar_second_deriv, voltages, weights, beta, neuron_qty):
        """
        Calculate the 2nd order u derivative (hessain matrix) of energy E:

        :math:`\\frac{\\partial^2 E}{\\partial^2 u} = \\mathbb{1} - W diag(\\bar{r}') - (diag(\\bar{r}')  W^T - diag(\\bar{r}') W^T W diag(\\bar{r}') + diag(W' * (u - W \\bar{r}')))`

        This function is also called H(u) and returns the hessian matrix of E.

        Args:
            r_bar:
            r_bar_deriv:
            voltages:
            weights:
            beta:
            neuron_qty:

        Returns: the hessian matrix of E.

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = r_bar.dtype

        hessian = (torch.eye(neuron_qty, device=device, dtype=dtype)
                   - weights * r_bar_deriv
                   - (
                           (
                                   r_bar_deriv.reshape(-1, 1) * weights.t()
                                   - (r_bar_deriv.reshape(-1, 1) * torch.matmul(weights.t(), weights)) * r_bar_deriv
                           )
                           + torch.diag(torch.mv(weights.t(), (voltages - torch.mv(weights, r_bar))) * r_bar_second_deriv)
                   )
                   + torch.diag(beta)
                   )

        return hessian

    @staticmethod
    def _calculate_1st_order_u_deriv_of_energy(r_bar, voltages, weights, r_bar_deriv, basal_input, target_voltage, beta):
        """
        Calculate the 1st order u derivative (jacobian vector) of energy E:

        :math:`\\frac{\\partial E}{\\partial u} = u - W  \\bar{r} - r' \\odot (W^T (u - W \\bar{r})) - beta (u^{trg} - u)`

        This function is also called f and returns the jacobian vector of E.

        Args:
            r_bar:
            voltages:
            weights:
            r_bar_deriv:
            basal_input:
            target_voltage:
            beta:

        Returns: the jacobian vector of E.

        """
        jacobian = (voltages
                    - torch.mv(weights, r_bar)
                    - LagrangeNetwork._get_bar_e(voltages, r_bar_deriv, basal_input, target_voltage, weights, beta)
                    )
        return jacobian

    @staticmethod
    def _calculate_energy(weights, r_bar, beta, target_voltage, voltages):
        """
        Calculate the energy E of the model.

        :math:`\\frac{1}{2}(u - W \\bar{r})^T (u - W \\bar{r}) + \\frac{beta}{2} (u^{trg} - u)^T  (u^{trg} - u)`

        The energy level E defines two things:

        * The first term measures how well the neuron compartments match :math:`(u - W \\bar{r})`
        * the second term measures how well the output neurons match the voltage targets :math:`(u^{trg} - u)`

        The LagrangeNetwork tries to reduce the energy E by iterating the equation:
        :math:`(2) \\: \\tau \\dot u = - u + W r +  e`

        This is impossible as such as the right-hand side as well as the left-hand side contain :math:`\\dot{u}` (:math:`r = \\bar{r} + \\tau \\bar{r}' \\dot{u}` and :math:`e = \\bar{e} + \\tau \\bar{e}' \\dot{u}`)

        The hessian approach turns the equation into a linear equation :math:`\\tau H(u) \\dot{u} = -f(u) + \\frac{\\partial f}{\\partial t}`, which follows from the original Euler-Lagrange equation:
        :math:`(1 + \\tau \\frac{d}{dt}) f = 0` (@see: _calculate_dot_voltages_hessian)

        The euler-richardson approach replaces the :math:`\\dot{u}` on the right-hand side into :math:`\\dot{u}(u-dt)` and turns the equation into:

        :math:`\\: \\tau \\dot u = - u + W r +  e` with :math:`r = \\bar{r} + \\tau \\bar{r}' \\dot{u}(t-dt)` and :math:`e = \\bar{e} + \\tau \\bar{e}' \\dot{u}(t-dt)`

        Thus, the equation is iterable and can be solved for :math:`\\dot{u}`.

        Args:
            weights:
            r_bar:
            beta:
            target_voltage:
            voltages:

        Returns:

        """
        energy = (voltages - torch.mv(weights, r_bar)).dot(voltages - torch.mv(weights, r_bar)) / 2 + (beta.t() / 2).dot((target_voltage - voltages).t() * (target_voltage - voltages))
        return energy

    @staticmethod
    def _calculate_dot_voltages_cholesky(r_bar, r_bar_deriv, r_bar_second_deriv,
                                         weights_r_bar_deriv, basal_inputs, voltages, voltages_I, target_size, tau, neuron_qty, beta, dt, input_voltages, old_input_voltages, input_size,
                                         weights, use_input_neurons, target_voltages, old_target_voltages, dot_weights,
                                         dot_biases, with_input_dynamics, solve):
        """
        Calculate time derivative of voltages by reformulating Eq. 2 as a
        Rewrite Eq. 2 as linear equation set of form: :math:`\\dot u = F \\cdot m` and solve it with linear equation solver.

        TODO: It seems that cholesky has some incorrect parts. Compare to original Dodo implementation
        TODO: (CHOLESKY)Check math again and document it

        :math:`(2) \\: \\tau \\dot u = - u + W r +  e`

        Args:
            r_bar: activity/rate
            r_bar_deriv: first derivative of activity/rate
            r_bar_second_deriv: second derivative of activity/rate
            weights_r_bar_deriv: pyramidal weights times r bar derivative
            basal_inputs: basal dendrite inputs from other neurons for each neuron (defines somatic voltage)

        Returns:
            voltage derivative based on cholesky linear solver
        """
        # build the linear equations to calculate the voltage derivatives
        # get the dendritic error, which is the mismatch between prediction and reality
        dendritic_error = LagrangeNetwork._calculate_dendritic_error(voltages, weights, voltages_I)  # Eq. 24

        dot_voltages, err_vector, err_matrix, err_w_deriv = LagrangeNetwork._solve_linear_equations_for_volts_deriv(r_bar, r_bar_deriv, r_bar_second_deriv, weights, dot_weights,
                                                                                                                    weights_r_bar_deriv, dendritic_error, basal_inputs, voltages,
                                                                                                                    neuron_qty, input_size, input_voltages, old_input_voltages,
                                                                                                                    target_size, target_voltages, old_target_voltages,
                                                                                                                    with_input_dynamics, use_input_neurons, dt, beta, tau,
                                                                                                                    dot_biases, voltages_I, solve)

        return dot_voltages

    @staticmethod
    def _calculate_dendritic_error(voltages, weights, voltages_I):
        """
        Calculates difference between what we can explain away and what is left as error:
        :math:`(B * u - W_PI * W_IP * r) = B * u - W_PI * u^I = e.  r_i'` terms are already included

        TODO: (CHOLESKY) Check math again and document it

        Used for cholesky only.

        Args:
            voltages: pyramidal neuron voltages
            weights_B: backward weights

        Returns:
            dendritic error
        """
        return torch.mv(weights.t(), voltages) - torch.mv(weights.t(), voltages_I)  # back projected neuron output - inter-neuron output, st_voltages_I = W_IP.r_i according to Eq. 33

    @staticmethod
    def _solve_linear_equations_for_volts_deriv(r_bar, r_bar_deriv, r_bar_second_deriv, weights, dot_weights, weights_r_bar_deriv, dendritic_error, basal_input, voltages,
                                                neuron_qty, input_size, input_voltages, old_input_voltages, target_size, targets, old_targets, with_input_dynamics,
                                                use_input_neurons, dt, beta, tau, dot_biases, voltages_I, solve):

        """Solve a linear equation for u derivative for the case where the neural network uses look aheads.

        TODO: (CHOLESKY) Check math again and document it

        :math:`(2.0) \\tau \\dot u = - u + W r +  e`

        Since on the right-hand side of equation (14.0) :math:`\\dot{u}` appears only linearly,
        the ODE can be rewritten in the following form:

        .. math::
            (64) F \\dot{u} = m

        with matrix F and vector m that are independent of \dot{u}. F and m are given by:

        .. math::
            (65) \\frac{1}{\\tau} F = \\mathds{1} - B_{\\bar{r}^T} - W_{\\bar{r}} + W^{PI}_{\\bar{r}^T} W_{\\beta^I} - (\\bar{r}'' \\odot e_{A})\\mathds{1} + \\beta \\mathds{1}_{y}

            (66)                m = \\bar{r}' e_{A} - \\tau W^{PI}_{\\bar{r}^T} \\dot{\\bar{x}} + \\beta (y - u) + (W\\bar{r} + b + x - u) + m_{\\dot{W}}

            (67.0)              W_{\\bar{r}} = W \\odot\\, &\\bar{r}'

            (67.1)           W_{\\bar{r}^T} = (W^T \\odot \\bar{r}')^T

            (68)               W_{\\beta^I} = \\tilde{\\beta}^I W^{IP}_{\\bar{r}} + \\beta^I \\tilde{\\beta}^I M

            (69)                      e_{A} = Bu-W^{PI} u^\I

            (70) \\frac{1}{\\tau} m_{\\dot{W}} = \\dot{W}\\bar{r} + \\dot{b} + \\bar{r}'(\\dot{B} u - \\dot{W}^{PI} u^I - \\tilde{\\beta}^I W^{PI} \\dot{W}^{IP}\\bar{r} - \\tilde{\\beta}^I W^{PI} \\dot{b}^I )

        Unite the previous functions to build the torch computational graph and solve the ODEs governing
        the time dependence of the voltages and weights.

        Args:
            r_bar:
            r_bar_deriv:
            r_bar_second_deriv:

            weights_r_bar_deriv:
            dendritic_error:
            basal_inputs:

        Returns:
            voltage derivative, error_vector, error_matrix, error weight derivative
        """
        # We use the dendritic_error for the next computation, described by Eq. 17 (and further explanations below).
        # We obtain a matrix and error that we will use for solving the linear system.
        # get error matrix, being ..., and error vector, being ..

        err_matrix, err_vector = LagrangeNetwork._get_error_terms(voltages, r_bar_deriv, r_bar_second_deriv, dendritic_error, weights_r_bar_deriv, neuron_qty,
                                                                  input_size, input_voltages, old_input_voltages, target_size, targets, old_targets, tau, beta,
                                                                  dt)

        # get error_w_deriv, being ..., and input_w_deriv, being ...
        err_w_deriv, inp_w_deriv = LagrangeNetwork._get_weight_derivative_terms(voltages, r_bar, r_bar_deriv,
                                                                                dot_weights, weights,
                                                                                dot_biases,
                                                                                tau, voltages_I)

        # get input_matrix, being ..., and input vector, being ...
        basal_input_matrix, basal_input_vector = LagrangeNetwork._get_basal_input_terms(voltages, basal_input, weights_r_bar_deriv,
                                                                                        tau, neuron_qty, use_input_neurons, input_size, dt, input_voltages, old_input_voltages)

        # calculate the voltage derivatives
        # we have all information available to solve the equation F(u) * u_dot = b(u) for u_dot using a linear equation solver. Below we treat both cases
        # of having dynamics in the input neurons or without such dynamics

        # if the input is not meant to be dynamic, we clamped the input, i.e. r_1 = visible input activity
        if use_input_neurons and not with_input_dynamics:
            # we store the matrix M(u) supposed to be multiplied by u_dot in a big matrix
            full_matrix = err_matrix + basal_input_matrix

            full_vector = (err_vector + basal_input_vector + err_w_deriv + inp_w_deriv)[input_size:]  # b(u) with removed input_contribution

            # In the clamped input case, we do not need to compute u_dot since all components are already known without dynamics.
            # So we force the derivative of the input neurons to take the value of arg_input. We do that by writing the formal derivative for the clamped input.
            input_u_dot_derivative = (input_voltages - voltages[:input_size]) / dt

            # With the desired derivative known, we simply multiply the input_u_dot_derivative with the respective part of M(u) to find the contribution of the input to b(u).
            input_contribution = torch.mv(full_matrix[input_size:, :input_size], input_u_dot_derivative)
            full_vector -= input_contribution

            # We can now subtract this contribution from b(u) to remove the contribution from the linear system.
            # The remaining linear system can now be solved for the remaining, unknown u_dot. This gives us the complete solution consisting of the input u_dot derivative and the
            # solution of the remaining linear system explaining the contribution of the non-input neurons. The two parts can be concatenated to form the complete solution.
            remainder_solution = solve(
                full_matrix[input_size:, input_size:],  # F(u)
                full_vector  # b(u)
            )

            dot_voltages = torch.cat((input_u_dot_derivative, remainder_solution), dim=0)
        else:
            # here, the input neurons are nudged, so we do not know their dynamics. We then solve the whole system altogether.
            dot_voltages = solve(err_matrix + basal_input_matrix, err_vector + basal_input_vector + err_w_deriv + inp_w_deriv)

        return dot_voltages, err_vector, err_matrix, err_w_deriv

    # INDIVIDUAL PARTS OF THE LINEAR EQUATIONS

    @staticmethod
    def _get_error_terms(voltages, r_bar_deriv, r_bar_second_deriv, dendritic_error, weights_r_bar_deriv, neuron_qty, input_size, input_voltages,
                         old_input_voltages, target_size, target_voltages, old_target_voltages, tau, beta, dt):
        """
        Part of the linear equations.
        Calculates all terms originating from the error :math:`e = e_bar + tau*d(e_bar)/dt`.
        e can be recovered by calculating matrix_comp * voltage_derivative + vector_comp, i.e., e was divided into a part depending
        on the voltage derivatives and an independent part: :math:`F(u) * u_derivative + m(u)` (cf drawings)

        TODO: (CHOLESKY) Check math again and document it

        Args:
            voltages:

            r_bar_deriv:
            r_bar_second_deriv:

            weights_IP_r_bar_deriv:

            dendritic_error:

        Returns:
            :math:`F(u), m(u)`, which are part of the linear equation :math:`e_deriv = F(u) * u_deriv + m(u)`
        """
        error_matrix = -tau * (weights_r_bar_deriv.t() - torch.mm(weights_r_bar_deriv.t(), weights_r_bar_deriv))
        error_matrix += -tau * torch.diag(r_bar_second_deriv * dendritic_error)
        target_term = torch.eye(neuron_qty, dtype=error_matrix.dtype, device=error_matrix.device)
        target_term[:-target_size, :-target_size] = 0
        error_matrix += target_term * beta * tau

        error_vector = r_bar_deriv * dendritic_error
        deriv = torch.cat((tau / dt * (input_voltages - old_input_voltages), error_matrix.new_zeros(neuron_qty - input_size)), dim=0)
        error_vector += -torch.mv(weights_r_bar_deriv.t(), deriv)
        #is always 0 in RtDeep: # error_vector += -r_bar_deriv * torch.mv(weights.t(), LagrangeNetwork._calculate_basal_inputs(r_bar, weights, biases_I, use_input_neurons, input_size, input_voltages) - voltages_I)  # Eq. 17

        # TODO: Quick-fix to reset to old version, turn to without target_size in the future
        target_error_vector = error_vector[-target_size:] \
                              + beta[-target_size:] * (target_voltages[-target_size:] - voltages[-target_size:]) \
                              + tau * beta[-target_size:] * (target_voltages[-target_size:] - old_target_voltages[-target_size:]) / dt

        return error_matrix, torch.cat((error_vector[:-target_size], target_error_vector), dim=0)

    @staticmethod
    def _get_weight_derivative_terms(voltages, r_bar, r_bar_deriv, dot_weights, weights, dot_biases, tau, voltages_I):
        """
        Part of the linear equations.
        Terms originating from taking the derivatives of the weights. Completely in vector form (i.e., independent of the derivative of the voltages).

        TODO: The weight deriv dot W is calculated, so Dodo calculates dot W for the error.
        TODO: (CHOLESKY) Check math again and document it

        Args:
            voltages:
            r_bar:
            r_bar_deriv:
            dot_weights:
            weights_IP_deriv:
            weights_PI:
            dot_biases:

        Returns:
            error and input vector contribution
        """
        error_w_deriv = tau * r_bar_deriv * (torch.mv(dot_weights.t(), voltages) - torch.mv(weights.t(), torch.mv(dot_weights, r_bar)))
        error_w_deriv += -tau * r_bar_deriv * (torch.mv(dot_weights.t(), voltages_I) + torch.mv(weights.t(), dot_biases))
        input_w_deriv = tau * (torch.mv(dot_weights, r_bar) + dot_biases)

        return error_w_deriv, input_w_deriv

    @staticmethod
    def _get_basal_input_terms(voltages, basal_inputs, weights_r_bar_deriv, tau, neuron_qty, use_input_neurons, input_size, dt, input_voltages, old_input_voltages):
        """
        Part of the linear equations.
        Terms originating from external and synaptic inputs.

        TODO: (CHOLESKY) Check math again and document it

        Args:
            voltages:
            basal_inputs:
            weights_r_bar_deriv:

        Returns:
            external and synaptic inputs
        """
        matrix_comp = tau * (torch.eye(neuron_qty, dtype=voltages.dtype, device=voltages.device) - weights_r_bar_deriv)
        vector_comp = basal_inputs - voltages

        # prepare batch of external layer inputs
        if use_input_neurons:
            vector_comp[:input_size] = vector_comp[:input_size] + tau / dt * (input_voltages - old_input_voltages)

        return matrix_comp, vector_comp

    # SOLVE ODE WITH u(t) ~ u(t-1) for small dt

    def _calculate_dot_voltages_euler(self, r_bar, r_bar_deriv, r_bar_scnd_deriv, layerwise_inputs, previous_dot_voltages):
        """
        (RtDeep style)
        TODO: RtDeel version missing
        Calculate time derivative of voltages using an euler update:

        TODO: Pyramidal input voltage insertion for correct dot voltages

        :math:`(2) \\dot u = 1 / \\tau \\cdot (-u + W r + \\dot u_input  + e )`

        This method assumes that the equation can be solved by using the former :previous_dot_voltages: dot u (t-1) as a slightly disturbed version of dot u (t).
        For smaller :math:`dt (< 1.0 ms = 0.1 \\cdot \\tau)`, this seems to hold.

        Args:
            r_bar: rates/activity of pyramidal neuron
            r_bar_deriv: first derivative of rate/activity of pyramidal neuron
            r_bar_scnd_deriv: second derivative of rate/activity of pyramidal neuron
            layerwise_inputs:
            previous_dot_voltages:

        Returns:
            voltage derivative based on euler method
        C: O(neuron_qty^2) (matrix-vector multiplication)

        """
        # calculate r lookahead (r) = bar r + tau * dot_voltages
        r_lookaheads = self._get_r_lookaheads(r_bar, r_bar_deriv, previous_dot_voltages)

        #  calculate W * r + b
        basal_input_lookaheads = self._get_basal_input_lookaheads(r_lookaheads, self.st_weights, self.st_biases)

        # calculate error lookahead (e_i)
        error_lookaheads = self._get_e_lookaheads(self.st_voltages, r_bar_deriv, r_bar_scnd_deriv,
                                                  self.st_targets, layerwise_inputs, previous_dot_voltages)

        # calculate derivative of voltages (dot(u)_i = 1/tau * (W_i * r - u_i + e_i + dot(u_input)))
        u_deriv = 1.0 / self.tau * (- self.st_voltages + basal_input_lookaheads + error_lookaheads)

        if self.use_input_neurons and not self.with_input_dynamics:
            # TODO: Allow convex combination of input and dynamics
            # In the clamped input case, we do not need to compute u_dot since all components are already known without dynamics.
            # So we force the derivative of the input neurons to take the value of :self.st_input_volts:. We do that by writing the formal derivative for the clamped input.
            u_deriv[:self.input_size] = u_deriv[:self.input_size] + (self.st_inputs - self.st_voltages[:self.input_size]) / self.dt

        return u_deriv

    @staticmethod
    def _calculate_voltages_deriv_t_dt_richardson(dot_voltages_t_dt, dot_voltages_t_2dt, dot_voltages_t_4dt, order=2):
        """
        Calculate extrapolated voltage deriv (t) from previous dts.

        Args:
            dot_voltages_t_dt: time derivative of pyramidal neuron voltages from t - dt
            dot_voltages_t_2dt: time derivative of pyramidal neuron voltages from t - 2dt
            dot_voltages_t_4dt: time derivative of pyramidal neuron voltages from t - 4dt
            order: order of richardson extrapolation, 0 is euler, 1 is richardson 1st order, 2 is richardson 2nd order

        Returns:
            extrapolated voltage derivative :math:`u_(t-dt)` based on :math:`u_(t-2dt)` and :math:`u_(t-4dt)`
        C: O(neuron_qty) (vector-scalar product)

        """
        if order == 1:
            voltages_deriv_t = 2 * dot_voltages_t_dt - dot_voltages_t_2dt
        elif order == 2:
            voltages_deriv_t = (8 * dot_voltages_t_dt - 6 * dot_voltages_t_2dt + dot_voltages_t_4dt) / 3
        else:  # fallback to euler
            voltages_deriv_t = dot_voltages_t_dt

        return voltages_deriv_t

    def _get_r_lookaheads(self, r_bar, r_bar_deriv, dot_voltages):
        """
        Calculate r lookaheads.
        Get :math:`r = \\bar r + \\tau \\dot \\bar`
        via :math:`\\dot \\bar r = \\bar r'(u) \\cdot \\dot u`

        Args:
            r_bar: :math:`\\overline{r_i)
            r_bar_deriv: :math:`\\bar r'`
            dot_voltages: :math:`\\dot u`

        Returns:
            r lookaheads

        C: O(neuron_qty) (vector product)

        """
        r_dot_bar = r_bar_deriv * dot_voltages  # \\dot \\bar r = \\bar r' * \\dot u
        return r_bar + self.tau * r_dot_bar

    @staticmethod
    def _calculate_basal_inputs(r_bar, weights, biases, use_input_neurons, input_size, input_voltages):
        """
        Returns external+network input voltage for all neurons:
        :math:`W \\bar r + [0 \\dots 0, u_{input}, 0 \\dots 0]`

        C: O(neuron_qty^2) (matrix-vector multiplication)

        Args:
            r_bar:
            weights:
            biases:

        Returns:
            layerwise inputs
        """
        basal_input = torch.mv(weights, r_bar) + biases  # activities * network_weights + biases
        if use_input_neurons:
            basal_input[:input_size] = basal_input[:input_size] + input_voltages  # internal input + external input = total input
        return basal_input

    def _get_basal_input_lookaheads(self, r_lookahead, weights, biases):
        """
        Calculate prospective basal inputs to each neuron by making an euler step with the derivative, thus looking ahead. The prospectivity adjusts for network signalling delay.
        Returns external + network lookahead input for all neurons:
        :math:`W  r + u_{input} + \\tau \\dot u_{input}`

        Args:
            r_lookahead:
            weights:
            biases:

        Returns:
            input lookaheads
        C: O(neuron_qty^2) (matrix-vector multiplication)
        """
        basal_input = torch.mv(weights, r_lookahead) + biases  # lookahead_activities * network_weights + biases

        if self.use_input_neurons:
            u_input_lookahead = self.st_inputs + self.tau / self.dt * (self.st_inputs - self.st_old_inputs)  # for basal input lookahead, we need the input lookahead
            basal_input[:self.input_size] = basal_input[:self.input_size] + u_input_lookahead  # internal input + external input = total input
        return basal_input

    def _get_e_lookaheads(self, voltages,
                          r_bar_deriv, r_bar_scnd_deriv,
                          target_voltage, basal_input, dot_voltages):
        """
        Calculate :math:` (2.1) \: e = \\bar e  + \\tau \cdot \\dot \\bar e`.

        Args:
            voltages: pyramidal voltages

            r_bar_deriv: first derivative of rate/activity
            r_bar_scnd_deriv: second derivative of rate/activity

            target_voltage: target voltage of output neurons
            basal_input: basal dendrite input from other neurons for each neuron (defines the somatic voltage)
            dot_voltages: time derivative of pyramidal neuron voltages

        Returns:
            somato-dendritic mismatch error lookaheads

        C: O(neuron_qty^2) (matrix-vector multiplication)

        """
        # bar(e)
        bar_e = LagrangeNetwork._get_bar_e(voltages, r_bar_deriv, basal_input, target_voltage, self.st_weights, self.beta)

        # e = bar e + tau * dot(bar(e))
        e = bar_e + self.tau * self._get_dot_bar_e(voltages, r_bar_deriv, r_bar_scnd_deriv, basal_input, dot_voltages)

        return e

    # INDIVIDUAL PARTS OF THE ODE

    @staticmethod
    def _get_bar_e(voltages, r_bar_deriv, basal_input, target_voltage, weights, beta):
        """
        Calculate :math:`\\bar e` (without :math:`e_trg`).

        :math:`\\bar e` has two representations:

        :math:`(2.1) \: \\bar e = u - W  \\bar r`

        :math:`(2.3)\; \\bar e  =  \\bar r' \\odot W^T \\cdot \\bar e + \\beta \\cdot \\bar e^{trg}`

        Here we first calculate :math:`\\bar e` with the first to evaluate the second one.

        Args:
            voltages: pyramidal neuron voltages
            r_bar_deriv: first derivative of activities/rates
            basal_input: basal dendrite input from all other neurons for each neuron (defines somatic voltage)
            target_voltage: the target voltages of the output neurons

        Returns:
            bar e
        C: O(neuron_qty^2) (matrix-vector multiplication)

        """
        bar_e = voltages - basal_input

        # bar(e_i) (missing e_{trg})
        bar_e2 = r_bar_deriv * torch.mv(weights.t(), bar_e)

        # + e_{trg)
        bar_e2 = bar_e2 + beta.t() * LagrangeNetwork._get_bar_e_trg(target_voltage, voltages)

        return bar_e2

    @staticmethod
    def _get_bar_e_trg(target_voltage, voltages):
        """
        Calculate e target.

        Inserted into errors as err += get_e_trg(...).

        Args:
            voltages: pyramidal neuron voltages
            target_voltage: target voltage of output neurons

        Returns:
            e_target

        C: O(neuron_qty) (vector-scalar product)

        """
        return target_voltage - voltages

    def _get_dot_bar_e(self, voltages,
                       r_bar_deriv, r_bar_scnd_deriv,
                       basal_inputs,
                       dot_voltages):
        """
         Calculate dot(bar(e_i).

         :math:`\\dot{\\bar{e}} = \\bar r'' \\dot u W^T (u - W \\bar r)
                                 + \\bar r' W^T (\\dot u - W \\bar r’ \\dot u)
                                 - \\beta \\dot{\\bar{e}}_{trg} \\`

         Args:
             voltages: pyramidal neuron voltages
             r_bar_deriv: first derivative of rates/activities
             r_bar_scnd_deriv: second derivatives of rates/activities
             basal_inputs: basal dendrite inputs from other neurons for each neuron (defines the somatic voltage)
             dot_voltages: time derivative of voltages

         Returns:
             dot(bar(e_i))

        C: O(neuron_qty^2) (matrix-vector multiplication)

         """
        dot_bar_e = r_bar_scnd_deriv * dot_voltages * torch.mv(self.st_weights.t(), (voltages - basal_inputs))
        dot_bar_e += r_bar_deriv * torch.matmul(self.st_weights.t(), (dot_voltages - self.weights_r_bar_deriv * dot_voltages))
        dot_voltages_trg = (self.st_targets - self.st_old_targets) / self.dt
        dot_bar_e = self.beta * (dot_voltages_trg - dot_voltages)
        return dot_bar_e

    # NETWORK INTEGRATOR

    def update_network(self, inputs, target_voltage, train_W=False):
        """
        Perform a single integration step of the neural network. The network's input neurons will be clamped towards the input voltages,
        while the output neurons will be nudged towards the target voltages. If any of the :train_W:. :train_B: or :train_PI: flags are activated,
        the network will adjust the respective synaptic connections to minimize the error.

        Args:
            inputs: The input towards which the input neurons are clamped.
            target_voltage: The output voltage towards which the output neurons are nudged.
            train_W: If the pyr-pyr weights are trained.

        """
        self._set_input_and_target_volts(inputs, target_voltage)
        self._perform_update_step(train_W)

    # GETTERS (OBSERVABLES) and SETTERS

    def set_beta(self, beta):
        """
        Set the nudging beta.

        :math:`\\beta`

        Args:
            beta:

        """

        # keep beta state for quick checking
        if np.isscalar(beta) and beta == 0 or np.all(beta) == 0:
            self.is_beta_zero = True
        else:
            self.is_beta_zero = False

        if np.isscalar(beta):
            beta = np.concatenate((np.zeros(sum(self.layers[:-1])), np.ones(self.layers[-1]) * beta), axis=0) # beta is a vector of neuron_qty size with zeros except at the last layer of neurons

        if beta.shape[0] != self.neuron_qty:
            beta = np.concatenate((np.zeros(self.neuron_qty-beta.shape[0]), self.beta), axis=0) # beta is a vector of neuron_qty size with zeros except at the target neurons

        self.beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
        self.params.arg_beta = beta

    def get_beta(self):
        """
        Get nudging beta of network.

        :math:`\\beta`

        Returns:
            Nudging beta
        """
        return self.beta.detach().cpu().numpy()

    def set_lrate_W(self, learning_rate):
        """
        Set weight learning rate.

        :math:`\\eta`

        Args:
            learning_rate: weight learning rate.
        """
        self.arg_lrate_W = learning_rate
        self.params.arg_lrate_W = learning_rate

    def set_lrate_biases(self, learning_rate):
        """
        Set bias learning rate.

        :math:`\\eta_{biases}`

        Args:
            learning_rate: bias learning rate.
        """
        self.arg_lrate_biases = learning_rate
        self.params.arg_lrate_biases = learning_rate

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.

        Args:
            rnd_seed: fixed random seed for all involved frameworks.
        """
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

    def get_voltages(self):
        """
        Get network pyramidal neuron voltages.

        :math:`u(t)`

        Returns:
            network pyramidal neuron voltages

        """
        return self.st_voltages.detach().cpu().numpy()

    def set_voltages(self, new_voltages):
        """
        Set network pyramidal neuron voltages.

        :math:`u(t)`

        Args:
            new_voltages: new network pyramidal neuron voltages.
        """
        if self.st_voltages.shape != new_voltages.shape:
            raise Exception(f'Incompatible voltage shapes {self.st_voltages.shape} {new_voltages.shape}')

        if isinstance(new_voltages, np.ndarray):
            new_voltages = torch.tensor(new_voltages, dtype=self.dtype, device=self.device)

        self.st_voltages = new_voltages

    def get_output_voltages(self):
        """
        Get the voltages of the output neurons.

        :math:`u(t)_{output}`

        Returns:
            output neuron voltages
        """
        return self.st_voltages[-self.layers[-1]:].detach().cpu().numpy()

    def get_dot_voltages(self):
        """
        Get network pyramidal neuron voltage time derivatives.

        :math:`\\dot u(t)`

        Returns:
            network pyramidal neuron voltage derivatives
        """
        return self.st_dot_voltages.detach().cpu().numpy()

    def get_errors(self):
        """
        Get error resulting from somato-dendritic mismatch between pyramidal neuron compartment voltages.

        :math:`\\bar e`

        Returns:
            pyramidal neuron compartment voltage mismatch error

        """
        return self.obs_errors.detach().cpu().numpy()

    def get_error_lookaheads(self):
        """
        Get somato-dendritic mismatch error lookaheads.

        :math:`e`

        Returns:
            somato-dendritic mismatch error lookaheads
        """
        return self.obs_error_lookaheads.detach().cpu().numpy()

    def get_weights(self):
        """
        Get network synaptic weights between pyramidal neurons.

        :math:`W`

        Returns:
            pyramidal neuron synaptic weights
        """
        return self.st_weights.detach().cpu().numpy()

    def set_weights(self, new_weights):
        """
        Set network synaptic weights between pyramidal neurons.

        :math:`W`

        Args:
            new_weights: The synaptic weights
        """
        if self.st_weights.shape != new_weights.shape:
            raise Exception(f'Incompatible weights shapes {self.st_weights.shape} {new_weights.shape}')

        if isinstance(new_weights, np.ndarray):
            new_weights = torch.tensor(new_weights, dtype=self.dtype, device=self.device)

        self.st_weights = new_weights

    def get_weight_derivatives(self):
        """
        Get network synaptic weight derivatives.

        :math:`\\dot W`

        Returns:
            network synaptic weight derivatives

        """
        return self.st_dot_weights.detach().cpu().numpy()

    def get_biases(self):
        """
        Get biases of pyramidal neurons.

        :math:`b`

        Returns:
            biases of pyramidal neurons.

        """
        return self.st_biases.detach().cpu().numpy()

    def set_biases(self, new_biases):
        """
        Set biases of pyramidal neurons.

        :math:`b`

        Args:
            new_biases: new biases of pyramidal neurons
        """

        if self.st_biases.shape != new_biases.shape:
            raise Exception(f'Incompatible biases shapes {self.st_biases.shape} {new_biases.shape}')

        if isinstance(new_biases, np.ndarray):
            new_biases = torch.tensor(new_biases, dtype=self.dtype, device=self.device)

        self.st_biases = new_biases

    def set_integration_method(self, integration_method):
        """
        Set integration method for network.

        Args:
            integration_method: Any of the :LagrangeNetwork.IntegrationMethod:s.
        """
        self.set_solver(integration_method)
        self.integration_method = integration_method
        self.params.integration_method = integration_method

    def set_solver(self, solver_type):
        """
        Set either Cholesky solver (faster, but might fail due to numerical issues) or LU solver (slower, but works always).

        Args:
            solver_type: Either one of :LangrangeNetwork.IntegrationMethod:
        """

        if solver_type == IntegrationMethod.CHOLESKY_SOLVER and self.integration_method != IntegrationMethod.CHOLESKY_SOLVER:
            self.solve = lambda matrix, vec: torch.cholesky_solve(vec.expand_as(matrix).t(), torch.cholesky(matrix))[:, 0]  # use cholesky decomposition and its solver to solve Ax = b
        elif solver_type == IntegrationMethod.LU_SOLVER and self.integration_method != IntegrationMethod.LU_SOLVER:
            self.solve = lambda matrix, vec: torch.solve(vec.expand_as(matrix).t(), matrix)[0][:, 0]  # use LU decomposition solver to solve Ax = b
        elif solver_type == IntegrationMethod.HESSIAN_SOLVER and self.integration_method != IntegrationMethod.HESSIAN_SOLVER:
            self.solve = lambda matrix, vec: torch.solve(vec.expand_as(matrix).t(), matrix)[0][:, 0]  # use LU decomposition solver to solve Ax = b

            # KERAS-like TRAINING AND PREDICTION INTERFACE

    def fit(self, x=None, y=None, n_updates: int = 100, epochs=1, verbose=1, is_timeseries=False, **kwargs):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per sample.
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
            kwargs: catch batch params in to simplify switching between this and the batch version.
        """
        n_samples = len(x)  # dataset size

        if self.is_beta_zero:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(0.1))
            self.set_beta(0.1)  # turn nudging on to enable training

        print("Learning with single samples")

        dataset = SimpleDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)

        for epoch_i in range(epochs):
            for sample_i, (x, y) in enumerate(data_loader):
                x = x[0].to(self.device)  # put first and only sample to device
                y = y[0].to(self.device)
                if sample_i == 0:
                    self.old_decay_x = x
                    self.old_decay_y = y

                if verbose >= 1:
                    print("train:: sample ", sample_i, "/", n_samples, " | update ", end=" ")

                if self.decay_steps.shape[0] != n_updates:  # update transients input vector to number of updates
                    self.decay_steps = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

                for update_i in range(n_updates):
                    if verbose >= 2 and update_i % 10 == 0:
                        print(update_i, end=" ")

                    if is_timeseries:
                        sample, label = x, y
                    else:
                        sample, label = self._decay_func(self.decay_steps[update_i], self.old_decay_x, x), self._decay_func(self.decay_steps[update_i], self.old_decay_y, y)
                    self.update_network(sample, label, train_W=True)

                self.old_decay_x = x
                self.old_decay_y = y

                if verbose >= 1:
                    print('')

    def predict(self, x, n_updates: int = 100, verbose=1, is_timeseries=False, **kwargs):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
            kwargs: catch batch params in to simplify switching between this and the batch version.

        Returns:
            output neuron voltages for the input :x:
        """
        n_samples = len(x)  # dataset size
        self.set_beta(0.0)   # turn nudging off to disable learning

        dataset = SimpleDataset(x, [np.zeros(self.neuron_qty) for _ in range(n_samples)])
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)

        predictions = []
        for sample_i, (x, _) in enumerate(data_loader):
            x = x[0].to(self.device)  # put first and only sample to device
            if sample_i == 0:
                self.old_decay_x = x

            if verbose >= 1:
                print("predict:: sample", sample_i, "/", n_samples, " | update ", end=" ")

            if self.decay_steps.shape[0] != n_updates:  # update transients input vector to number of updates
                self.decay_steps = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

            for update_i in range(n_updates):
                if verbose >= 2 and update_i % 10 == 0:
                    print(update_i, end=" ")

                if is_timeseries:
                    sample = x
                else:
                    sample = self._decay_func(self.decay_steps[update_i], self.old_decay_x, x)

                self.update_network(sample, self.dummy_label, train_W=False)

            self.old_decay_x = x

            volts = self.st_voltages.detach().cpu().numpy()
            prediction = volts[-self.layers[-1]:]
            predictions.append(prediction)

            if verbose >= 1:
                print('')

        return predictions

    def __call__(self, x, n_updates: int = 100, verbose=1, is_timeseries=False):
        self.predict(x, n_updates, verbose, is_timeseries)

    # SAVE AND LOAD NETWORK

    def save(self, save_path):
        """
        Save the lagrange model parameters to file.

        Args:
            save_path: The path to save the network model to.

        TODO: Save and load the configuration parameters as well.
        """
        voltages = self.st_voltages.detach().cpu().numpy()
        weights = self.st_weights.detach().cpu().numpy()
        biases = self.st_biases.detach().cpu().numpy()

        np.save('{0}/voltages'.format(save_path), voltages)
        np.save('{0}/weights'.format(save_path), weights)
        np.save('{0}/biases'.format(save_path), biases)

    def load(self, load_path):
        """
        Load the lagrange model parameters from file.

        Args:
            load_path: The path to load the network model from.

        Returns:
            True if loaded successfully, else False.

        TODO: Save and load the config parameters as well.
        """
        try:
            voltages = np.load('{0}/voltages.npy'.format(load_path))
            weights = np.load('{0}/weights.npy'.format(load_path))
            biases = np.load('{0}/biases.npy'.format(load_path))
        except Exception:
            return False

        self.st_voltages = torch.tensor(voltages, device=self.device)
        self.st_weights = torch.tensor(weights, device=self.device)
        self.st_biases = torch.tensor(biases, device=self.device)

        return True

    def deepcopy(self):
        """
        Generate a deep copy of the network.

        Returns:
            the deepcopy of the network
        """
        n = LagrangeNetwork(self.params)
        n.set_weights(self.get_weights())
        n.set_biases(self.get_biases())
        n.set_voltages(self.get_voltages())

        return n

    # TIME SERIES BUILDER FUNCTIONS

    @staticmethod
    def _decay_func(fading_parameter, fade_from_sample, fade_to_sample):
        """
        Decay function to transform sample and label by exponential fading.

        Args:
            fading_parameter: Fading parameter to determine fading progress, out of np.linspace(-20, 90, n_updates)
            fade_from_sample: The sample to start fading from.
            fade_to_sample: The sample to fade to.

        Returns:
            Exponentially faded sample between :fade_from_sample: and :fade_to_sample:.
        """
        return fade_from_sample + (fade_to_sample - fade_from_sample) / (1 + torch.exp(-fading_parameter / 4.0))

