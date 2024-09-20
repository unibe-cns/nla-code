# Pytorch implementation of the Lagrange model. Batch-trainable version. Used in the framework agnostic model lagrange_model.py
# In order to train the network with batches, this version makes heavy use of batched matrix notations. For better understanding, the reader is advised to read
# the model.lagrange_model_torch.py version, which only supports training with single samples.
#
# Authors: Benjamin Ellenberger (benelot@github)


import numpy as np
import time
import torch
import torch.nn as nn

# network parameters
from model.network_params import ArchType, IntegrationMethod, ActivationFunction, NetworkParams

# utils
from torch.utils.data import DataLoader
from utils.torch_bottleneck import bottleneck
from utils.torch_utils import get_torch_dtype, SimpleDataset
from tqdm import tqdm


class LagrangeNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Lagrange model.
    Prefixes: st = state, pl = placeholder, obs = observable, arg = argument.
    """

    # INIT
    ########
    def __init__(self, params: NetworkParams):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # find device for torch

        # store important params
        self.layers = params.layers
        self.tau = params.arg_tau  # ms
        self.dt = params.arg_dt  # ms

        # learning rates
        self.arg_lrate_W = params.arg_lrate_W
        self.arg_lrate_B = params.arg_lrate_W_B
        self.arg_lrate_PI = params.arg_lrate_PI

        self.arg_lrate_biases = params.arg_lrate_biases
        self.arg_lrate_biases_I = params.arg_lrate_biases_I

        # weight and bias parameters
        self.arg_w_init_params = params.arg_w_init_params
        self.arg_bias_init_params = params.arg_bias_init_params
        self.arg_clip_weights = params.arg_clip_weight
        self.arg_clip_weight_derivs = params.arg_clip_weight_deriv
        self.arg_weight_deriv_clip_value = params.arg_weight_deriv_clip_value

        # network parameters
        self.dtype = get_torch_dtype(params.dtype)
        self.network_architecture = params.network_architecture
        self.use_interneurons = params.use_interneurons
        self.activation_function = params.activation_function
        self.rnd_seed = params.rnd_seed
        self._set_random_seed(params.rnd_seed)
        self.set_beta(params.arg_beta)

        # extra configs for computational graph
        self.dynamic_interneurons = params.dynamic_interneurons
        self.use_biases = params.use_biases
        self.turn_off_visible_dynamics = params.with_input_dynamics

        # setup network inputs and outputs
        self.input_size = params.layers[0]  # input neurons
        self.target_size = params.layers[-1]  # target neurons
        self.neuron_qty = sum(params.layers)
        self.st_input_voltage = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.st_old_input_voltage = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.st_target_voltage = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)
        self.st_old_target_voltage = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)

        # prepare input vector to generate input transients
        self.xr = torch.linspace(-20, 90, 100, dtype=self.dtype, device=self.device)
        self.old_x = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.old_y = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)
        self.dummy_label = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)

        # integration methods supported by this implementation
        self.INTEGRATION_METHODS = [IntegrationMethod.CHOLESKY_SOLVER,                  # linear equation solver based on a cholesky-decomposition
                                    IntegrationMethod.LU_SOLVER,                        # linear equation solver based on a LU-decomposition
                                    IntegrationMethod.EULER_METHOD,                     # lagrange framework implementing euler method
                                    IntegrationMethod.RICHARDSON_1ST_METHOD,            # lagrange framework implementing euler method, improved with a 1st order extrapolation
                                    IntegrationMethod.RICHARDSON_2ND_METHOD,            # lagrange framework implementing euler method, improved with a 1st order extrapolation
                                    IntegrationMethod.COMPARE_CHOLESKY_SOLVER,          # INTEGRATE with cholesky-solver, COMPARE to all others
                                    IntegrationMethod.COMPARE_EULER_METHOD,             # INTEGRATE with euler method, COMPARE to all others
                                    IntegrationMethod.COMPARE_RICHARDSON_1ST_METHOD,    # INTEGRATE with richardson 1st order method, COMPARE to all others
                                    IntegrationMethod.COMPARE_RICHARDSON_2ND_METHOD]    # INTEGRATE with richardson 2nd order method, COMPARE to all others

        # setup neuron weight and bias masks
        self.weight_mask = self._make_connection_weight_mask(params.layers, params.learning_rate_factors, params.network_architecture, params.only_discriminative)
        self.bias_mask = self._make_bias_mask(params.layers, self.input_size, params.learning_rate_factors, params.only_discriminative)

        # setup activation function and solver
        self.act_function, self.act_func_deriv, self.act_func_second_deriv = self._generate_activation_function(params.activation_function)
        self.integration_method = params.integration_method
        self.solve = self._set_solver(params.integration_method)  # # we need a solver to find the solution u_derivs from the linear ODEs of the form M * u_deriv = b(u)
        #                                                             Source: https://en.wikipedia.org/wiki/Cholesky_decomposition#Applications
        #                   instead of using a solver, we can just use pure forward euler, which works if the weights and biases stay in a certain range

        # initialize network
        self._create_observables()  # setup observable outputs (for observing network state)
        self._initialize_network(params.arg_w_init_params, params.use_biases, params.arg_bias_init_params, params.use_interneurons, params.arg_interneuron_b_scale)

        # perform basic profiling
        if params.check_with_profiler:
            print("\nProfile model for single sample:\n--------------------------------")
            self._profile_model(1)

            print("\nProfile model for batch size 32:\n--------------------------------")
            self._profile_model(32)

        # perform single sample test run to get network stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # perform mini-batch test run to get network stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run(32)
        self.report_simulation_params()

        # reinitialize network variables
        self._initialize_network(params.arg_w_init_params, params.use_biases, params.arg_bias_init_params, params.use_interneurons, params.arg_interneuron_b_scale)

    # NETWORK SETUP & INITIALIZATION METHODS
    ########################################

    def _make_connection_weight_mask(self, layers, learning_rate_factors, network_architecture, only_discriminative=True):
        """
        Create weight mask encoding of the network structure.
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.
        """
        if network_architecture == ArchType.LAYERED_FEEDFORWARD:  # proper feed forward mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)  # # make a feed forward mask

        elif network_architecture == ArchType.LAYERED_FEEDBACKWARD:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask = weight_mask.T

        elif network_architecture == ArchType.LAYERED_RECURRENT:  # #                       Recurrence within each layer mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)  # #  make a feed forward mask
            weight_mask += weight_mask.T  # #                                               make recurrent connections among the neurons of one layer

        elif network_architecture == ArchType.LAYERED_RECURRENT_RECURRENT:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask += weight_mask.T
            weight_mask = self._add_recurrent_connections(weight_mask, layers, learning_rate_factors)

        elif network_architecture == ArchType.IN_RECURRENT_OUT:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask = self._add_recurrent_connections(weight_mask, layers, learning_rate_factors)

        elif network_architecture == ArchType.FULLY_RECURRENT:  # # ALL to ALL connection mask without loops
            weight_mask = np.ones((sum(layers), sum(layers)))  # #  create a complete graph mask of weights
            np.fill_diagonal(weight_mask, 0)  # #                   drop loops

        else:
            raise NotImplementedError("Mask type ", network_architecture.name, " not implemented.")

        # only discriminative = no connections projecting back to the visible layer.
        if only_discriminative:
            weight_mask[:layers[0], :] *= 0

        weight_mask = torch.tensor(weight_mask, dtype=self.dtype, device=self.device)
        return weight_mask

    @staticmethod
    def _make_feed_forward_mask(layers, learning_rate_factors):
        """
        Returns a mask for a feedforward architecture.

        Args:
            layers: is a list containing the number of neurons per layer.
            learning_rate_factors: contains learning rate multipliers for each layer.

        Adapted from Jonathan Binas (@MILA)

        Weight mask encoding of the network structure
        ---------------------------------------------
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.
        """
        neuron_qty = int(np.sum(layers))  # total quantity of neurons
        layer_qty = len(layers)  # total quantity of layers
        mask = np.zeros((neuron_qty, neuron_qty))  # create adjacency matrix of neuron connections
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layers, the postsynaptic neurons to each layer
        for i in range(len(learning_rate_factors)):
            mask[layer_offsets[i + 1]:layer_offsets[i + 2], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]  # connect all of layer i (i to i+1) with layer i + 1

        return mask

    @staticmethod
    def _add_recurrent_connections(weight_mask, layers, learning_rate_factors):
        """
        Adds recurrent structure to every layer of network.
        """
        layer_qty = len(layers)  # total quantity of layers
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layer offsets

        for i in range(1, len(learning_rate_factors)):  # exclude input layer
            weight_mask[layer_offsets[i]:layer_offsets[i + 1], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]  # connect all of layer i with itself
        np.fill_diagonal(weight_mask, 0)  # drop loops (autosynapses)
        return weight_mask

    def _make_bias_mask(self, layers, input_size, learning_rate_factors, only_discriminative):
        """
        Creates mask for biases (similar to mask for weights).
        """
        neuron_qty = int(np.sum(layers))  # total quantity of neurons
        layer_qty = len(layers)  # total quantity of layers
        bias_mask = np.ones(neuron_qty)
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layers
        for i in range(len(learning_rate_factors)):
            bias_mask[layer_offsets[i]:layer_offsets[i + 1]] *= learning_rate_factors[i]  # set learning rate factors for biases of each layer

        if only_discriminative:
            bias_mask[:input_size] *= 0

        bias_mask = torch.tensor(bias_mask, dtype=self.dtype, device=self.device)
        return bias_mask

    @staticmethod
    def _set_solver(solver_type):
        """
        Either uses Cholesky solver (faster, but might fail due to numerical issues) or LU solver (slower, but works always).
        """

        if solver_type == IntegrationMethod.CHOLESKY_SOLVER:
            solve = lambda matrix, vec: torch.cholesky_solve(vec, torch.cholesky(matrix))  # use cholesky decomposition and the cholesky solver to solve Ax = b
        elif solver_type == IntegrationMethod.LU_SOLVER:
            solve = lambda matrix, vec: torch.solve(vec, matrix)[0]  # use LU decomposition solver to solve Ax = b
        else:
            print('Solver ' + solver_type.name + ' not implemented. Defaulting to LU solver.')
            solve = LagrangeNetwork._set_solver(IntegrationMethod.LU_SOLVER)

        return solve

    def _generate_activation_function(self, activation_function):
        """
        Implementation of different activation functions.
        """
        if activation_function == ActivationFunction.SIGMOID:  # if the activation is a sigmoid
            act_function = nn.Sigmoid()  # define the activation function as a sigmoid of voltages
            act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))  # function of the 1st derivative
            act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))  # function of the 2nd derivative
        elif activation_function == ActivationFunction.RELU:  # regular ReLu unit
            act_function = nn.ReLU()
            act_func_deriv = lambda voltages: torch.tensor((voltages > 0) * 1, dtype=self.dtype, device=self.device)  # compute the derivative of the relu function
            act_func_second_deriv = lambda voltages: 0.0
        elif activation_function == ActivationFunction.HARD_SIGMOID:  # ReLU which is clipped to 0-1
            act_function = lambda voltages: voltages.clamp(0, 1)
            act_func_deriv = lambda voltages: torch.tensor((voltages >= 0) * (voltages <= 1), dtype=self.dtype, device=self.device)
            act_func_second_deriv = lambda voltages: voltages * 0.
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv

    def _create_observables(self):
        """
        Make variables that can be recorded for cross-checks (but are not needed otherwise).
        """
        # observables
        self.obs_errors = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)    # observe neural errors
        self.obs_error_lookaheads = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # observe neural error look aheads

    def _initialize_network(self, weight_init_params, use_biases, bias_init_params, use_interneurons, backward_weight_scale):
        """
        Set up voltages, weights, and their derivatives.
        """

        # states
        # setup voltage variable for neurons
        self.st_voltages = torch.zeros((1, self.neuron_qty), dtype=self.dtype, device=self.device)  # initialize all voltages to the resting mb potential, 0
        self.st_voltages_deriv = torch.zeros((1, self.neuron_qty), dtype=self.dtype, device=self.device)  # initialize all voltage derivatives to 0

        # setup bias variables for neurons
        self.st_biases = torch.tensor(self._create_initial_biases(use_biases, self.bias_mask, **bias_init_params), dtype=self.dtype, device=self.device)
        self.st_biases_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

        # setup voltage variable for inter-neurons
        self.st_voltages_I = torch.zeros((1, self.neuron_qty), dtype=self.dtype, device=self.device)  # voltages of interneurons to resting mb potential
        self.st_voltages_I_deriv = torch.zeros((1, self.neuron_qty), dtype=self.dtype, device=self.device)

        # setup weights variable for pyramidal neurons
        self.st_weights = torch.tensor(self._create_initial_weights(self.weight_mask, **weight_init_params), dtype=self.dtype, device=self.device)
        self.st_weights_deriv = self.weight_mask * 0.0  # initialize the weight derivatives to zero

        if use_interneurons:
            self.st_biases_I = torch.tensor(self._create_initial_biases(use_biases, self.bias_mask, **bias_init_params), dtype=self.dtype, device=self.device)
            self.st_biases_I_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

            # setup weights from Pyr to IN
            self.st_weights_IP = torch.tensor(self._create_initial_weights(self.weight_mask, **weight_init_params), dtype=self.dtype, device=self.device)
            self.st_weights_IP_deriv = self.weight_mask * 0.0

            # setup weights from IN to Pyr
            self.st_weights_PI = torch.tensor(self._create_initial_weights(self.weight_mask, **weight_init_params).T, dtype=self.dtype, device=self.device)
            self.st_weights_PI_deriv = self.weight_mask * 0.0

            # setup backward weights
            self.st_weights_B = torch.tensor(self._create_initial_weights(self.weight_mask, **weight_init_params).T * backward_weight_scale, dtype=self.dtype, device=self.device)
            self.st_weights_B_deriv = self.weight_mask * 0.0
        else:  # weight copying
            self.st_biases_I = self.st_biases
            self.st_biases_I_deriv = self.st_biases_deriv

            self.st_weights_IP = self.st_weights
            self.st_weights_IP_deriv = self.st_weights_deriv

            self.st_weights_PI = torch.t(self.st_weights)
            self.st_weights_PI_deriv = torch.t(self.st_weights_deriv)

            self.st_weights_B = torch.t(self.st_weights)  # see below equation 24
            self.st_weights_B_deriv = torch.t(self.st_weights_deriv)

        # observables
        self._create_observables()

        # traces
        # Fill voltage_deriv_trace and error_deriv_trace with 4 sensible data points each (How? WS: Just add 4 good ones from the root solver, then later add 4 constant ones)
        self.voltage_deriv_trace = [self.st_voltages_deriv,
                                    self.st_voltages_deriv,
                                    self.st_voltages_deriv,
                                    self.st_voltages_deriv]
        self.error_deriv_trace = [torch.tensor(np.zeros(np.shape(self.st_voltages))),
                                  torch.tensor(np.zeros(np.shape(self.st_voltages))),
                                  torch.tensor(np.zeros(np.shape(self.st_voltages))),
                                  torch.tensor(np.zeros(np.shape(self.st_voltages)))]

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Create randomly initialized weight matrix.
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), -clip, clip) * (weight_mask.cpu().numpy() > 0)  # initialize weights with normal sample where mask is larger 0

    @staticmethod
    def _create_initial_biases(use_biases, bias_mask, mean, std, clip):
        """
        Create randomly initialized bias matrix (or set biases to zero if only weights are used).
        """
        neuron_qty = bias_mask.shape[0]
        if use_biases:
            return np.clip(np.random.normal(mean, std, size=neuron_qty), -clip, clip) * (bias_mask.cpu().numpy() > 0)  # initialize biases with normal sample where mask is larger 0
        else:
            return np.zeros(neuron_qty)  # set biases to zero

    # BASIC PROFILE AND TEST METHODS
    ################################

    def _profile_model(self, batch_size=32):
        """
        Profiling for the comp. graph. Performs one network update step and saves the profile data.
        The data can be loaded in Chrome at chrome://tracing/.
        """
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        sample_input = np.ones((batch_size, self.input_size))  # input voltages set to 1
        sample_output = np.ones((batch_size, self.target_size))  # output voltages set to 1

        _, cpu_profiler, cuda_profiler = bottleneck(lambda: self.update_network(sample_input, sample_output, True))

        if cpu_profiler is not None:
            cpu_profiler.export_chrome_trace(f"{stamp}_b{batch_size}_torch_cpu_timeline.json")

        if cuda_profiler is not None:
            cuda_profiler.export_chrome_trace(f"{stamp}_b{batch_size}_torch_cuda_timeline.json")

    def _test_simulation_run(self, batch_size=1):
        """
        Test network run. Estimates the average time used to perform a time/integration step.
        """
        sample_size = 50.0  # number of samples to calculate the average integration time of

        if batch_size == 1:
            sample_input = np.ones(self.input_size)  # input voltages set to 1
            sample_output = np.ones(self.target_size)  # output voltages set to 1
        else:
            sample_input = np.ones((batch_size, self.input_size))  # input voltages set to 1
            sample_output = np.ones((batch_size, self.target_size))  # output voltages set to 1

        # test prediction
        init_time = time.time()  # initial time
        for i in range(int(sample_size)):
            self.update_network(sample_input, sample_output, False)  # run sample_size prediction updates of the network

        time_per_prediction_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        # test training
        init_time = time.time()
        for i in range(int(sample_size)):
            self.update_network(sample_input, sample_output, True)  # run sample_size training updates of the network

        time_per_train_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        return time_per_prediction_step, time_per_train_step

    def report_simulation_params(self):
        """
        Print report of simulation setup to the terminal.
        """
        print('------------------')
        print('SUMMARY')
        print('------------------')
        print('Total number neurons: ', self.neuron_qty)
        print('Total number syn. connections: ', int(torch.sum(self.weight_mask).item()))
        print('Layer structure: ', self.layers)
        print('Network architecture: ', self.network_architecture.name)
        print('Use inter-neurons: ', self.use_interneurons)
        print('Activation function: ', self.activation_function.name)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Bias initial distribution: ', self.arg_bias_init_params)
        print('Learning rate: ', self.arg_lrate_W)
        print('Beta (nudging parameter): ', self.beta)
        print('Membrane time constant (Tau): ', self.tau)
        print('Time step: ', self.dt)
        print('Learning mini-batch size: ', self.st_voltages.shape[0])  # effectively checks how many different networks are simulated in parallel
        print('Time per prediction step in test run: ', self.time_per_prediction_step, 's')
        print('Time per training step in test run: ', self.time_per_train_step, 's')
        print('------------------')
        print("Simulation framework: Torch ", torch.__version__)
        print('Simulation running on :', self.device)
        print('------------------')

    # PERFORM UPDATE STEP OF NETWORK DYNAMICS
    #########################################

    def _perform_update_step(self, use_interneurons, dynamic_interneurons, use_biases, turn_off_visible_dynamics, train_W, train_B, train_PI):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i
        (14.1)           r_i = \bar r_i + \tau \dot{\bar r}_i
        (14.2)           e_i = \bar e_i + \tau \dot{\bar e}_i

        (14.3)     \bar e_i  =  \bar r'_i\odot [ W_{i+1}^\T  (u_{i+1} - W_{i+1} \bar r_i)]
        (14.4)      \bar e_N = \beta (u_N^{trg}(t) - u_N(t))

        Args:
            use_interneurons:
            dynamic_interneurons:
            use_biases:
            turn_off_visible_dynamics:
            train_W:
            train_B:
            train_PI:

        Returns:

        """
        self._adapt_parallel_network_qty()  # (BATCH) adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required, drop current sets)

        # prepare tensors which will be reused in the calculations quite often
        # get current activities and derivatives + the inputs each neuron receives
        rho, rho_deriv, rho_second_deriv, layerwise_input = self._get_activity_and_input(self.st_voltages, self.st_weights, self.st_biases)
        weights_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, self.st_weights)  # get the weights multiplied by rho', as written in Eq. 17.

        # Use either dynamic inter-neurons or instantaneous inter-neurons
        if dynamic_interneurons:  # dynamic inter-neurons integrate voltage through a diff. eq.
            self.st_voltages_I = self.st_voltages_I + self.dt * self.st_voltages_I_deriv  # i_voltage += i_voltage + i_voltage_deriv * dt, Eq. 32. Simple Euler equation
        else:                     # instantaneous inter-neurons get voltage from weighted activity
            self.st_voltages_I = self._calculate_layerwise_inputs(rho, self.st_weights_IP, self.st_biases_I)  # i_voltage = activity * weight + bias (Eq. 33)

        # CALCULATE WEIGHT AND BIAS DERIVATIVES
        # The weight update depends only on the current weights and voltages and can be updated first.
        if train_W:
            # (BATCH) we can take average of the weight derivatives to use batching
            self.st_weights_deriv = self._update_weights(self.st_voltages, rho, layerwise_input).mean(0).squeeze()
            if self.arg_clip_weight_derivs:
                self.st_weights_deriv = torch.clamp(self.st_weights_deriv, -self.arg_weight_deriv_clip_value, self.arg_weight_deriv_clip_value)
        else:
            self.st_weights_deriv = torch.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype, device=self.device)

        # calculate weight derivatives (with or without interneuron circuit)
        if use_interneurons:
            # derivative of backward weights
            if train_B:
                self.st_weights_B_deriv = self._update_weights_B(self.st_voltages, self.st_weights_B)  # do weights update (Eq. 31)
            else:
                self.st_weights_B_deriv = torch.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype, device=self.device)  # else weights deriv is zero

            # derivative of PI weights
            if train_PI:
                self.st_weights_PI_deriv = self._update_weights_PI(self.st_voltages, self.st_weights_B, self.st_weights_PI)  # do weights update (Eq. 34)
            else:
                self.st_weights_PI_deriv = torch.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype, device=self.device)  # else weights deriv is zero

            # precalculate weights with rho derivative
            # perform batched transpose
            weights_B_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, torch.t(self.st_weights_B)).permute(0, 2, 1)  # we just component-wise multiply the weight by rho_deriv (Eq. 36)
            weights_PI_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, torch.t(self.st_weights_PI)).permute(0, 2, 1)  # same
            weights_IP_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, self.st_weights_IP)  # same

        # without inter-neurons, we use weight transport as described in the theory.
        else:
            # copy weight derivatives
            self.st_weights_B_deriv = torch.t(self.st_weights_deriv)
            self.st_weights_PI_deriv = torch.t(self.st_weights_deriv)
            self.st_weights_IP_deriv = self.st_weights_deriv  # reset IP derivatives to weight derivatives

            # perform batched transpose
            weights_B_rho_deriv = weights_rho_deriv.permute(0, 2, 1)  # see Eq. 24 for relationships.
            weights_PI_rho_deriv = weights_rho_deriv.permute(0, 2, 1)  # same. Recall that top-down weights must be equal to PI weights
            weights_IP_rho_deriv = weights_rho_deriv

        # calculate bias derivatives
        if use_biases:  # if we use biases
            if train_W:
                # (BATCH) we can take average of the bias derivatives to use batching
                self.st_biases_deriv = self._update_biases(self.st_voltages, layerwise_input).mean(0)  # do bias update
            else:
                self.st_biases_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)

            # calculate derivative of interneuron biases
            if use_interneurons:
                if train_PI:
                    # (BATCH) we can take average of the bias derivatives to use batching
                    self.st_biases_I_deriv = self._update_biases_I(self.st_voltages, self.st_weights_B, self.st_weights_PI).mean(0)
                else:
                    self.st_biases_I_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # else inter-neuron bias deriv is zero
            else:  # copy bias derivatives
                self.st_biases_I_deriv = self.st_biases_deriv

        # CALCULATE VOLTAGE DERIVATIVES
        # NUMERICAL METHODS
        if self.integration_method == IntegrationMethod.EULER_METHOD:
            # Calculate voltages deriv in an euler step using the st_voltages_deriv and st_weights_deriv from the previous update
            self.st_voltages_deriv = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                          layerwise_input, self.st_voltages_deriv, self.st_weights_deriv)

        elif self.integration_method == IntegrationMethod.RICHARDSON_1ST_METHOD:
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=1)
            self.st_voltages_deriv = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                          layerwise_input, voltages_deriv_t_dt, self.st_weights_deriv)

        elif self.integration_method == IntegrationMethod.RICHARDSON_2ND_METHOD:
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=2)
            self.st_voltages_deriv = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                          layerwise_input, voltages_deriv_t_dt, self.st_weights_deriv)
        # COMPARATIVE MODES
        elif self.integration_method == IntegrationMethod.COMPARE_CHOLESKY_SOLVER or self.integration_method == IntegrationMethod.COMPARE_EULER_METHOD \
                or self.integration_method == IntegrationMethod.COMPARE_RICHARDSON_1ST_METHOD or self.integration_method == IntegrationMethod.COMPARE_RICHARDSON_2ND_METHOD:

            # calculate cholesky solution
            self._voltages_deriv_base = self._calculate_voltages_deriv_cholesky(turn_off_visible_dynamics,
                                                                                rho, rho_deriv, rho_second_deriv,
                                                                                weights_rho_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv,
                                                                                layerwise_input)

            # calculate euler solution
            self._voltages_deriv_euler = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                              layerwise_input, self.st_voltages_deriv, self.st_weights_deriv)

            # calculate richardson 1st solution
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=1)
            self._voltages_deriv_richardson_1st = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                                       layerwise_input, voltages_deriv_t_dt, self.st_weights_deriv)

            # calculate richardson 2nd solution
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=2)
            self._voltages_deriv_richardson_2nd = self._calculate_voltages_deriv_euler(rho, rho_deriv, rho_second_deriv,
                                                                                       layerwise_input, voltages_deriv_t_dt, self.st_weights_deriv)

            # select solution used for network voltage integration
            if self.integration_method == IntegrationMethod.COMPARE_CHOLESKY_SOLVER:
                self.st_voltages_deriv = self._voltages_deriv_base
            elif self.integration_method == IntegrationMethod.COMPARE_EULER_METHOD:
                self.st_voltages_deriv = self._voltages_deriv_euler
            elif self.integration_method == IntegrationMethod.COMPARE_RICHARDSON_1ST_METHOD:
                self.st_voltages_deriv = self._voltages_deriv_richardson_1st
            elif self.integration_method == IntegrationMethod.COMPARE_RICHARDSON_2ND_METHOD:
                self.st_voltages_deriv = self._voltages_deriv_richardson_2nd
            else:
                self.st_voltages_deriv = self._voltages_deriv_base
        # DEFAULT: LINEAR EQUATION SOLVERS
        else:
            self.st_voltages_deriv = self._calculate_voltages_deriv_cholesky(turn_off_visible_dynamics,
                                                                             rho, rho_deriv, rho_second_deriv,
                                                                             weights_rho_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv,
                                                                             layerwise_input)

        # update interneuron voltage derivatives if dynamic interneurons are used
        if dynamic_interneurons:
            # here we add the derivative of the interneuron voltages.
            # TODO: Check if interneurons are affected by implementations without solver (BE: I expect no problems)
            # (BATCH) we can calculate all voltage derivatives for the whole batch and run separate voltage dynamics for the whole batch, only averaging resulting weights
            self.st_voltages_I_deriv = self._get_interneuron_dynamics_derivatives(self.st_voltages_I, rho + self.tau * rho_deriv * self.st_voltages_deriv, rho, self.st_weights_IP,
                                                                                  self.st_weights_IP_deriv, self.st_biases_I, self.st_biases_I_deriv)

        if self.integration_method == IntegrationMethod.EULER_METHOD or self.integration_method == IntegrationMethod.RICHARDSON_1ST_METHOD or self.integration_method == IntegrationMethod.RICHARDSON_2ND_METHOD:
            error, error_lookaheads, error_derivs = self._calculate_error(rho, rho_deriv, rho_second_deriv,
                                                                          self.st_voltages, self.st_voltages_deriv, self.st_weights, self.st_weights_deriv, self.st_biases)

            # keep values of...
            self.obs_errors = error                       # self-prediction errors (observable)
            self.obs_error_lookaheads = error_lookaheads  # self-prediction error lookaheads (observable)
            self.error_derivs = error_derivs              # self-prediction error derivatives

            # keep traces of...(for extrapolations)
            window_size = 10
            self.voltage_deriv_trace.append(self.st_voltages_deriv)             # voltage derivatives (used for Richardson exptrapolation!)
            self.voltage_deriv_trace = self.voltage_deriv_trace[-window_size:]

            self.error_deriv_trace.append(self.error_derivs)                    # self-prediction error derivatives (used for Richardson exptrapolation!)
            self.error_deriv_trace = self.error_deriv_trace[-window_size:]

        # UPDATE NETWORK STATE
        # update voltages and weights with the derivatives calculated earlier
        # this describes a simple Euler integration to get the next voltages and weights
        # EULER EQUATIONS
        self.st_voltages = self.st_voltages + self.dt * self.st_voltages_deriv  # voltages += dt * voltages_deriv (EULER)
        self.st_weights = self.st_weights + self.dt * self.st_weights_deriv  # weights += dt * weights_deriv (EULER)

        if self.arg_clip_weights:
            self.st_weights = torch.clamp(self.st_weights, -self.arg_w_init_params['clip'], self.arg_w_init_params['clip'])

        # update interneurons
        if use_interneurons:
            self.st_weights_B = self.st_weights_B + self.dt * self.st_weights_B_deriv
            self.st_weights_PI = self.st_weights_PI + self.dt * self.st_weights_PI_deriv

            if self.arg_clip_weights:
                self.st_weights_B = torch.clamp(self.st_weights_B, -self.arg_w_init_params['clip'], self.arg_w_init_params['clip'])
                self.st_weights_PI = torch.clamp(self.st_weights_PI, -self.arg_w_init_params['clip'], self.arg_w_init_params['clip'])
        else:  # keep backward and PI weights in sync with transpose of weights
            self.st_weights_B = torch.t(self.st_weights)  # see Eq. 24 for relationships.
            self.st_weights_PI = torch.t(self.st_weights)  # same. Recall that top-down weights must be equal to PI weights

        self.st_weights_IP = self.st_weights

        # update biases
        if use_biases:
            self.st_biases = self.st_biases + self.dt * self.st_biases_deriv  # biases += dt * biases_deriv

            if use_interneurons:
                self.st_biases_I = self.st_biases_I + self.dt * self.st_biases_I_deriv  # inter-neuron_biases += dt * inter-neuron_biases_deriv
            else:  # copy biases
                self.st_biases_I = self.st_biases

            if self.arg_clip_weights:
                self.st_biases = torch.clamp(self.st_biases, -self.arg_bias_init_params['clip'], self.arg_bias_init_params['clip'])
                self.st_biases_I = torch.clamp(self.st_biases_I, -self.arg_bias_init_params['clip'], self.arg_bias_init_params['clip'])

    # FUNCTIONS OF CALCULATIONS NEEDED TO SOLVE ODE
    ########################################################

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.st_input_voltage.shape[0]
        if len(self.st_voltages.shape) != 2 or self.st_voltages.shape[0] != batch_size:
            voltage_size = self.st_voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder
            self.st_voltages = torch.repeat_interleave(self.st_voltages, repetition_vector, dim=0).clone()
            self.st_voltages_I = torch.repeat_interleave(self.st_voltages_I, repetition_vector, dim=0).clone()
            self.st_voltages_deriv = torch.repeat_interleave(self.st_voltages_deriv, repetition_vector, dim=0).clone()

    def _get_activity_and_input(self, voltages, weights, biases):
        """
        Return neural activity + derivatives thereof as well as the synaptic input of each neuron.
        Useful for computations u - Wr
        """
        rho = self.act_function(voltages)  # neural activity
        rho_deriv = self.act_func_deriv(voltages)  # derivative of neural activity
        rho_second_deriv = self.act_func_second_deriv(voltages)  # second derivative of neural activity
        layerwise_inputs = self._calculate_layerwise_inputs(rho, weights, biases)  # all the inputs to the all neurons
        return rho, rho_deriv, rho_second_deriv, layerwise_inputs

    def _calculate_layerwise_inputs(self, rho, weights, biases):
        """
        Returns external+network input for all neurons W * r + x.
        """
        internal_input = torch.matmul(weights, rho.t()).t() + biases  # activities * network_weights + biases
        external_input = internal_input[:, :self.input_size] + self.st_input_voltage  # internal input + external input = total input
        return torch.cat((external_input, internal_input[:, self.input_size:]), dim=1)  # concat the row receiving external input with the others

    def _set_input_and_target(self, input_voltage, target_voltage):
        """
        Set input and target of the network.
        Note: We need the input of the previous time step to approximate the derivatives of the inputs.
        Args:
            input_voltage:
            target_voltage:

        Returns:

        """
        # assert that input and output voltages are tensors
        if isinstance(input_voltage, np.ndarray):
            input_voltage = torch.tensor(input_voltage, dtype=self.dtype, device=self.device)

        if isinstance(target_voltage, np.ndarray):
            target_voltage = torch.tensor(target_voltage, dtype=self.dtype, device=self.device)

        # Make a batch of size one if the input is a single sample
        if len(input_voltage.shape) == 2:
            batch_size = input_voltage.shape[0]
            # assuming inputs and outputs have the same batch size
        else:
            batch_size = 1
            input_voltage = input_voltage.unsqueeze(0).expand((batch_size, -1)).clone()
            target_voltage = target_voltage.unsqueeze(0).expand((batch_size, -1)).clone()

        if len(self.st_input_voltage.shape) != 2:
            self.st_input_voltage = self.st_input_voltage.unsqueeze(0)

        if len(self.st_target_voltage.shape) != 2:
            self.st_target_voltage = self.st_target_voltage.unsqueeze(0)

        if self.st_input_voltage.shape[0] != batch_size:
            self.st_input_voltage = self.st_input_voltage.mean(0).expand((batch_size, -1)).clone()

        if self.st_target_voltage.shape[0] != batch_size:  # new input batch size is unequal the previous batch size
            self.st_target_voltage = self.st_target_voltage.mean(0).expand((batch_size, -1)).clone()

        # set (formerly) current as old and new input as current
        self.st_old_input_voltage = self.st_input_voltage
        self.st_input_voltage = input_voltage

        self.st_old_target_voltage = self.st_target_voltage
        self.st_target_voltage = target_voltage

    def _calculate_dendritic_error(self, voltages, weights_B, weights_PI):
        """
        Calculates difference between what we can explain away and what is left as error:
            :math:`(B * u - W_PI * W_IP * r) = B * u - W_PI * u^I = e.  r_i'` terms are already included
        """
        return torch.matmul(weights_B, voltages.t()).t() - torch.matmul(weights_PI, self.st_voltages_I.t()).t()  # back projected neuron output - inter-neuron output, st_voltages_I = W_IP.r_i according to Eq. 33

    # DERIVATIVES FOR WEIGHT & BIAS UPDATE
    ######################################

    def _update_weights(self, voltages, rho, layerwise_inputs):
        """
        Weight update of the network weights. See Eq. 18
        """
        # multiply by weight_mask to keep only existing connections
        # Batch Outer Product (https://discuss.pytorch.org/t/batch-outer-product/4025)
        return torch.einsum('bi,bj->bij', [voltages - layerwise_inputs, rho]  # (voltages - layer input mismatch) * activities
                            ) * self.weight_mask * self.arg_lrate_W  # * weight mask * learning rate

    def _update_biases(self, voltages, layerwise_inputs):
        """
        Bias update of the network biases. Same as weights_update but without pre-synaptic activities (Eq. 18)
        """
        return (voltages - layerwise_inputs) * self.arg_lrate_biases * self.bias_mask

    def _update_biases_I(self, voltages, weights_B, weights_PI):
        """
        Bias update of the inter-neuron biases. Same as weights_PI_update without pre-synaptic voltages (Eq. 34)
        TODO: Interneuron bias update looks odd (mismatch has no brackets)
        """
        return torch.matmul(weights_B, voltages.t()).t() - torch.matmul(weights_PI, self.st_voltages_I.t()).t() * self.bias_mask * self.arg_lrate_biases_I

    def _update_weights_B(self, voltages, weights_B):
        """
        Weight update of the generative weights of the interneuron circuit. See Eq. 31
        """
        return torch.einsum('bi,bj->bij', [voltages  # #                                   network voltages
                                           - torch.matmul(weights_B, voltages.t()).t(),  # #           weighted generative voltages
                                           voltages]  # #                                   ^^mismatch * network voltages
                            ) * self.weight_mask.t() * self.arg_lrate_B  # #   * all respective network weights * learning rate

    def _update_weights_PI(self, voltages, weights_B, weights_PI):
        """
        Weight update of the interneuron to pyramidal weights (to learn a self-predicting state). See Eq. 34
        """
        return torch.einsum('bi,bj->bij', [torch.matmul(weights_B, voltages.t()).t()  # #                           weighted network voltages
                                           - torch.matmul(weights_PI, self.st_voltages_I.t()).t(),  # #             weighted interneuron voltages
                                           self.st_voltages_I]  # #                                      ^^mismatch * interneuron voltage
                            ) * self.weight_mask.t() * self.arg_lrate_PI  # #                  * all respective network connections * learning rate

    @staticmethod
    def _dress_weights_with_rho_deriv(rho_deriv, weights):
        """
        Dress weights with the derivative of the act. function. See Eq. 17
        """
        return weights.unsqueeze(0) * rho_deriv.unsqueeze(1)

    # SOLVE ODE WITH LINEAR EQUATION SOLVER
    ########################################

    def _calculate_voltages_deriv_cholesky(self, turn_off_visible_dynamics,
                                           rho, rho_deriv, rho_second_deriv,
                                           weights_rho_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv,
                                           layerwise_input):
        """Rewrite Eq. 14 as linear equation set of form: dot u = F * m and solve it with linear equation solver.

        (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i

        Args:
            turn_off_visible_dynamics:
            rho:
            rho_deriv:
            rho_second_deriv:
            weights_rho_deriv:
            weights_B_rho_deriv:
            weights_PI_rho_deriv:
            weights_IP_rho_deriv:
            layerwise_input:

        Returns:

        """
        # build the linear equations to calculate the voltage derivatives
        # get the dendritic error, which is the mismatch between prediction and reality
        dendritic_error = self._calculate_dendritic_error(self.st_voltages, self.st_weights_B, self.st_weights_PI)  # Eq. 24

        self.st_voltages_deriv, err_vector, err_matrix, err_w_deriv = self._solve_linear_equations_for_volts_deriv(
            turn_off_visible_dynamics,
            rho, rho_deriv, rho_second_deriv,
            weights_rho_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv,
            dendritic_error, layerwise_input)

        return self.st_voltages_deriv

    def _solve_linear_equations_for_volts_deriv(self, turn_off_visible_dynamics,
                                                rho, rho_deriv, rho_second_deriv,
                                                weights_rho_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv,
                                                dendritic_error, layerwise_input):

        """Solve a linear equation for u derivative for the case where the neural network uses look aheads.

        (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i

        Since on the right-hand side of equation (14.0) :math:`\dot{u}` appears only linearly,
        the ODE can be rewritten in the following form:

        .. math::
            (64) F \dot{u} = m

        with matrix F and vector m that are independent of \dot{u}. F and m are given by:

        .. math::
            (65) \frac{1}{\tau} F = \mathds{1} - B_{\bar{r}^\T} - W_{\bar{r}} + W^{PI}_{\bar{r}^\T} W_{\beta^\I} - (\bar{r}'' \odot e_{A})\mathds{1} + \beta \mathds{1} _{y}
            (66)                m = \bar{r}' e_{A} - \tau W^{PI}_{\bar{r}^\T} \dot{\bar{x}} + \beta (y - u) + (W\bar{r} + b + x - u) + m_{\dot{W}}


            (67.0)              W_{\bar{r}} = W \odot\, &\bar{r}'
            (67.1)           W_{\bar{r}^\T} = (W^\T \odot \bar{r}')^\T
            (68)               W_{\beta^\I} = \tilde{\beta}^\I W^{IP}_{\bar{r}} + \beta^\I \tilde{\beta}^\I M
            (69)                      e_{A} = Bu-W^{PI} u^\I
            (70) \frac{1}{\tau} m_{\dot{W}} = \dot{W}\bar{r} + \dot{b} + \bar{r}'(\dot{B} u - \dot{W}^{PI} u^\I - \tilde{\beta}^\I W^{PI} \dot{W}^{IP}\bar{r} - \tilde{\beta}^\I W^{PI} \dot{b}^\I )

        Unite the previous functions to build the torch computational graph and solve the ODEs governing
        the time dependence of the voltages and weights.

        Args:
            turn_off_visible_dynamics:

            rho:
            rho_deriv:
            rho_second_deriv:

            weights_rho_deriv:
            weights_B_rho_deriv:
            weights_PI_rho_deriv:
            weights_IP_rho_deriv:

            dendritic_error:
            layerwise_input:

        Returns:

        """
        # We use the dendritic_error for the next computation, described by Eq. 17 (and further explanations below).
        # We obtain a matrix and error that we will use for solving the linear system.
        # get error matrix, being ..., and error vector, being ..
        err_matrix, err_vector = self._get_error_terms(self.st_voltages, rho, rho_deriv, rho_second_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv, dendritic_error)

        # get error_w_deriv, being ..., and input_w_deriv, being ...
        err_w_deriv, inp_w_deriv = self._get_weight_derivative_terms(self.st_voltages, rho, rho_deriv,
                                                                     self.st_weights_deriv, self.st_weights_B_deriv,
                                                                     self.st_weights_PI_deriv,
                                                                     self.st_weights_IP_deriv, self.st_weights_PI,
                                                                     self.st_biases_deriv,
                                                                     self.st_biases_I_deriv)

        # get input_matrix, being ..., and input vector, being ...
        input_matrix, input_vector = self._get_layer_input_terms(self.st_voltages, layerwise_input, weights_rho_deriv)

        # calculate the voltage derivatives
        # we have all information available to solve the equation F(u) * u_dot = b(u) for u_dot using a linear equation solver. Below we treat both cases
        # of having dynamics in the input neurons or without such dynamics

        # if the input is not meant to be dynamic, we clamped the input, i.e. r_1 = visible input activity
        if turn_off_visible_dynamics:
            # we store the matrix M(u) supposed to be multiplied by u_dot in a big matrix
            full_matrix = err_matrix + input_matrix

            # In the clamped input case, we do not need to compute u_dot since all components are already known without dynamics.
            # So we force the derivative of the input neurons to take the value of arg_input. We do that by writing the formal derivative for the clamped input.
            input_u_dot_derivative = (self.st_input_voltage - self.st_voltages[:, :self.input_size]) / self.dt

            # With the desired derivative known, we simply multply the input_u_dot_derivative with the respective part of M(u) to find the contribution of the input to b(u).
            input_contribution = torch.bmm(full_matrix[:, self.input_size:, :self.input_size], input_u_dot_derivative.unsqueeze(-1)).squeeze()

            # We can now subtract this contribution from b(u) to remove the contribution from the linear system.
            # The remaining linear system can now be solved for the remaining, unknown u_dot. This gives us the complete solution consisting of the input u_dot derivative and the
            # solution of the remaining linear system explaining the contribution of the non-input neurons. The two parts can be concatenated to form the complete solution.
            st_voltages_deriv = torch.cat((input_u_dot_derivative,
                                           self.solve(
                                               full_matrix[:, self.input_size:, self.input_size:],  # F(u)
                                               (
                                                       (err_vector + input_vector + err_w_deriv + inp_w_deriv)[:, self.input_size:]  # b(u) - input_contribution
                                                       - input_contribution
                                               ).unsqueeze(-1)).squeeze(-1)  # adjust dimensions to handle batches properly
                                           ), dim=1)
        else:
            # here, the input neurons are nudged, so we do not know their dynamics. We then solve the whole system altogether.
            # (BATCH) we can calculate all voltage derivatives for the whole batch and run separate voltage dynamics for the whole batch, only averaging resulting weights
            st_voltages_deriv = self.solve((err_matrix + input_matrix).unsqueeze(0), (err_vector + input_vector + err_w_deriv + inp_w_deriv).unsqueeze(2)).squeeze()

        return st_voltages_deriv, err_vector, err_matrix, err_w_deriv

    # INDIVIDUAL PARTS OF THE LINEAR EQUATIONS
    ##########################################

    def _get_error_terms(self, voltages, rho, rho_deriv, rho_second_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv, dendritic_error):
        """
        Part of the linear equations.
        Calculates all terms originating from the error :math:`e = e_bar + tau*d(e_bar)/dt`.
        e can be recovered by calculating matrix_comp * voltage_derivative + vector_comp, i.e., e was divided into a part depending
        on the voltage derivatives and an independent part: :math:`F(u) * u_derivative + m(u)` (cf drawings)

        Args:
            voltages:

            rho:
            rho_deriv:
            rho_second_deriv:

            weights_B_rho_deriv:
            weights_PI_rho_deriv:
            weights_IP_rho_deriv:

            dendritic_error:

        Returns: :math:`F(u), m(u)`, which are part of the linear equation :math:`e_deriv = F(u) * u_deriv + m(u)`
        """
        matrix_comp = -self.tau * (weights_B_rho_deriv - torch.bmm(weights_PI_rho_deriv, weights_IP_rho_deriv))
        matrix_comp += -self.tau * torch.diag_embed(torch.mul(rho_second_deriv, dendritic_error))
        target_term = torch.eye(self.neuron_qty, dtype=self.dtype, device=self.device)
        target_term[:-self.target_size, :-self.target_size] = 0
        beta_factors = torch.cat((torch.zeros(sum(self.layers[:-1]), dtype=self.dtype, device=self.device), self.beta), dim=0)
        matrix_comp += target_term * beta_factors * self.tau

        vector_comp = torch.mul(rho_deriv, dendritic_error)
        batch_size = self.st_input_voltage.shape[0]
        deriv = torch.cat((self.tau / self.dt * (self.st_input_voltage - self.st_old_input_voltage), torch.zeros(batch_size, self.neuron_qty - self.input_size, dtype=self.dtype, device=self.device)), dim=1)
        vector_comp += -torch.bmm(weights_PI_rho_deriv, deriv.unsqueeze(-1)).squeeze()
        vector_comp += -rho_deriv * torch.matmul(self.st_weights_PI, (self._calculate_layerwise_inputs(rho, self.st_weights_IP, self.st_biases_I) - self.st_voltages_I).t()).t()  # Eq. 17

        target_vector_comp = vector_comp[:, -self.target_size:] \
                             + self.beta * (self.st_target_voltage - voltages[:, -self.target_size:]) \
                             + self.tau * self.beta * (self.st_target_voltage - self.st_old_target_voltage) / self.dt

        return matrix_comp, torch.cat((vector_comp[:, : -self.target_size], target_vector_comp), dim=1)

    def _get_weight_derivative_terms(self, voltages, rho, rho_deriv, weights_deriv, weights_B_deriv, weights_PI_deriv, weights_IP_deriv, weights_PI, biases_deriv, biases_I_deriv):
        """
        Part of the linear equations.
        Terms originating from taking the derivatives of the weights. Completely in vector form (i.e., independent of the derivative of the voltages).
        """
        error_comp = self.tau * rho_deriv * (torch.matmul(weights_B_deriv, voltages.t()).t() - torch.matmul(weights_PI, torch.matmul(weights_IP_deriv, rho.t())).t())
        error_comp += -self.tau * rho_deriv * (torch.matmul(weights_PI_deriv, self.st_voltages_I.t()).t() + torch.mv(weights_PI, biases_I_deriv))
        input_comp = self.tau * (torch.matmul(weights_deriv, rho.t()).t() + biases_deriv)

        return error_comp, input_comp

    def _get_layer_input_terms(self, voltages, layerwise_inputs, weights_rho_deriv):
        """
        Part of the linear equations.
        Terms originating from external and synaptic inputs.
        """
        matrix_comp = self.tau * (torch.eye(self.neuron_qty, dtype=self.dtype, device=self.device) - weights_rho_deriv)
        vector_comp = layerwise_inputs - voltages

        # prepare batch of external layer inputs
        ext_vector_comp = vector_comp[:, :self.input_size] + self.tau / self.dt * (self.st_input_voltage - self.st_old_input_voltage)

        vector_comp = torch.cat((ext_vector_comp, vector_comp[:, self.input_size:]), dim=1)
        return matrix_comp, vector_comp

    # SOLVE ODE WITHOUT SOLVER
    ###########################

    def _calculate_voltages_deriv_euler(self, rho, rho_deriv, rho_scnd_deriv, layerwise_inputs, voltages_t_deriv, dot_weights):
        """
        Calculate derivative of voltages using an euler update:
            (16) dot u_i = 1/tau * (- u_i + W_i * r_{i-1}  + e_i + dot(u_input))

        This method assumes that the equation can be solved by using the former voltage_t_deriv dot u (t-1) as a slightly disturbed version of dot u (t).
        For smaller dt (< 1.0 ms = 0.1 * \tau), this seems to hold.

        Args:
            rho:
            rho_deriv:
            rho_scnd_deriv:
            layerwise_inputs:
            voltages_t_deriv:
            dot_weights:

        Returns:

        """
        # calculate rho lookahead (r) from bar r + tau * rho_t_deriv
        rho_lookaheads = self._get_rho_lookaheads(rho, rho_deriv, voltages_t_deriv)

        #  calculate W * r  = r_(i-1)
        layerwise_input_lookaheads = self._get_layerwise_input_lookaheads(rho_lookaheads, self.st_weights, self.st_biases)

        # calculate error lookahead (e_i)
        error_lookaheads = self._get_error_lookaheads(self.st_voltages, rho, rho_deriv, rho_scnd_deriv,
                                                      self.st_target_voltage, layerwise_inputs, layerwise_input_lookaheads,
                                                      voltages_t_deriv, dot_weights)

        # dot u of input
        u_deriv_input = self.tau * torch.matmul(dot_weights, rho.t()).t()

        # calculate derivative of voltages (dot(u)_i = 1/tau * (W_i * r - u_i + e_i + dot(u_input)))
        u_deriv = 1.0 / self.tau * (layerwise_input_lookaheads - self.st_voltages + error_lookaheads + u_deriv_input)
        return u_deriv

    @staticmethod
    def _calculate_voltages_deriv_t_dt_richardson(voltages_deriv_t_dt, voltages_deriv_t_2dt, voltages_deriv_t_4dt, order=2):
        if order == 1:
            voltages_deriv_t = 2 * voltages_deriv_t_dt - voltages_deriv_t_2dt
        elif order == 2:
            voltages_deriv_t = (8 * voltages_deriv_t_dt - 6 * voltages_deriv_t_2dt + voltages_deriv_t_4dt) / 3
        else:  # fallback to euler
            voltages_deriv_t = voltages_deriv_t_dt

        return voltages_deriv_t

    def _get_rho_lookaheads(self, rho, rho_deriv, voltages_t_deriv):
        """
        Calculate rho lookaheads.
        Get r_i = \bar r_i + \tau \dot{\bar r}_i
        Args:
            rho:
            rho_deriv: bar r'
            voltages_t_deriv: dot (u)

        Returns:

        """
        rho_t_deriv = rho_deriv * voltages_t_deriv  # bar dot(r) = bar r' * dot(u)
        return rho + self.tau * rho_t_deriv

    def _get_layerwise_input_lookaheads(self, rho_lookahead, weights, biases):
        """
        Calculate future inputs to each layer by looking ahead into the future by taking an euler step with the derivative.
        Returns external + network lookahead input for all neurons W * r + x.

        Args:
            rho_lookahead:
            weights:
            biases:

        Returns:

        """
        internal_input = torch.matmul(weights, rho_lookahead.t()).t() + biases  # lookahead_activities * network_weights + biases
        external_input = internal_input[:, :self.input_size] + self.st_input_voltage \
                         + self.tau / self.dt * (self.st_input_voltage - self.st_old_input_voltage)  # internal input + external input = total input
        return torch.cat((external_input, internal_input[:, self.input_size:]), dim=1)  # concat the row receiving external input with the others

    def _get_error_lookaheads(self, voltages,
                              rho, rho_deriv, rho_scnd_deriv,
                              target, layerwise_inputs, layerwise_input_lookaheads,
                              voltages_t_deriv, dot_weights):
        """
        Calculate e_i = bar(e_i) + tau * dot(bar(e_i)). (Eq. 14.2)
        Args:
            voltages:

            rho:
            rho_deriv:
            rho_scnd_deriv:

            target:
            layerwise_inputs:
            layerwise_input_lookaheads:

            voltages_t_deriv:
            dot_weights:

        Returns:

        """

        # TODO: Many weights.t() could be some of the tied matrices, this should be checked.
        # bar(e_i) (missing e_{trg})
        err = self._get_bar_e(voltages, rho_deriv, layerwise_input_lookaheads, voltages_t_deriv)

        # + tau * dot(bar(e_i))
        err += self.tau * self._get_dot_bar_e(voltages, rho, rho_deriv, rho_scnd_deriv, layerwise_inputs, voltages_t_deriv, dot_weights)

        # + e_{trg)
        target_err = err[:, -self.target_size:] + self._get_e_trg(voltages, target, voltages_t_deriv)
        return torch.cat((err[:, : -self.target_size], target_err), dim=1)

    # INDIVIDUAL PARTS OF THE ODE
    ###############################

    def _get_bar_e(self, voltages, rho_deriv, layerwise_input_lookaheads, voltages_t_deriv):
        """
        Calculate bar e (without e_trg).
        Args:
            voltages:
            rho_deriv:
            layerwise_input_lookaheads:
            voltages_t_deriv:

        Returns:

        """
        return rho_deriv * torch.matmul(self.st_weights.t(), (voltages + self.tau * voltages_t_deriv - layerwise_input_lookaheads).t()).t()

    def _get_e_trg(self, voltages, target, voltages_t_deriv):
        """
        Calculate e target.

        Insert into errors as err[-self.target_size:] = get_e_trg(...).
        Args:
            voltages:
            target:
            voltages_t_deriv:

        Returns:

        """
        return self._get_squared_loss(target, voltages[:, -self.target_size:]) + self.tau * self.beta * (
                (target - self.st_old_target_voltage) / self.dt - voltages_t_deriv[:, -self.target_size:])

    def _get_dot_bar_e(self, voltages,
                       rho, rho_deriv, rho_scnd_deriv,
                       layerwise_inputs,
                       dot_voltages, dot_weights):
        """
         Calculate dot(bar(e_i).
         Args:
             voltages:
             rho:
             rho_deriv:
             rho_scnd_deriv:
             layerwise_inputs:
             dot_voltages:
             dot_weights:

         Returns:

         """
        dot_bar_e = rho_scnd_deriv * dot_voltages * torch.matmul(self.st_weights.t(), (voltages - layerwise_inputs).t()).t()
        dot_bar_e += rho_deriv * torch.matmul(dot_weights.t(), (voltages - layerwise_inputs).t()).t()
        dot_bar_e -= rho_deriv * torch.matmul(self.st_weights.t(), torch.matmul(dot_weights, rho.t())).t()
        return dot_bar_e

    def _get_squared_loss(self, target, output):
        """
        Calculate squared loss function.
        Args:
            target:
            output:

        Returns:

        """
        return self.beta * (target - output)

    def _calculate_error(self, rho, rho_deriv, rho_scnd_deriv, voltages, voltages_deriv, weights, weights_deriv, biases):
        """
        (OBSERVABLES) Calculate error and error lookaheads. Used as observables.
        Args:
            voltages:
            voltages_deriv:
            weights:
            weights_deriv:
            biases:

        Returns:

        """
        # calculate inputs from other layers for each layer
        layerwise_inputs = self._calculate_layerwise_inputs(rho, weights, biases)  # (W_i * r) + u_input

        # calculate lookaheads
        rho_lookaheads = self._get_rho_lookaheads(rho, rho_deriv, voltages_deriv)
        input_lookaheads = self._get_layerwise_input_lookaheads(rho_lookaheads, weights, biases)

        # calculate observables
        errors = self._calculate_layerwise_errors(voltages, rho_deriv, weights, layerwise_inputs, self.st_target_voltage)
        errors_derivs = self._get_dot_bar_e(voltages, rho, rho_deriv, rho_scnd_deriv, layerwise_inputs, voltages_deriv, weights_deriv)
        error_lookaheads = self._get_error_lookaheads(voltages, rho, rho_deriv, rho_scnd_deriv,
                                                      self.st_target_voltage, layerwise_inputs, input_lookaheads,
                                                      voltages_deriv, weights_deriv)

        return errors, error_lookaheads, errors_derivs

    def _calculate_layerwise_errors(self, voltages, rho_deriv, weights, layer_inputs, target):
        """
        Used for observables.
        Calculate errors for each layer.
        Args:
            voltages:
            rho_deriv:
            weights:
            layer_inputs:
            target:

        Returns:

        """
        err = rho_deriv * torch.matmul(weights.t(), (voltages - layer_inputs).t()).t()
        target_err = err[:, -self.target_size:] + self._get_squared_loss(target, voltages[:, -self.target_size:])
        return torch.cat((err[:, : -self.target_size], target_err), dim=1)

    # INTERNEURON METHODS
    #####################

    def _get_interneuron_dynamics_derivatives(self, voltage_I, rho_look_ahead, rho, weights_IP, weights_IP_deriv, biases_I, biases_I_deriv):
        """
        Calculate interneuron dynamics if they are enabled.
        """

        # internal input = - current interneuron voltage + layerwise inputs to interneuron + tau * (bias derivative + weights derivative * activities)
        # left part corresponds to Eq. 32 with lookahead rates.
        internal_input = -voltage_I + self._calculate_layerwise_inputs(rho_look_ahead, weights_IP, biases_I) + self.tau * (biases_I_deriv + torch.matmul(weights_IP_deriv, rho.t()).t())  # Eq. 32

        # external input = internal input to input layer + tau / dt * input derivative, for the derivative of interneuron voltages of input neurons, it can be expressed explicitly by using the visible_input variables (definition of derivative)
        external_input = internal_input[:, :self.input_size] + self.tau / self.dt * (self.st_input_voltage - self.st_old_input_voltage)

        # we divide by tau to obtain u_dot^I in Eq. 32 -> what is called interneuron dynamics.
        return (1. / self.tau) * torch.cat((external_input, internal_input[:, self.input_size:]), dim=1)

    # NETWORK INTEGRATOR
    #####################

    def update_network(self, input_voltage, target_voltage, train_W=False, train_B=False, train_PI=False):
        """
        Perform a single integration step.
        """
        self._set_input_and_target(input_voltage, target_voltage)
        self._perform_update_step(self.use_interneurons, self.dynamic_interneurons, self.use_biases, self.turn_off_visible_dynamics, train_W, train_B, train_PI)

    # GET and SET
    def set_beta(self, beta):
        """
        Set the nudging beta.
        Args:
            beta:

        Returns:

        """

        # keep beta state for quick checking
        if np.isscalar(beta) and beta == 0 or np.all(beta) == 0:
            self.is_beta_zero = True
        else:
            self.is_beta_zero = False

        if np.isscalar(beta):
            beta = np.ones(self.layers[-1]) * beta

        self.beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

    def get_beta(self):
        return self.beta.detach().cpu().numpy()[0]

    def set_lrate_W(self, learning_rate):
        """
        Set weight learning rate.
        Args:
            learning_rate:

        Returns:

        """
        self.arg_lrate_W = learning_rate

    def get_lrate_W(self):
        return self.arg_lrate_W

    def set_lrate_B(self, learning_rate):
        """
        Set backwards weights learning rate (used if they are learned separately).
        Args:
            self:
            learning_rate:

        Returns:

        """
        self.arg_lrate_B = learning_rate

    def set_lrate_biases(self, learning_rate):
        """
        Set bias learning rate.
        Args:
            self:
            learning_rate:

        Returns:

        """
        self.arg_lrate_biases = learning_rate

    def set_lrate_biases_I(self, learning_rate):
        """
        Set interneuron bias learning rate.
        Args:
            self:
            learning_rate:

        Returns:

        """
        self.arg_lrate_biases_I = learning_rate

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

    # OBSERVABLES
    #############
    def get_voltages(self):
        """
        Get network pyramidal neuron voltages.
        Args:
            self:

        Returns:

        """
        return self.st_voltages.detach().cpu().numpy()

    def get_output_voltages(self):
        """
        Get the voltages of the output neurons.
        Returns:
        """
        return self.st_voltages[:, -self.layers[-1]:].detach().cpu().numpy()

    def get_voltage_derivatives(self):
        """
        Get network pyramidal neuron voltage derivatives.
        Args:
            self:

        Returns:

        """
        return self.st_voltages_deriv.detach().cpu().numpy()

    def get_errors(self):
        """
        Get self-prediction error resulting from mismatch between pyramidal neuron compartment voltages.
        Args:
            self:

        Returns:

        """
        return self.obs_errors.detach().cpu().numpy()

    def get_error_lookaheads(self):
        """
        Get self-prediction error lookaheads.
        Args:
            self:

        Returns:

        """
        return self.obs_error_lookaheads.detach().cpu().numpy()

    def get_weights(self):
        """
        Get network synaptic weights between pyramidal neurons.
        Args:
            self:

        Returns:

        """
        return self.st_weights.detach().cpu().numpy()

    def get_biases(self):
        """
        Get network synaptic biases between pyramidal neurons.
        Args:
            self:

        Returns:

        """
        return self.st_biases.detach().cpu().numpy()

    def get_voltage_deriv_base(self):
        return self._voltages_deriv_base.detach().cpu().numpy()

    def get_voltage_deriv_euler(self):
        return self._voltages_deriv_euler.detach().cpu().numpy()

    def get_voltage_deriv_richardson_1st(self):
        return self._voltages_deriv_richardson_1st.detach().cpu().numpy()

    def get_voltage_deriv_richardon_2nd(self):
        return self._voltages_deriv_richardson_2nd.detach().cpu().numpy()

    # KERAS-like TRAINING AND PREDICTION INTERFACE
    ##############################################

    def fit(self, x=None, y=None, n_updates: int = 100, batch_size=1, epochs=1, verbose=1, beta=0.1, is_timeseries=False):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per batch.
            batch_size: Number of examples per batch (batch training).
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        """
        n_samples = len(x)  # dataset size

        if self.is_beta_zero:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(beta))
            self.set_beta(beta)  # turn nudging on to enable training

        print("Learning with batch size {0}".format(batch_size))

        dataset = SimpleDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

        batch_qty = int(np.floor(n_samples/batch_size))

        for epoch_i in range(epochs):
            for batch_i, (x, y) in tqdm(enumerate(data_loader), desc="Batches", total=batch_qty):
                x = x.to(self.device)
                y = y.to(self.device)
                if batch_i == 0:
                    self.old_x = x
                    self.old_y = y

                self.fit_batch(x, y, n_updates, batch_i, batch_qty, verbose, is_timeseries)

    def fit_batch(self, x, y, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose=1, is_timeseries=False):

        if verbose >= 1:
            print("train:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
            self.xr = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

        batch_size = x.shape[0]
        if batch_size != self.old_y.shape[0]:  # new input batch size is unequal the previous batch size
            self.old_x = x
            self.old_y = y

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            if is_timeseries:
                samples, labels = x, y
            else:
                samples, labels = self._decay_func(self.xr[update_i], self.old_x, x), self._decay_func(self.xr[update_i], self.old_y, y)
            self.update_network(samples, labels, train_W=True, train_B=False, train_PI=False)

        self.old_x = x
        self.old_y = y

        if verbose >= 1:
            print('')

    def predict(self, x, n_updates: int = 100,  batch_size=1, verbose=1, is_timeseries=False):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Size of batch to be predicted.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        :return:
        """
        n_samples = len(x)  # dataset size
        self.set_beta(0.0)   # turn nudging off to disable learning

        dataset = SimpleDataset(x, [np.zeros(self.target_size) for _ in range(n_samples)])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=True)

        predictions = []
        for batch_i, (x, _) in enumerate(data_loader):
            x = x.to(self.device)

            batch_predictions = self.predict_batch(x, n_updates, batch_iteration=batch_i, batch_qty=int(n_samples/batch_size), verbose=verbose, is_timeseries=is_timeseries)
            predictions.extend(batch_predictions.detach().cpu().numpy())

        return predictions

    def predict_batch(self, x, n_updates=100, batch_iteration=-1, batch_qty=-1, verbose=1, is_timeseries=False):

        if verbose >= 1:
            print("predict:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
            self.xr = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

        batch_size = x.shape[0]
        if batch_size != self.old_x.shape[0]:  # new input batch size is unequal the previous batch size
            self.old_x = x

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            if is_timeseries:
                samples = x
            else:
                samples = self._decay_func(self.xr[update_i], self.old_x, x)
            self.update_network(samples, self.dummy_label, train_W=False, train_B=False, train_PI=False)

        self.old_x = x

        batch_predictions = self.st_voltages[:, -self.target_size:]

        if verbose >= 1:
            print('')

        return batch_predictions

    def __call__(self, x, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose=1, is_timeseries=False):
        self.predict_batch(x, n_updates, batch_iteration, batch_qty, verbose, is_timeseries)

    # SAVE AND LOAD NETWORK
    #######################

    def save(self, save_path):
        """
        Save the lagrange model to file.
        Args:
            save_path:

        Returns:

        """
        voltages = self.st_voltages[0].clone().detach().cpu().numpy()  # (BATCH) keep only one set of voltages of the whole batch
        weights = self.st_weights.detach().cpu().numpy()
        biases = self.st_biases.detach().cpu().numpy()

        np.save('{0}/voltages'.format(save_path), voltages)
        np.save('{0}/weights'.format(save_path), weights)
        np.save('{0}/biases'.format(save_path), biases)

    def load(self, load_path):
        """
        Load the lagrange model from file.
        Args:
            load_path:

        Returns:

        """
        try:
            voltages = np.load('{0}/voltages.npy'.format(load_path))
            weights = np.load('{0}/weights.npy'.format(load_path))
            biases = np.load('{0}/biases.npy'.format(load_path))
        except Exception:
            return False

        self.st_voltages = torch.tensor(voltages, device=self.device).unsqueeze(0)  # (BATCH) prepare it for batches
        self.st_weights = torch.tensor(weights, device=self.device)
        self.st_biases = torch.tensor(biases, device=self.device)

        return True

    # TIME SERIES BUILDER FUNCTIONS
    ###############################
    @staticmethod
    def _decay_func(x, equi1, equi2):
        """
        Decay function to transform sample and label by exponential fading.
        Sample x from xr2 = np.linspace(-20, 90, n_updates)
        """
        return equi1 + (equi2 - equi1) / (1 + torch.exp(-x / 4.0))

