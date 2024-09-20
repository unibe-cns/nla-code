# Numpy implementation of the Lagrange model (RtDeep). Trainable only with single samples. Not used, but kept for understanding the model.
#
#
# Authors: Benjamin Ellenberger (benelot@github)

import time

import numpy as np
# for optimization with root solver
from scipy.optimize import root

from model.network_params import NetworkParams, ActivationFunction, IntegrationMethod


class LagrangeNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Lagrange model.
    Prefixes: st = state, pl = placeholder, arg = argument.
    """

    # INIT
    ########
    def __init__(self, params: NetworkParams):

        # store important params
        self.layers = params.layers
        self.tau = params.arg_tau  # ms
        self.dt = params.arg_dt  # ms

        # learning rates
        self.arg_lrate = params.arg_lrate_W

        self.arg_w_init_params = params.arg_w_init_params

        self.dtype = params.dtype

        self.set_beta(params.arg_beta)

        self.rnd_seed = params.rnd_seed

        # this network has no biases

        # setup inputs and outputs
        self.input_size = params.layers[0]  # input neurons
        self.target_size = params.layers[-1]  # target neurons
        self.neuron_qty = sum(params.layers)
        self.st_input_voltage = np.zeros(self.input_size, dtype=self.dtype)
        self.st_old_input_voltage = np.zeros(self.input_size, dtype=self.dtype)
        self.st_target_voltage = np.zeros(self.target_size, dtype=self.dtype)
        self.st_old_target_voltage = np.zeros(self.target_size, dtype=self.dtype)

        assert (len(params.layers) - 1 == len(params.learning_rate_factors))

        # prepare input vector to generate input transients
        self.xr = np.linspace(-20, 90, 100, dtype=self.dtype)
        self.old_x = np.zeros(self.input_size, dtype=self.dtype)
        self.old_y = np.zeros(self.target_size, dtype=self.dtype)
        self.dummy_label = np.zeros(self.target_size, dtype=self.dtype)

        self.INTEGRATION_METHODS = [IntegrationMethod.ROOT_SOLVER,
                                    IntegrationMethod.EULER_METHOD,
                                    IntegrationMethod.RICHARDSON_1ST_METHOD,
                                    IntegrationMethod.RICHARDSON_2ND_METHOD]

        # setup neuron weight masks
        self.weight_mask = self._make_feed_forward_mask(self.layers, params.learning_rate_factors, feedback=False)
        self.integration_method = IntegrationMethod.ROOT_SOLVER

        self.act_function, self.act_func_deriv, self.act_func_second_deriv = self._generate_activation_function(params.activation_function)

        # setup network
        self._initialize_network()

        # perform single sample test run to get network performance stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # reinitialize network variables
        self._initialize_network()

    # NETWORK SETUP METHODS
    ###########################

    @staticmethod
    def _make_feed_forward_mask(layers, learning_rate_factors, feedback=True):
        """
        Returns a mask for a feedforward architecture.

        Args:
            layers: is a list containing the number of neurons per layer.
            learning_rate_factors: contains learning rate multipliers for each layer.
            feedback: add feedback connections to every forward connection

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

        return mask + feedback * mask.T

    # # ODE PARTS # #
    # ## ACTIVATION FUNCTIONS AND DERIVATIVES ##
    @staticmethod
    def _generate_activation_function(activation_function):
        """
        Implementation of different activation functions.
        """
        if activation_function == ActivationFunction.SIGMOID:  # if the activation is a sigmoid
            act_function = lambda voltages: 1.0 / (1 + np.exp(-voltages))  # define the activation function as a sigmoid of voltages
            act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))  # function of the 1st derivative
            act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))  # function of the 2nd derivative
        elif activation_function == ActivationFunction.RELU:  # regular ReLu unit
            act_function = lambda  voltages: voltages* (voltages > 0)
            act_func_deriv = lambda voltages: (voltages > 0) * 1  # compute the derivative of the relu function
            act_func_second_deriv = lambda voltages: 0.0
        elif activation_function == ActivationFunction.HARD_SIGMOID:  # ReLU which is clipped to 0-1
            act_function = lambda voltages: voltages.clamp(0, 1)
            act_func_deriv = lambda voltages: (voltages >= 0) * (voltages <= 1)
            act_func_second_deriv = lambda voltages: voltages * 0.0
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv

    def _initialize_network(self):
        """
        Set up voltages, weights, and their derivatives.
        """
        self._set_random_seed(self.rnd_seed)

        # states
        # setup voltage variable for neurons
        self.st_voltages = np.zeros(self.neuron_qty, dtype=self.dtype)
        self.st_voltages_deriv = np.zeros(self.neuron_qty, dtype=self.dtype)
        self.st_weights = self._create_initial_weights(self.weight_mask,
                                                       self.arg_w_init_params['mean'],
                                                       self.arg_w_init_params['std'],
                                                       self.arg_w_init_params['clip'])
        # observables
        self.voltages_deriv_diff = np.zeros(np.shape(self.st_voltages))        # voltage derivative residual from the dot u equation
        self.error = np.zeros(np.shape(self.st_voltages))                      # self-prediction errors
        self.error_lookaheads = np.zeros(np.shape(self.st_voltages))           # self-prediction error lookaheads
        self.error_derivs = np.zeros(np.shape(self.st_voltages))               # self-prediction error derivatives

        # traces
        # self.voltage_trace = []
        # Fill voltage_deriv_trace and error_deriv_trace with 4 sensible data points each (How? WS: Just add 4 good ones from the root solver, then later add 4 constant ones)
        self.voltage_deriv_trace = [self.st_voltages_deriv,
                                    self.st_voltages_deriv,
                                    self.st_voltages_deriv,
                                    self.st_voltages_deriv]
        # self.error_trace, self.error_lookahead_trace = [], []
        self.error_deriv_trace = [np.zeros(np.shape(self.st_voltages)),
                                  np.zeros(np.shape(self.st_voltages)),
                                  np.zeros(np.shape(self.st_voltages)),
                                  np.zeros(np.shape(self.st_voltages))]
        # self.voltages_diff_root_error_trace = []

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Create randomly initialized weight matrix.
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), -clip, clip) * (weight_mask > 0)  # initialize weights with normal sample where mask is larger 0

    # BASIC PROFILE AND TEST METHODS
    ############################################
    def _test_simulation_run(self):
        """
        Test network run. Estimates the average time used to perform a time/integration step.
        """
        sample_size = 50.0  # number of samples to calculate the average integration time of

        sample_input = np.ones(self.input_size)  # input voltages set to 1
        sample_output = np.ones(self.target_size)  # output voltages set to 1

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
        print('Total number syn. connections: ', int(np.sum(self.weight_mask).item()))
        print('Layer structure: ', self.layers)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Learning rate: ', self.arg_lrate)
        print('Beta (nudging parameter): ', self.beta)
        print('Membrane time constant (Tau): {0} ms'.format(self.tau))
        print('Time step: {0} ms'.format(self.dt))
        print('Time per prediction step in test run: {0} s'.format(self.time_per_prediction_step))
        print('Time per training step in test run: {0} s'.format(self.time_per_train_step))
        print('------------------')
        print("Simulation framework: Numpy ", np.__version__)
        print('simulation running on : cpu')
        print('------------------')

    # PERFORM UPDATE STEP OF NETWORK DYNAMICS
    #########################################
    def _perform_update_step(self, train_W=False):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i
        (14.1)           r_i = \bar r_i + \tau \dot{\bar r}_i
        (14.2)           e_i = \bar e_i + \tau \dot{\bar e}_i

        (14.3)     \bar e_i  =  \bar r'_i\odot [ W_{i+1}^\T  (u_{i+1} - W_{i+1} \bar r_i)]
        (14.4)      \bar e_N = \beta (u_N^{trg}(t) - u_N(t))

        Args:
            train_W:

        Returns:

        """

        # CALCULATE WEIGHT DERIVATIVES AND UPDATE WEIGHTS
        if train_W:
            dot_weights = self._calculate_weight_derivatives(self.st_voltages, self.st_weights)
            new_weights = self.st_weights + self.dt * dot_weights

            if self.integration_method != IntegrationMethod.ROOT_SOLVER and self.integration_method != IntegrationMethod.COMPARE_ROOT_SOLVER:
                new_weights = np.clip(new_weights, -self.arg_w_init_params[2], self.arg_w_init_params[2])  # Clip weight size to avoid unstable learning
        else:
            dot_weights = np.zeros((self.neuron_qty, self.neuron_qty))
            new_weights = self.st_weights

        # CALCULATE VOLTAGE DERIVATIVES AND UPDATE VOLTAGES

        if self.integration_method == IntegrationMethod.ROOT_SOLVER:
            voltages_deriv = self._calculate_voltages_deriv_root(dot_weights)
        elif self.integration_method == IntegrationMethod.EULER_METHOD:
            voltages_deriv = self._calculate_voltages_deriv_euler(self.st_voltages_deriv, dot_weights)
        elif self.integration_method == IntegrationMethod.RICHARDSON_1ST_METHOD:
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=1)
            voltages_deriv = self._calculate_voltages_deriv_euler(voltages_deriv_t_dt, dot_weights)
        elif self.integration_method == IntegrationMethod.RICHARDSON_2ND_METHOD:
            voltages_deriv_t_dt = self._calculate_voltages_deriv_t_dt_richardson(self.voltage_deriv_trace[-1], self.voltage_deriv_trace[-2], self.voltage_deriv_trace[-4], order=2)
            voltages_deriv = self._calculate_voltages_deriv_euler(voltages_deriv_t_dt, dot_weights)
        else:
            voltages_deriv = self._calculate_voltages_deriv_root(dot_weights)

        new_voltages = self.st_voltages + self.dt * voltages_deriv

        # CALCULATE OBSERVABLES
        voltages_deriv_diff = self._get_voltages_deriv_root_function(voltages_deriv, dot_weights)

        # assert(np.max(np.fabs(voltages_deriv_diff)) < 1e-1)  # assert that dot u equation (root solver) does not go out of bounds
        error, error_lookaheads, error_derivs = self._calculate_error(self.st_voltages, voltages_deriv, self.st_weights, dot_weights)

        # keep values of...
        self.voltages_deriv_diff = voltages_deriv_diff          # voltage derivative residual from the dot u equation
        self.error = error                                      # self-prediction errors
        self.error_lookaheads = error_lookaheads                # self-prediction error lookaheads
        self.error_derivs = error_derivs                        # self-prediction error derivatives

        self.st_voltages = new_voltages
        self.st_voltages_deriv = voltages_deriv
        self.st_weights = new_weights

        # keep traces of...
        window_size = 10
        self.voltage_deriv_trace.append(self.st_voltages_deriv)             # voltage derivatives (used for Richardson exptrapolation!)
        self.voltage_deriv_trace = self.voltage_deriv_trace[-window_size:]

        self.error_deriv_trace.append(self.error_derivs)                    # self-prediction error derivatives (used for Richardson exptrapolation!)
        self.error_deriv_trace = self.error_deriv_trace[-window_size:]

        return new_voltages, voltages_deriv, new_weights, dot_weights, voltages_deriv_diff, error, error_lookaheads

    # FUNCTIONS OF CALCULATIONS NEEDED TO SOLVE ODE
    ########################################################

    def set_input_and_target(self, input_voltage, target_voltage):
        """
        Set input and target of the network.
        Note: We need the input of the previous time step to approximate the derivatives of the inputs.
        Args:
            input_voltage:
            target_voltage:

        Returns:

        """
        # set (formerly) current as old and new input as current
        self.st_old_input_voltage = self.st_input_voltage
        self.st_input_voltage = input_voltage

        self.st_old_target_voltage = self.st_target_voltage
        self.st_target_voltage = target_voltage

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def _calculate_weight_derivatives(self, voltages, weights):
        """
        Calculate weight derivative (Eq. 18).
        Args:
            voltages:
            weights:

        Returns:

        """
        rho = self.act_function(voltages)  # r
        layerwise_inputs = self._calculate_layerwise_inputs(rho, weights)  # W_i * r
        return np.outer(voltages - layerwise_inputs, rho) * self.weight_mask * self.arg_lrate  # (u_i - W_i * r) * r * eta * wm

    def _calculate_layerwise_inputs(self, rho, weights):
        """
        Calculate the inputs coming from other layers to each layer (W_i * r) + u_input.
        Args:
            rho:
            weights:

        Returns:

        """
        layerwise_inputs = np.dot(weights, rho)  # internal input (W_i * r)
        layerwise_inputs[:self.input_size] = self.st_input_voltage  # internal input + u_input (external input)
        return layerwise_inputs

    # SOLVE ODE WITH ROOT SOLVER
    ###########################

    def _calculate_voltages_deriv_root(self, weights_deriv):
        """
        Calculate derivative of voltages with a root solver method (a method to find a voltage deriv that returns a zero from the
        get_voltages_deriv_root_function).
        Args:
            weights_deriv:

        Returns:

        """
        return root(self._get_voltages_deriv_root_function, self.st_voltages_deriv, args=(weights_deriv,)).x

    def _get_voltages_deriv_root_function(self, voltage_deriv, weights_deriv):
        """
        This function represents the voltage derivatives equation,
        in which the voltage_derivative occurs on left and right hand side.

        The root solver tries to find a voltage_deriv for which this function returns zero.
        Args:
            voltage_deriv:
            weights_deriv:

        Returns:

        """
        return voltage_deriv - self._calculate_voltages_deriv_euler(voltage_deriv, weights_deriv)

    # SOLVE ODE WITHOUT SOLVER
    ###########################

    def _calculate_voltages_deriv_euler(self, voltage_t_deriv, dot_weights):
        """
        Calculate derivative of voltages:
            (16) dot u_i = 1/tau * (- u_i + W_i * r_{i-1}  + e_i + dot(u_input))

        Args:
            voltage_t_deriv: Previous time derivative of voltages: dot(u)_i
            dot_weights: Time derivative of weights: dot(W)

        Returns:

        """
        # precalculate activation functions
        rho = self.act_function(self.st_voltages)  # bar r
        rho_deriv = self.act_func_deriv(self.st_voltages)  # bar r'
        rho_scnd_deriv = self.act_func_second_deriv(self.st_voltages)  # bar r''

        # calculate rho lookahead (r) from bar r + tau * rho_t_deriv
        rho_lookaheads = self._get_rho_lookaheads(rho, rho_deriv, voltage_t_deriv)

        # calculate W * r = r_(i -1)
        layerwise_input_lookaheads = self._get_layerwise_input_lookaheads(rho_lookaheads, self.st_weights)

        # calculate inputs from other layers for each layer = W_i * r_(i-1)
        layerwise_inputs = self._calculate_layerwise_inputs(rho, self.st_weights)  # (W_i * bar r) + u_input

        # calculate error lookahead (e_i)
        error_lookaheads = self._get_error_lookaheads(self.st_voltages, rho, rho_deriv, rho_scnd_deriv,
                                                      self.st_target_voltage, layerwise_inputs, layerwise_input_lookaheads,
                                                      voltage_t_deriv, dot_weights)

        # W_i
        u_deriv_input = self.tau * np.dot(dot_weights, rho)

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

    def _get_layerwise_input_lookaheads(self, rho_lookahead, weights):
        """
        Calculate future inputs to each layer by looking ahead.
        Returns external + network lookahead input for all neurons W * r + x.

        Args:
            rho_lookahead:
            weights:

        Returns:

        """
        input_lookaheads = np.dot(weights, rho_lookahead)
        input_lookaheads[:self.input_size] = self.st_input_voltage + self.tau / self.dt * (self.st_input_voltage - self.st_old_input_voltage)
        return input_lookaheads

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
        # bar(e_i) (missing e_{trg})
        err = self._get_bar_e(voltages, rho_deriv, layerwise_input_lookaheads, voltages_t_deriv)

        # + tau * dot(bar(e_i))
        err += self.tau * self._get_dot_bar_e(voltages, rho, rho_deriv, rho_scnd_deriv, layerwise_inputs, voltages_t_deriv)

        # + e_{trg)
        err[-self.target_size:] = self._get_e_trg(voltages, target, voltages_t_deriv)
        return err

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
        return rho_deriv * np.dot(self.st_weights.T, voltages + self.tau * voltages_t_deriv - layerwise_input_lookaheads)

    def _get_e_trg(self, voltages, target, voltages_deriv):
        """
        Get e_trg (insert into errors as err[-self.target_size:] = get_e_trg(...).)
        Args:
            voltages:
            target:
            voltages_deriv:

        Returns:

        """
        return self._get_squared_loss(target, voltages[-self.target_size:]) + self.tau * self.beta * (
                (target - self.st_old_target_voltage) / self.dt - voltages_deriv[-self.target_size:])

    def _get_dot_bar_e(self, voltages,
                       r_bar, r_bar_deriv, r_bar_scnd_deriv,
                       basal_inputs,
                       dot_voltages):
        """
        Calculate dot(bar(e_i).
        Args:
            voltages:
            r_bar:
            r_bar_deriv:
            r_bar_scnd_deriv:
            basal_inputs:

        Returns:

        """
        dot_bar_e = r_bar_scnd_deriv * dot_voltages * np.dot(self.st_weights.T, (voltages - basal_inputs))
        dot_bar_e += r_bar_deriv * np.dot(self.st_weights.T, (dot_voltages - np.dot(self.st_weights, r_bar) * dot_voltages))
        dot_voltages_trg = (self.st_target_voltage - self.st_old_target_voltage) / self.dt
        dot_bar_e += self.beta * (dot_voltages_trg - dot_voltages)

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

    def _calculate_error(self, voltages, voltages_deriv, weights, weights_deriv):
        """
        Calculate error and error lookaheads.
        Args:
            voltages:
            voltages_deriv:
            weights:
            weights_deriv:

        Returns:

        """
        # precalculate activation functions
        rho = self.act_function(voltages)  # r
        rho_deriv = self.act_func_deriv(voltages)  # dot(r)
        rho_scnd_deriv = self.act_func_second_deriv(voltages)  # dot(dot(r))

        # calculate inputs from other layers for each layer
        layerwise_inputs = self._calculate_layerwise_inputs(rho, weights)  # (W_i * r) + u_input

        # calculate lookaheads
        rho_lookaheads = self._get_rho_lookaheads(rho, rho_deriv, voltages_deriv)
        input_lookaheads = self._get_layerwise_input_lookaheads(rho_lookaheads, weights)

        # calculate observables
        errors = self._calculate_layerwise_errors(voltages, rho_deriv, weights, layerwise_inputs, self.st_target_voltage)
        errors_derivs = self._get_dot_bar_e(voltages, rho, rho_deriv, rho_scnd_deriv, layerwise_inputs, voltages_deriv)
        error_lookaheads = self._get_error_lookaheads(voltages, rho, rho_deriv, rho_scnd_deriv,
                                                      self.st_target_voltage, layerwise_inputs, input_lookaheads,
                                                      voltages_deriv, weights_deriv)

        return errors, error_lookaheads, errors_derivs

    def _calculate_layerwise_errors(self, voltages, rho_deriv, weights, layer_inputs, target):
        """
        Calculate errors for each layer.
        Args:
            voltages:
            rho_deriv:
            weights:
            layer_inputs:
            target:

        Returns:

        """
        err = rho_deriv * np.dot(weights.T, voltages - layer_inputs)
        err[-self.target_size:] = self._get_squared_loss(target, voltages[-self.target_size:])
        return err

    # NETWORK INTEGRATOR
    #####################

    def update_network(self, input_voltage, target_voltage, train_W=False):
        """
        Perform a single integration step.
        """
        self.set_input_and_target(input_voltage, target_voltage)
        self._perform_update_step(train_W)

    # GET and SET
    def set_beta(self, beta):
        """
        Set the nudging beta.
        Args:
            beta:

        Returns:

        """
        self.beta = beta

    def get_voltages(self):
        return self.st_voltages

    def get_voltage_derivatives(self):
        return self.st_voltages_deriv

    def get_errors(self):
        return self.error

    def get_error_lookaheads(self):
        return self.error_lookaheads

    def get_weights(self):
        return self.st_weights

    # GET and SET
    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)

    # KERAS-like INTERFACE
    def fit(self, x=None, y=None, n_updates: int = 100, epochs=1, verbose=1, is_timeseries=False):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per sample.
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        """
        n_samples = len(x)  # dataset size

        if self.beta == 0:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(0.1))
            self.set_beta(0.1)  # turn nudging on to enable training

        print("Learning with single samples")

        for epoch_i in range(epochs):
            for sample_i, (x, y) in enumerate(zip(x, y)):
                if sample_i == 0:
                    self.old_x = x
                    self.old_y = y

                if verbose >= 1:
                    print("train:: sample ", sample_i, "/", n_samples, " | update ", end=" ")

                if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
                    self.xr = np.linspace(-20, 90, n_updates, dtype=self.dtype)

                for update_i in range(n_updates):
                    if verbose >= 2 and update_i % 10 == 0:
                        print(update_i, end=" ")

                    if is_timeseries:
                        sample, label = x, y
                    else:
                        sample, label = self._decay_func(self.xr[update_i], self.old_x, x), self._decay_func(self.xr[update_i], self.old_y, y)
                    self.update_network(sample, label, train_W=True)

                self.old_x = x
                self.old_y = y

                if verbose >= 1:
                    print('')

    def predict(self, x, n_updates: int = 100, verbose=1, is_timeseries=False):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        :return:
        """
        n_samples = len(x)  # dataset size
        self.set_beta(0.0)   # turn nudging off to disable learning

        predictions = []
        for sample_i, x in enumerate(x):

            if sample_i == 0:
                self.old_x = x

            if verbose >= 1:
                print("predict:: sample", sample_i, "/", n_samples, " | update ", end=" ")

            if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
                self.xr = np.linspace(-20, 90, n_updates, dtype=self.dtype)

            for update_i in range(n_updates):
                if verbose >= 2 and update_i % 10 == 0:
                    print(update_i, end=" ")

                if is_timeseries:
                    sample = x
                else:
                    sample = self._decay_func(self.xr[update_i], self.old_x, x)
                self.update_network(sample, self.dummy_label, train_W=False)

            self.old_x = x

            volts = self.st_voltages
            prediction = volts[-self.target_size:]
            predictions.append(prediction)

            if verbose >= 1:
                print('')

        return predictions

    def __call__(self, x, n_updates: int = 100, verbose=1, is_timeseries=False):
        self.predict(x, n_updates, verbose, is_timeseries)

    # SAVE AND LOAD NETWORK
    def save(self, save_path):
        """
        Save the lagrange model to file.
        Args:
            save_path:

        Returns:

        """
        voltages = self.st_voltages
        weights = self.st_weights

        np.save('{0}/voltages'.format(save_path), voltages)
        np.save('{0}/weights'.format(save_path), weights)

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
        except Exception:
            return False

        self.st_voltages = voltages
        self.st_weights = weights

        return True

    # TIME SERIES BUILDER FUNCTIONS
    @staticmethod
    def _decay_func(x, equi1, equi2):
        """
        Decay function to transform sample and label by exponential fading.
        Sample x from xr2 = np.linspace(-20, 90, n_updates)
        """
        return equi1 + (equi2 - equi1) / (1 + np.exp(-x / 4.0))



