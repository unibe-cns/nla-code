# Tensorflow implementation of the Lagrange model. Trainable only with single samples.
#
# Authors: Dominik Dold (dodo47@github) and Benjamin Ellenberger (benelot@github)


import numpy as np
import time
import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.python.client import timeline

# network parameters
from model.network_params import ArchType, IntegrationMethod, ActivationFunction, NetworkParams

# utils
from utils.tensorflow_utils import tf_mat_vec_dot, tf_outer_product, check_nan


class LagrangeNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Lagrange model.
    Prefixes: st = state, pl = placeholder, obs = observable, arg = argument.
    """

    # INIT
    ########
    def __init__(self, params: NetworkParams):
        # store important params
        self.layers = params.layers
        self.arg_tau = params.arg_tau
        self.arg_dt = params.arg_dt
        self.set_beta(params.arg_beta)

        self.arg_lrate_W = params.arg_lrate_W
        self.arg_lrate_B = params.arg_lrate_W_B
        self.arg_lrate_PI = params.arg_lrate_PI

        self.arg_lrate_biases = params.arg_lrate_biases
        self.arg_lrate_biases_I = params.arg_lrate_biases_I

        self.arg_w_init_params = params.arg_w_init_params
        self.arg_bias_init_params = params.arg_bias_init_params
        self.arg_clip_weight_derivs = params.arg_clip_weight_deriv
        self.arg_weight_deriv_clip_value = params.arg_weight_deriv_clip_value

        self.use_sparse_mult = params.use_sparse_mult
        self.dtype = params.dtype
        self.network_architecture = params.network_architecture
        self.use_interneurons = params.use_interneurons
        self.activation_function = params.activation_function
        self.rnd_seed = params.rnd_seed

        self._set_random_seed(params.rnd_seed)

        # setup inputs and outputs
        self.input_size = params.layers[0]
        self.target_size = params.layers[-1]
        self.neuron_qty = sum(params.layers)
        self.arg_input = np.zeros(self.input_size)
        self.arg_old_input = np.zeros(self.input_size)
        self.arg_target = np.zeros(self.target_size)
        self.arg_old_target = np.zeros(self.target_size)

        # setup neuron weight and bias masks
        self.weight_mask = self._make_connection_weight_mask(params.layers, params.learning_rate_factors, params.network_architecture, params.only_discriminative)
        self.bias_mask = self._make_bias_mask(params.layers, self.input_size, params.learning_rate_factors, params.only_discriminative)

        # setup activation function and solver
        self.act_function, self.act_func_deriv, self.act_func_second_deriv = self._generate_activation_function(params.activation_function)
        self.solve = self._set_linear_equation_solver(params.integration_method)  # # we need a solver to find the solution u_derivs from the linear ODEs of the form M * u_deriv = b(u)
        #                                                                      Source: https://en.wikipedia.org/wiki/Cholesky_decomposition#Applications
        #                   instead of using a solver, we could use some traditional euler/RK4 scheme, but this takes more steps for convergence due to the noisier gradient

        # setup network
        self._create_placeholders()  # setup changeable parameters
        self._create_observables()  # setup observable outputs (for observing network progress)
        self._initialize_network(params.arg_w_init_params, params.use_biases, params.arg_bias_init_params, params.use_interneurons, params.arg_interneuron_b_scale)
        self.network_updates = self._create_computational_graph(params.use_interneurons, params.dynamic_interneurons, params.use_biases, params.with_input_dynamics)

        # perform basic profiling
        if params.check_with_profiler:
            print("\nProfile model for single sample:\n--------------------------------")
            self._profile_model()

        if params.write_tensorboard:
            self._write_basic_tensorboard()

        # perform single sample test run to get network stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # reinitialize network variables
        self.sess.run(tf.global_variables_initializer())

    # NETWORK SETUP METHODS
    ###########################

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

        elif network_architecture == ArchType.FULLY_RECURRENT:  # # All to ALL connection mask without loops
            weight_mask = np.ones((sum(layers), sum(layers)))  # #  create a complete graph mask of weights
            np.fill_diagonal(weight_mask, 0)  # #                   drop loops

        else:
            raise NotImplementedError("Mask type ", network_architecture.name, " not implemented.")

        # only discriminative = no connections projecting back to the visible layer.
        if only_discriminative:
            weight_mask[:layers[0], :] *= 0

        return weight_mask

    @staticmethod
    def _make_feed_forward_mask(layers, learning_rate_factors):
        """
        Returns a mask for a feedforward architecture.

        Args:
            layers: is a list containing the number of neurons per layer.
            learning_rate_factors: contains learning rate multipliers for each layer.

        Adapted from Jonathan Binas (@MILA)
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

    @staticmethod
    def _make_bias_mask(layers, input_size, learning_rate_factors, only_discriminative):
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
        return bias_mask

    @staticmethod
    def _set_linear_equation_solver(solver_type):
        """
        Either uses Cholesky solver (faster, but might fail due to numerical issues) or LU solver (slower, but works always).
        """

        if solver_type == IntegrationMethod.CHOLESKY_SOLVER:
            solve = lambda matrix, vec: tf.cholesky_solve(tf.cholesky(matrix), tf.expand_dims(vec, 1))[:, 0]  # use cholesky decomposition and the cholesky solver to solve Ax = b
        elif solver_type == IntegrationMethod.LU_SOLVER:
            solve = lambda matrix, vec: tf.matrix_solve(matrix, tf.expand_dims(vec, 1))[:, 0]  # use LU decomposition solver to solve Ax = b
        elif solver_type == IntegrationMethod.LS_SOLVER:
            solve = lambda matrix, vec: tf.matrix_solve_ls(matrix, tf.expand_dims(vec, 1))[:, 0]  # use least squares solver
        else:
            print('Solver ' + solver_type.name + ' not implemented. Defaulting to LU solver.')
            solve = LagrangeNetwork._set_linear_equation_solver(IntegrationMethod.LU_SOLVER)

        return solve

    @staticmethod
    def _generate_activation_function(activation_function):
        """
        Implementation of different activation functions.
        """
        if activation_function == ActivationFunction.SIGMOID:  # if the activation is a sigmoid
            act_function = lambda voltages: tf.sigmoid(voltages)  # define the activation function as a sigmoid of voltages
            act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))  # function of the 1st derivative
            act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))  # function of the 2nd derivative
        elif activation_function == ActivationFunction.RELU:  # regular ReLu unit
            act_function = lambda voltages: nn.relu(voltages)
            act_func_deriv = lambda voltages: tf.gradients(nn.relu(voltages), voltages)[0]  # compute the derivative of the relu function
            act_func_second_deriv = lambda voltages: tf.gradients(tf.gradients(nn.relu(voltages), voltages), voltages)[0]
        elif activation_function == ActivationFunction.HARD_SIGMOID:  # ReLU which is clipped to 0-1
            act_function = lambda voltages: tf.clip_by_value(voltages, 0, 1)
            act_func_deriv = lambda voltages: tf.gradients(tf.clip_by_value(voltages, 0, 1), voltages)[0]
            act_func_second_deriv = lambda voltages: voltages * 0.
        elif activation_function == ActivationFunction.SIGMOID_INTEGRAL:  # Integral of e^x/(1+ exp(x)) = ln(1 + exp(x))
            act_function = lambda voltages: tf.log(1 + tf.exp(voltages))
            act_func_deriv = lambda voltages: tf.gradients(tf.log(1 + tf.exp(voltages)), voltages)[0]
            act_func_second_deriv = lambda voltages: tf.gradients(tf.gradients(tf.log(1 + tf.exp(voltages)), voltages), voltages)[0]
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv

    def _create_placeholders(self):
        """
        Make placeholders for parameters that should be changeable during experiments.
        """
        with tf.name_scope("placeholders"):
            with tf.name_scope("inputs"):
                self.pl_visible_input = tf.placeholder(dtype=self.dtype, shape=self.input_size, name="input")  # current input
                self.pl_old_visible_input = tf.placeholder(dtype=self.dtype, shape=self.input_size, name="old_input")  # previous input

            with tf.name_scope("outputs"):
                self.pl_target = tf.placeholder(dtype=self.dtype, shape=self.target_size, name="target")  # current target
                self.pl_old_target = tf.placeholder(dtype=self.dtype, shape=self.target_size, name="old_target")  # previous target

            with tf.name_scope("dt/tau"):
                self.pl_dt = tf.placeholder(dtype=self.dtype, shape=(), name="dt")  # integration constant
                self.pl_tau = tf.placeholder(dtype=self.dtype, shape=(), name="tau")  # membrane time constant

            with tf.name_scope("learning_rates"):
                self.pl_beta = tf.placeholder(dtype=self.dtype, shape=(self.target_size,), name="beta")  # nudging parameter beta
                self.pl_lrate_W = tf.placeholder(dtype=self.dtype, shape=(), name="lrate_W")  # learning rate of the network weights
                self.pl_lrate_biases = tf.placeholder(dtype=self.dtype, shape=(), name="lrate_biases")  # learning rate of the biases
                self.pl_lrate_biases_I = tf.placeholder(dtype=self.dtype, shape=(), name="lrate_biases_I")  # learning rate of the inter-neuron biases
                self.pl_lrate_B = tf.placeholder(dtype=self.dtype, shape=(), name="lrate_B")  # learning rate of the feedback weights B (Figure 3.)
                self.pl_lrate_PI = tf.placeholder(dtype=self.dtype, shape=(), name="lrate_PI")  # learning rate from inter-neuron to principal neurons

            with tf.name_scope("derivative_configs"):
                self.pl_weight_deriv_clip_value = tf.placeholder(dtype=bool, shape=(), name="weight_deriv_clip")  # if we clip weight derivatives

            with tf.name_scope("flags"):
                # flags
                self.pl_clip_weight_deriv = tf.placeholder(dtype=bool, shape=(), name="clip_weight_deriv")  # if we clip weight derivatives
                self.pl_train_W = tf.placeholder(dtype=bool, shape=(), name="train_W")  # if we train network weights and biases
                self.pl_train_B = tf.placeholder(dtype=bool, shape=(), name="train_B")  # if we train feedback weights B
                self.pl_train_PI = tf.placeholder(dtype=bool, shape=(), name="train_PI")  # if we train inter-neuron to principal neuron weights
                self.pl_record_observables = tf.placeholder(dtype=bool, shape=(), name="record_observables")  # if we record the observables

    def _create_observables(self):
        """
        Make variables that can be recorded for cross-checks (but are not needed otherwise).
        """
        with tf.name_scope("observables"):
            self.obs_errors = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="obs_errors")  # observe neural errors
            self.obs_error_look_aheads = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="obs_error_look_aheads")  # observe neural error look aheads

    def _initialize_network(self, weight_init_params, use_biases, bias_init_params, use_interneurons, interneuron_backward_weight_scale):
        """
        Set up voltages, weights, and their derivatives.
        """
        self._set_random_seed(self.rnd_seed)
        self.sess = tf.Session()

        with tf.name_scope("states"):
            # setup voltage variable for neurons
            self.st_voltages = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="voltages")
            self.st_voltages_deriv = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="voltages_deriv")

            # setup bias variables for neurons
            self.st_biases = tf.Variable(self._create_initial_biases(use_biases, self.bias_mask, **bias_init_params), dtype=self.dtype, name="biases")
            self.st_biases_deriv = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="biases_deriv")

            # setup voltage variable for inter-neurons
            self.st_voltages_I = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="voltages_I")
            self.st_voltages_I_deriv = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="voltages_I_deriv")

            # setup weights variable for pyramidal neurons
            self.st_weights = tf.Variable(self._create_initial_weights(self.weight_mask, **weight_init_params), dtype=self.dtype, name="weights")
            self.st_weights_deriv = tf.Variable(self.weight_mask * 0., dtype=self.dtype, name="weights_deriv")  # initialize the weight derivatives to zero

            if use_interneurons:
                self.st_biases_I = tf.Variable(self._create_initial_biases(use_biases, self.bias_mask, **bias_init_params), dtype=self.dtype, name="biases_I")
                self.st_biases_I_deriv = tf.Variable(np.zeros(self.neuron_qty), dtype=self.dtype, name="biases_I_deriv")

                # setup weights from Pyr to IN
                self.st_weights_IP = tf.Variable(self._create_initial_weights(self.weight_mask, **weight_init_params), dtype=self.dtype, name="weights_IP")  # pyramidal to interneuron connection
                self.st_weights_IP_deriv = tf.Variable(self.weight_mask * 0., dtype=self.dtype, name="weights_IP_deriv")

                # setup weights from IN to Pyr
                self.st_weights_PI = tf.Variable(self._create_initial_weights(self.weight_mask, **weight_init_params).T, dtype=self.dtype, name="weights_PI")
                self.st_weights_PI_deriv = tf.Variable(self.weight_mask * 0., dtype=self.dtype, name="weights_PI_deriv")

                # setup backward weights
                self.st_weights_B = tf.Variable(self._create_initial_weights(self.weight_mask, **weight_init_params).T * interneuron_backward_weight_scale, dtype=self.dtype, name="weights_B")  # backward weights
                self.st_weights_B_deriv = tf.Variable(self.weight_mask * 0., dtype=self.dtype, name="weights_B_deriv")
            else:  # weight copying (THEY SHARE THE SAME TENSOR REFERENCE)
                self.st_biases_I = self.st_biases
                self.st_biases_I_deriv = self.st_biases_deriv

                self.st_weights_IP = self.st_weights
                self.st_weights_IP_deriv = self.st_weights_deriv

                self.st_weights_PI = tf.transpose(self.st_weights)
                self.st_weights_PI_deriv = tf.transpose(self.st_weights_deriv)

                self.st_weights_B = tf.transpose(self.st_weights)  # see below equation 24
                self.st_weights_B_deriv = tf.transpose(self.st_weights_deriv)

            self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Create randomly initialized weight matrix.
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), -clip, clip) * (weight_mask > 0)  # initialize weights with normal sample where mask is larger 0

    @staticmethod
    def _create_initial_biases(use_biases, bias_mask, mean, std, clip):
        """
        Create randomly initialized bias matrix (or set biases to zero if only weights are used).
        """
        neuron_qty = bias_mask.shape[0]
        if use_biases:
            return np.clip(np.random.normal(mean, std, size=neuron_qty), -clip, clip) * (bias_mask > 0)  # initialize biases with normal sample where mask is larger 0
        else:
            return np.zeros(neuron_qty)  # set biases to zero

    # BASIC PROFILE AND TEST METHODS
    ########################################
    def _profile_model(self):
        """
        Profiling for the comp. graph. Performs one network update step and saves the profile data.
        The data can be loaded in Chrome at chrome://tracing/.
        Adapted from Illarion Khlestov (https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d, Mar 23, 2017).
        """
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        feed_dict = self._return_feed_dict()
        self._set_input_and_target(np.zeros(self.input_size), np.zeros(self.target_size))

        for i in range(10):
            self.sess.run(self.network_updates, feed_dict, options=options, run_metadata=run_metadata)

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        with open('tf_timeline.json', 'w') as f:
            f.write(chrome_trace)

    def _write_basic_tensorboard(self):
        """
        Write some computational graph visualization in tensorboard.
        https://databricks.com/tensorflow/visualisation
        https://www.tensorflow.org/guide/graph_viz
        """
        writer = tf.summary.FileWriter("tensorboard", self.sess.graph)

        feed_dict = self._return_feed_dict()
        self._set_input_and_target(np.zeros(self.input_size), np.zeros(self.target_size))

        for i in range(10):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            self.sess.run(self.network_updates, feed_dict, options=options, run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step %d' % i)

        writer.close()

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
        print('Total number syn. connections: ', int(np.sum(self.weight_mask)))
        print('Layer structure: ', self.layers)
        print('Network architecture: ', self.network_architecture.name)
        print('Use inter-neurons: ', self.use_interneurons)
        print('Activation function: ', self.activation_function.name)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Bias initial distribution: ', self.arg_bias_init_params)
        print('Learning rate: ', self.arg_lrate_W)
        print('Beta (nudging parameter): ', self.arg_beta)
        print('Membrane time constant (Tau): ', self.arg_tau)
        print('Time step: ', self.arg_dt)
        print('Time per prediction step in test run: ', self.time_per_prediction_step, 's')
        print('Time per training step in test run: ', self.time_per_train_step, 's')
        print('------------------')
        print("Simulation framework: Tensorflow ", tf.__version__)
        print('Simulation running on:', 'gpu' if tf.test.is_gpu_available() else 'cpu')
        print('------------------')

    # BUILD THE COMPUTATIONAL GRAPH
    ##############################
    def _create_computational_graph(self, use_interneurons, dynamic_interneurons, use_biases, turn_off_visible_dynamics):
        """Unites the functions below to build the tensorflow graph to the following equations defining the ODE of the neural network dynamics:

        (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i
        (14.1)           r_i = \bar r_i + \tau \dot{\bar r}_i
        (14.2)           e_i = \bar e_i + \tau \dot{\bar e}_i

        (14.3)     \bar e_i  =  \bar r'_i\odot [ W_{i+1}^\T  (u_{i+1} - W_{i+1} \bar r_i)]
        (14.4)      \bar e_N = \beta (u_N^{trg}(t) - u_N(t))
        """

        # prepare tensors which will be reused in the graph quite often
        # get current activities and derivatives + the inputs each neuron receives
        rho, rho_deriv, rho_second_deriv, layerwise_input = self._get_activity_and_input(self.st_voltages, self.st_weights, self.st_biases)
        weights_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, self.st_weights)  # get the weights multiplied by rho', as written in Eq. 17.

        # collect network updates
        network_updates = []

        # Use either dynamic inter-neurons or instantaneous inter-neurons
        if dynamic_interneurons:  # dynamic inter-neurons integrate voltage through a diff. eq.
            with tf.name_scope("use_dynamic_interneurons"):
                network_updates.append(self.st_voltages_I.assign(self.st_voltages_I + self.pl_dt * self.st_voltages_I_deriv))  # i_voltage += i_voltage + i_voltage_deriv * dt, Eq. 32. Simple Euler equation
        else:  # non-dyn inter-neurons get voltage from weighted activity
            with tf.name_scope("use_nondynamic_interneurons"):
                network_updates.append(self.st_voltages_I.assign(self._calculate_layerwise_inputs(rho, self.st_weights_IP, self.st_biases_I)))  # i_voltage = activity * weight + bias, Eq. 33

        with tf.control_dependencies([network_updates[0]]):  # the following is dependent on current weights and voltages
            # CALCULATE WEIGHT AND BIAS DERIVATIVES
            # The weight update depends only on the current weights and voltages and can be updated first.
            with tf.name_scope("update_weight_derivatives"):
                if self.arg_clip_weight_derivs:  # if weight derivatives should be clipped
                    network_updates.append(self.st_weights_deriv.assign(tf.cond(self.pl_train_W,  # if weights should be trained
                                                                                lambda: tf.clip_by_value(check_nan(self._update_weights(self.st_voltages, rho, layerwise_input)), -self.pl_weight_deriv_clip_value, self.pl_weight_deriv_clip_value),  # do weights update
                                                                                lambda: np.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype))))  # else weights deriv is zero
                else:
                    network_updates.append(self.st_weights_deriv.assign(tf.cond(self.pl_train_W,  # if weights should be trained
                                                                                lambda: self._update_weights(self.st_voltages, rho, layerwise_input),  # do weights update
                                                                                lambda: np.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype))))  # else weights deriv is zero

            # calculate weight derivatives (with or without interneuron circuit)
            if use_interneurons:
                with tf.name_scope("calculate_interneurons"):
                    # derivative of backward weights
                    network_updates.append(self.st_weights_B_deriv.assign(tf.cond(self.pl_train_B,  # if backwards matrix should be trained
                                                                                  lambda: self._update_weights_B(self.st_voltages, self.st_weights_B),  # do weights update with Eq. 31
                                                                                  lambda: np.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype))))  # else weights deriv is zero

                    # derivative of PI weights
                    network_updates.append(self.st_weights_PI_deriv.assign(tf.cond(self.pl_train_PI,
                                                                                   lambda: self._update_weights_PI(self.st_voltages, self.st_weights_B, self.st_weights_PI,
                                                                                                                   self.st_weights_IP, rho, self.st_biases_I),  # do weights update with Eq. 34
                                                                                   lambda: np.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype))))  # else weights deriv is zero

                    # precalculate weights with rho derivative
                    # perform transpose
                    weights_B_rho_deriv = tf.transpose(self._dress_weights_with_rho_deriv(rho_deriv, tf.transpose(self.st_weights_B)))  # we just component-wise multiply the weight by rho_deriv (Eq. 36)
                    weights_PI_rho_deriv = tf.transpose(self._dress_weights_with_rho_deriv(rho_deriv, tf.transpose(self.st_weights_PI)))  # same
                    weights_IP_rho_deriv = self._dress_weights_with_rho_deriv(rho_deriv, self.st_weights_IP)  # same

            # without inter-neurons, we use weight transport as described in the theory.
            else:
                # perform transpose
                with tf.name_scope("use_weight_transport"):
                    weights_B_rho_deriv = tf.transpose(weights_rho_deriv)  # see Eq. 24 for relationships.
                    weights_PI_rho_deriv = tf.transpose(weights_rho_deriv)  # same. Recall that top-down weights must be equal to PI weights
                    weights_IP_rho_deriv = weights_rho_deriv

            # calculate bias derivatives
            if use_biases:  # if we use biases
                with tf.name_scope("calculate_biases"):
                    network_updates.append(self.st_biases_deriv.assign(tf.cond(self.pl_train_W,  # if biases should be trained
                                                                               lambda: self._update_biases(self.st_voltages, rho, layerwise_input),  # do bias update
                                                                               lambda: np.zeros(self.neuron_qty, dtype=self.dtype))))  # else bias deriv is zero

                    # calculate derivative of interneuron biases
                    if use_interneurons:
                        with tf.name_scope("of_interneurons"):
                            network_updates.append(self.st_biases_I_deriv.assign(tf.cond(self.pl_train_PI,
                                                                                         lambda: self._update_biases_I(self.st_voltages, self.st_weights_B, self.st_weights_PI,
                                                                                                                       self.st_weights_IP, rho, self.st_biases_I),
                                                                                         lambda: np.zeros(self.neuron_qty, dtype=self.dtype))))  # else inter-neuron bias deriv is zero

            # CALCULATE VOLTAGE DERIVATIVES
            # build the linear equations to calculate the voltage derivatives
            # get the dendritic error, which is the mismatch between prediction and reality
            dendritic_error = self._calculate_dendritic_error(self.st_voltages, rho, self.st_weights_B, self.st_weights_PI, self.st_weights_IP, self.st_biases_I)  # Eq. 24

            # We use the dendritic_error for the next computation, described by Eq. 17 (and further explanations below). We obtain a matrix and error that we will use for solving the linear system.
            # get error matrix, being ..., and error vector, being ..
            err_matrix, err_vector = self._get_error_terms(self.st_voltages, rho, rho_deriv, rho_second_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv, dendritic_error)

            # get error_w_deriv, being ..., and input_w_deriv, being ...
            err_w_deriv, inp_w_deriv = self._get_weight_derivative_terms(self.st_voltages, rho, rho_deriv,
                                                                         self.st_weights_deriv, self.st_weights_B_deriv,
                                                                         self.st_weights_PI_deriv,
                                                                         self.st_weights_IP_deriv, self.st_weights_PI,
                                                                         self.st_weights_IP, self.st_biases,
                                                                         self.st_biases_deriv,
                                                                         self.st_biases_I, self.st_biases_I_deriv)

            # get input_matrix, being ..., and input vector, being ...
            input_matrix, input_vector = self._get_layer_input_terms(self.st_voltages, layerwise_input, rho_deriv, weights_rho_deriv, self.st_weights_PI)

        # calculate the voltage derivatives
        # if we clamped the input, i.e. r_1 = visible input activity
        with tf.control_dependencies(network_updates):
            with tf.name_scope("update_voltage_derivatives"):
                if turn_off_visible_dynamics:
                    # we store the matrices M(u) supposed to be multiplied by u_dot in a big matrix
                    full_matrix = err_matrix + input_matrix
                    # We write a formal derivative for the clamped input. Indeed, in this case, we do not need to compute u_dot, as we already know it for input neurons.
                    # So only for input neurons, we force the derivative to take the value of pl_visible_input
                    visibles_du = (self.pl_visible_input - self.st_voltages[:self.input_size]) / self.pl_dt
                    # Here, we simply write M(u)*visibles_du (linear system) by cutting M(u) so we multiply the corresponding visible derivative (:self.input_size)
                    # only at the level of the rest of the network (self.input_size:). This will be added to b(u) as it is now a vector, and not anymore in M(u) to solve the system
                    # The rest of the matrix product will be done with the classical u_dot that is unknown to solve the system
                    visible_contr = tf_mat_vec_dot(full_matrix[self.input_size:, :self.input_size], visibles_du)
                    # now we want to solve the system and store u_dot for all neurons. We already know it for the visible neurons so we do not need to solve in that case, we directly
                    # store visibles_du computed above. We concatenate it with the rest of the u_dot we solve as it is unknown.
                    # to solve them, we use the full_matrix M(u) that ONLY concerns the non-input neurons, and the vectors b(u) + visible_contr computed above only for non-input neurons.
                    network_updates.append(self.st_voltages_deriv.assign(tf.concat([visibles_du, self.solve(full_matrix[self.input_size:, self.input_size:],
                                                                                                            (err_vector + input_vector + err_w_deriv + inp_w_deriv)[
                                                                                                            self.input_size:] - visible_contr)], axis=0)))
                else:
                    # here, the input neurons are nudged, so we do not know their dynamics. We then solve the whole system altogether.
                    network_updates.append(self.st_voltages_deriv.assign(self.solve(err_matrix + input_matrix, err_vector + input_vector + err_w_deriv + inp_w_deriv)))

        # record observables for displaying network internals (if needed, otherwise return crap and do nothing).
        with tf.control_dependencies(network_updates):
            with tf.name_scope("record_observables"):
                # the following corresponds to Eq. 17. e_i = r_i' * dendritic_error + e_i^trgt (only for output neurons)
                network_updates.append(tf.cond(self.pl_record_observables,  # if we record observables
                                               lambda: self.obs_errors.assign(rho_deriv * dendritic_error + tf.concat(
                                                   [np.zeros(self.neuron_qty - self.target_size), self.pl_beta * (self.pl_target - self.st_voltages[-self.target_size:])], axis=0)),
                                               lambda: np.array([0.0], dtype=self.dtype)))

                # the following corresponds to Eq. 16, with look ahead errors, as a function of the above-computed errors.
                network_updates.append(tf.cond(self.pl_record_observables,  # if we record observables
                                               lambda: self.obs_error_look_aheads.assign(err_vector - tf_mat_vec_dot(err_matrix, self.st_voltages_deriv) + err_w_deriv),
                                               lambda: np.array([0.0], dtype=self.dtype)))

        # update interneuron voltage derivatives if dynamic interneurons are used
        if dynamic_interneurons:
            with tf.control_dependencies(network_updates):
                with tf.name_scope("update_dynamic_interneurons"):
                    # here we add the derivative of the interneuron voltages.
                    network_updates.append(self.st_voltages_I_deriv.assign(self._get_interneuron_dynamics_derivatives(self.st_voltages_I, rho + self.pl_tau * rho_deriv * self.st_voltages_deriv, rho, self.st_weights_IP,
                                                                                                                      self.st_weights_IP_deriv, self.st_biases_I, self.st_biases_I_deriv)))

        # UPDATE NETWORK STATE
        # update voltages and weights with the derivatives calculated earlier
        # this describes a simple Euler integration to get the next voltages and weights
        # EULER EQUATIONS
        with tf.control_dependencies(network_updates):
            with tf.name_scope("update_voltages_and_weights"):
                network_updates.append(self.st_voltages.assign(self.st_voltages + self.pl_dt * self.st_voltages_deriv))  # voltages += dt * voltages_deriv (EULER)
                network_updates.append(self.st_weights.assign(self.st_weights + self.pl_dt * self.st_weights_deriv))  # weights += dt * weights_deriv

                # update interneurons
                if use_interneurons:
                    network_updates.append(self.st_weights_B.assign(self.st_weights_B + self.pl_dt * self.st_weights_B_deriv))
                    network_updates.append(self.st_weights_PI.assign(self.st_weights_PI + self.pl_dt * self.st_weights_PI_deriv))

                # update biases
                if use_biases:
                    network_updates.append(self.st_biases.assign(self.st_biases + self.pl_dt * self.st_biases_deriv))  # biases += dt * biases_deriv

                    if use_interneurons:
                        network_updates.append(self.st_biases_I.assign(self.st_biases_I + self.pl_dt * self.st_biases_I_deriv))  # inter-neuron_biases += dt * inter-neuron_biases_deriv

        # return the computation graph
        return network_updates

    # FUNCTIONS OF SMALL CALCULATIONS NEEDED TO SOLVE ODE
    ########################################################

    def _get_activity_and_input(self, voltages, weights, biases):
        """
        Return neural activity + derivatives thereof as well as the synaptic input of each neuron.
        Useful for computations u - Wr
        """
        with tf.name_scope("get_activity_and_input"):
            rho = self.act_function(voltages)  # neural activity
            rho_deriv = self.act_func_deriv(voltages)  # derivative of neural activity
            rho_second_deriv = self.act_func_second_deriv(voltages)  # second derivative of neural activity
            layerwise_inputs = self._calculate_layerwise_inputs(rho, weights, biases)  # all the inputs to the all neurons
            return rho, rho_deriv, rho_second_deriv, layerwise_inputs

    def _calculate_layerwise_inputs(self, rho, weights, biases):
        """
        Returns external+network input for all neurons W * r + x.
        """
        with tf.name_scope("calculate_layerwise_inputs"):
            internal_input = tf_mat_vec_dot(weights, rho) + biases  # activities * network_weights + biases
            external_input = internal_input[:self.input_size] + self.pl_visible_input  # internal input + external input = total input
            return tf.concat([external_input, internal_input[self.input_size:]], axis=0)  # concat the row receiving external input with the others

    def _set_input_and_target(self, input_voltage, output_voltage):
        """
        Sets input and output and ensures proper transition of single sample and batch prediction
        Note: We need the input of the previous time step to approximate the derivatives of the inputs.
        """

        # set (formerly) current as old and new input as current
        self.arg_old_input = self.arg_input
        self.arg_input = input_voltage

        self.arg_old_target = self.arg_target
        self.arg_target = output_voltage

    def _calculate_dendritic_error(self, voltages, rho, weights_B, weights_PI, weights_IP, biases_I):
        """
        Calculates difference between what we can explain away and what is left as error:
            (B * u - W_PI * W_IP * r) = B * u-W_PI * U_I = e.  r_i' terms are already included
        """
        with tf.name_scope("calculate_dendritic_error"):
            return tf_mat_vec_dot(weights_B, voltages) - tf_mat_vec_dot(weights_PI, self.st_voltages_I)  # back projected neuron output - inter-neuron output, st_voltages_I = W_IP.r_i according to Eq. 33

    @staticmethod
    def _dress_weights_with_rho_deriv(rho_deriv, weights):
        """
        Dress weights with the derivative of the act. function. See Eq. 17
        """
        with tf.name_scope("dress_weights_with_rho_deriv"):
            return weights * rho_deriv

    # INDIVIDUAL PARTS OF THE ODE
    ###############################

    # The ODE is rewritten in the form matrix * voltages_deriv = vector.
    # Thus, we can solve for voltages_deriv by solving this linear system.
    def _get_error_terms(self, voltages, rho, rho_deriv, rho_second_deriv, weights_B_rho_deriv, weights_PI_rho_deriv, weights_IP_rho_deriv, dendritic_error):
        """
        Calculates all terms originating from the error e = e_bar + tau*d(e_bar)/dt.
        e can be recovered by calculating matrix_comp*voltage_derivative+vector_comp, i.e., e was divided into a part depending
        on the voltage derivates and an independent part. M.u_derivative + b(u) (cf drawings)
        """
        with tf.name_scope("get_error_terms"):
            matrix_comp = -self.pl_tau * (weights_B_rho_deriv - tf.matmul(weights_PI_rho_deriv, weights_IP_rho_deriv, a_is_sparse=self.use_sparse_mult, b_is_sparse=self.use_sparse_mult))  # TODO: for it to be the hessian and jacobian, this should be -(r_bar_deriv.reshape(-1, 1) * weights_B - torch.matmul(r_bar_deriv.reshape(-1, 1) * weights_PI, weights_IP * r_bar_deriv))
            matrix_comp += -self.pl_tau * tf.matrix_diag(rho_second_deriv * dendritic_error)
            target_term = np.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype)
            np.fill_diagonal(target_term[-self.target_size:, -self.target_size:], 1)
            beta_factors = tf.concat([np.zeros(sum(self.layers[:-1])), self.pl_beta], axis=0)
            matrix_comp += target_term * beta_factors * self.pl_tau

            vector_comp = rho_deriv * dendritic_error
            deriv = tf.concat([self.pl_tau / self.pl_dt * (self.pl_visible_input - self.pl_old_visible_input), np.zeros(self.neuron_qty - self.input_size)], axis=0)
            vector_comp += -tf_mat_vec_dot(weights_PI_rho_deriv, deriv)
            vector_comp += -rho_deriv * tf_mat_vec_dot(self.st_weights_PI, self._calculate_layerwise_inputs(rho, self.st_weights_IP, self.st_biases_I) - self.st_voltages_I)  # Eq. 17

            target_vector_comp = vector_comp[-self.target_size:] \
                                 + self.pl_beta * (self.pl_target - voltages[-self.target_size:]) \
                                 + self.pl_tau * self.pl_beta * (self.pl_target - self.pl_old_target) / self.pl_dt

            return matrix_comp, tf.concat([vector_comp[:-self.target_size], target_vector_comp], axis=0)

    def _get_weight_derivative_terms(self, voltages, rho, rho_deriv, weights_deriv, weights_B_deriv, weights_PI_deriv, weights_IP_deriv, weights_PI, weights_IP, biases,
                                     biases_deriv, biases_I, biases_I_deriv):
        """
        Terms originating from taking the derivatives of the weights. Completely in vector form (i.e., independent of the derivative of the voltages).
        """
        with tf.name_scope("get_weight_derivative_terms"):
            error_comp = self.pl_tau * rho_deriv * (tf_mat_vec_dot(weights_B_deriv, voltages) - tf_mat_vec_dot(weights_PI, tf_mat_vec_dot(weights_IP_deriv, rho)))
            error_comp += -self.pl_tau * rho_deriv * (tf_mat_vec_dot(weights_PI_deriv, self.st_voltages_I) + tf_mat_vec_dot(weights_PI, biases_I_deriv))
            input_comp = self.pl_tau * (tf_mat_vec_dot(weights_deriv, rho) + biases_deriv)

            return error_comp, input_comp

    def _get_layer_input_terms(self, voltages, layerwise_inputs, rho_deriv, weights_rho_deriv, weights_PI):
        """
        Terms originating from external and synaptic inputs.
        """
        with tf.name_scope("get_layer_input_terms"):
            matrix_comp = self.pl_tau * (np.identity(self.neuron_qty) - weights_rho_deriv)
            vector_comp = layerwise_inputs - voltages
            ext_vector_comp = vector_comp[:self.input_size] + self.pl_tau / self.pl_dt * (self.pl_visible_input - self.pl_old_visible_input)

        return matrix_comp, tf.concat([ext_vector_comp, vector_comp[self.input_size:]], axis=0)

    def _get_interneuron_dynamics_derivatives(self, voltage_I, rho_look_ahead, rho, weights_IP, weights_IP_deriv, biases_I, biases_I_deriv):
        """
        Calculate interneuron dynamics if they are enabled.
        """
        with tf.name_scope("get_interneuron_dynamics"):
            # internal input = - current interneuron voltage + layerwise inputs to interneuron + tau * (bias derivative + weights derivative * activities)
            # left part corresponds to Eq. 32 with lookahead rates.
            internal_input = - voltage_I + self._calculate_layerwise_inputs(rho_look_ahead, weights_IP, biases_I) + self.pl_tau * (biases_I_deriv + tf_mat_vec_dot(weights_IP_deriv, rho))  # Eq. 32

            # external input = internal input to input layer + tau / dt * input derivative, for the derivative of interneuron voltages of input neurons, it can be expressed explicitly by using the visible_input variables (definition of derivative)
            external_input = internal_input[:self.input_size] + self.pl_tau / self.pl_dt * (self.pl_visible_input - self.pl_old_visible_input)

            # we divide by tau to obtain u_dot^I in Eq. 32 -> what is called interneuron dynamics.
            return 1. / self.pl_tau * tf.concat([external_input, internal_input[self.input_size:]], axis=0)  # 1 / tau * total_input

    # WEIGHT DERIVATIVES
    ######################

    def _update_weights(self, voltages, rho, layerwise_inputs):
        """
        Weight update of the network weights. See Eq. 18
        """
        with tf.name_scope("update_weights"):
            # multiply by weight_mask to keep only existing connections
            return tf_outer_product(voltages - layerwise_inputs, rho  # (voltages - layer input mismatch) * activities
                                    ) * self.weight_mask * self.pl_lrate_W  # * weight mask * learning rate

    def _update_biases(self, voltages, rho, layerwise_inputs):
        """
        Bias update of the network biases. Same as weights_update but without pre-synaptic activities (Eq. 18)
        """
        return (voltages - layerwise_inputs) * self.pl_lrate_biases * self.bias_mask

    def _update_biases_I(self, voltages, weights_B, weights_PI, weights_IP, rho, biases_I):
        """
        Bias update of the inter-neuron biases. Same as weights_PI_update without pre-synaptic voltages (Eq. 34)
        TODO: Interneuron bias update looks odd (mismatch has no brackets)
        """
        return tf_mat_vec_dot(weights_B, voltages) - tf_mat_vec_dot(weights_PI, self.st_voltages_I) * self.bias_mask * self.pl_lrate_biases_I

    def _update_weights_B(self, voltages, weights_B):  # = mismatch * network voltages * network weights * learning rate
        """
        Weight update of the generative weights of the interneuron circuit. See Eq. 31
        """
        return tf_outer_product(voltages  # #                                   network voltages
                                - tf_mat_vec_dot(weights_B, voltages),  # #     weighted generative voltages
                                voltages  # #                                   ^^mismatch * network voltages
                                ) * self.weight_mask.T * self.pl_lrate_B  # #   * all respective network weights * learning rate

    def _update_weights_PI(self, voltages, weights_B, weights_PI, weights_IP, rho, biases_I):  # = mismatch * interneuron voltage * weight mask * learning rate
        """
        Weight update of the interneuron to pyramidal weights (to learn a self-predicting state). See Eq. 34
        """
        return tf_outer_product(
            tf_mat_vec_dot(weights_B, voltages)  # #                 weighted network voltages
            - tf_mat_vec_dot(weights_PI, self.st_voltages_I),  # #   weighted interneuron voltages
            self.st_voltages_I  # #                                      ^^mismatch * interneuron voltage
        ) * self.weight_mask.T * self.pl_lrate_PI  # #                       * all respective network connections * learning rate

    # NETWORK INTEGRATOR
    ######################

    def _return_feed_dict(self, train_W=False, train_B=False, train_PI=False, record_observables=False):
        """
        Dict with flexible parameters that can be fed into the comp. graph (e.g. time constants, learning rates, whether weights are trained, etc.).
        """
        return {self.pl_dt: self.arg_dt, self.pl_tau: self.arg_tau,  # time step and time constant tau
                self.pl_beta: self.arg_beta,  # nudging beta
                self.pl_lrate_W: self.arg_lrate_W, self.pl_lrate_biases: self.arg_lrate_biases, self.pl_lrate_B: self.arg_lrate_B,  # learning rates
                self.pl_lrate_PI: self.arg_lrate_PI, self.pl_lrate_biases_I: self.arg_lrate_biases_I,  # more learning rates
                self.pl_visible_input: self.arg_input, self.pl_old_visible_input: self.arg_old_input,  # inputs
                self.pl_target: self.arg_target, self.pl_old_target: self.arg_old_target,  # targets
                self.pl_weight_deriv_clip_value: self.arg_weight_deriv_clip_value,  # config flags
                self.pl_train_W: train_W, self.pl_train_B: train_B, self.pl_train_PI: train_PI, self.pl_record_observables: record_observables}  # config flags

    def update_network(self, input_voltage, output_voltage, train_W=False, train_B=False, train_PI=False, record_observables=False):
        """
        Perform a single integration step.
        """
        self._set_input_and_target(input_voltage, output_voltage)
        feed_dict = self._return_feed_dict(train_W, train_B, train_PI, record_observables)
        self.sess.run(self.network_updates, feed_dict)

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

        self.arg_beta = beta

    def set_lrate_W(self, learning_rate):
        self.arg_lrate_W = learning_rate

    def set_lrate_B(self, learning_rate):
        self.arg_lrate_B = learning_rate

    def set_lrate_biases(self, learning_rate):
        self.arg_lrate_biases = learning_rate

    def set_lrate_biases_I(self, learning_rate):
        self.arg_lrate_biases_I = learning_rate

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)
        tf.set_random_seed(rnd_seed)

    def get_voltages(self):
        return self.sess.run(self.st_voltages)

    # BASIC KERAS INTERFACE
    def fit(self, x=None, y=None, n_updates: int = 100, batch_size=1, epochs=1, verbose=1, is_timeseries=False):
        """
        Train network on dataset. BATCH TRAINING IS NOT IMPLEMENTED.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per sample.
            batch_size: Number of examples per batch (batch training NOT IMPLEMENTED).
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        """
        n_samples = len(x)  # dataset size
        xr = np.linspace(-20, 90, n_updates)

        if self.is_beta_zero:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(0.1))
            self.set_beta(0.1)  # turn nudging on to enable training

        for epoch_i in range(epochs):
            for sample_i in range(n_samples):
                if verbose >= 1:
                    print("train:: sample ", sample_i, "/", n_samples, " | update ", end=" ")

                for update_i in range(n_updates):
                    if verbose >= 2 and update_i % 10 == 0:
                        print(update_i, end=" ")

                    if is_timeseries:
                        sample, label = x[sample_i], y[sample_i]
                    else:
                        sample, label = self._decay_func(xr[update_i], x[sample_i - 1], x[sample_i]), self._decay_func(xr[update_i], y[sample_i - 1], y[sample_i])
                    self.update_network(sample, label, train_W=True, train_B=False, train_PI=False, record_observables=False)

                if verbose >= 1:
                    print('')

    def fit_batch(self, x, y, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose=1, is_timeseries=False):
        """
        TODO: NOT IMPLEMENTED
        Args:
            x:
            y:
            n_updates:
            verbose:
            is_timeseries:

        Returns:

        """
        return None

    def predict(self, x, n_updates: int = 100, batch_size=1, verbose: int=1, is_timeseries: bool=False):
        """
        Predict batch with trained network. BATCH PREDICTION IS NOT IMPLEMENTED.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Size of batch to be predicted.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        :return:
        """
        n_samples = len(x)  # dataset size
        xr = np.linspace(-20, 90, n_updates)
        self.set_beta(0.0)  # turn nudging off to disable learning

        dummy_label = np.zeros(self.target_size)

        predictions = []
        for sample_i in range(n_samples):
            if verbose >= 1:
                print("predict:: sample", sample_i, "/", n_samples, " | update ", end=" ")

            for update_i in range(n_updates):
                if verbose >= 2 and update_i % 10 == 0:
                    print(update_i, end=" ")

                if is_timeseries:
                    sample = x[sample_i]
                else:
                    sample = self._decay_func(xr[update_i], x[sample_i - 1], x[sample_i])
                self.update_network(sample, dummy_label, train_W=False, train_B=False, train_PI=False, record_observables=False)

            if verbose >= 1:
                print('')

            volts = self.sess.run(self.st_voltages)
            prediction = volts[-self.target_size:]
            predictions.append(prediction)

        return predictions

    def predict_batch(self, x, n_updates=100, batch_iteration=-1, batch_qty=-1, verbose=1, is_timeseries=False):
        """
        TODO: NOT IMPLEMENTED
        Args:
            x:
            n_updates:
            verbose:
            is_timeseries:

        Returns:

        """
        return None

    def __call__(self, x, n_updates: int = 100, verbose=1, is_timeseries=False):
        self.predict_batch(x, n_updates, verbose, is_timeseries)

    # SAVE AND LOAD NETWORK
    def save(self, save_path):
        """
        Save the lagrange model to file.
        Args:
            save_path:

        Returns:

        """
        voltages = self.sess.run(self.st_voltages)
        weights = self.sess.run(self.st_weights)
        biases = self.sess.run(self.st_biases)

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

        self.sess.run(self.st_voltages.assign(voltages))
        self.sess.run(self.st_weights.assign(weights))
        self.sess.run(self.st_biases.assign(biases))

        return True

    # TIME SERIES BUILDER FUNCTIONS
    @staticmethod
    def _decay_func(x, equi1, equi2):
        """
        Decay function to transform sample and label by exponential fading.
        Sample x from xr2 = np.linspace(-20, 90, n_updates)
        """
        o = np.array(equi1 + (equi2 - equi1) / (1 + np.exp(-x / 4.0)))

        if o.shape == ():  # turn shape () into shape (1,)
            o = np.expand_dims(o, axis=0)

        return o

