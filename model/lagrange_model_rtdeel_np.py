# Numpy implementation of the Lagrange model. Trainable only with single samples. Just for reference.

# Authors: Dominik Dold (dodo47@github) and Benjamin Ellenberger (benelot@github)


import numpy as np
from numpy.linalg import solve
import time

# network parameters
from model.network_params import ArchType, ActivationFunction, NetworkParams


class LagrangeNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Lagrange model.
    """

    # INIT
    ########
    def __init__(self, params: NetworkParams):

        # store important params
        self.layers = params.layers
        self.tau = params.arg_tau
        self.dt = params.arg_dt

        self.arg_lrate_W = params.arg_lrate_W
        self.arg_lrate_B = params.arg_lrate_W_B
        self.arg_lrate_PI = params.arg_lrate_PI

        self.arg_w_init_params = params.arg_w_init_params

        self.dtype = params.dtype
        self.is_recurrent = False if params.network_architecture == ArchType.LAYERED_FEEDFORWARD else True
        self.use_interneurons = params.use_interneurons
        self.activation_function = params.activation_function
        self.rnd_seed = params.rnd_seed

        self.beta = params.arg_beta

        # setup inputs and outputs
        self.input_size = params.layers[0]
        self.target_size = params.layers[-1]
        self.neuron_qty = sum(params.layers)
        self.input = np.zeros(self.input_size)
        self.old_input = np.zeros(self.input_size)
        self.target = np.zeros(self.target_size)
        self.old_target = np.zeros(self.target_size)

        self._set_random_seed(params.rnd_seed)

        # setup neuron weight mask
        self.weight_mask = self._make_connection_weight_mask(params.layers, params.learning_rate_factors, feedback=self.is_recurrent)

        # setup activation function
        self.act_function, self.act_func_deriv, self.act_func_second_deriv = self._generate_activation_function(params.activation_function)

        # setup network
        self._initialize_network(params.arg_w_init_params, params.use_interneurons)

        # perform single sample test run to get network stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # reinitialize network variables
        self._initialize_network(params.arg_w_init_params, params.use_interneurons)

    # NETWORK SETUP METHODS
    ###########################
    @staticmethod
    def _make_connection_weight_mask(layers, learning_rate_factors, feedback=False):
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
        neuron_qty = np.sum(layers)
        layer_qty = len(layers)
        mask = np.zeros((neuron_qty, neuron_qty))
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]
        for i in range(layer_qty - 1):
            mask[layer_offsets[i + 1]:layer_offsets[i + 2], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]

        return mask + feedback * mask.T

    @staticmethod
    def _generate_activation_function(activation_function):
        """
        Implementation of different activation functions.
        """
        if activation_function == ActivationFunction.SIGMOID:
            act_function = lambda voltages: 1. / (1 + np.exp(-voltages))  # define the activation function as a sigmoid of voltages
            act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))  # function of the 1st derivative
            act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))  # function of the 2nd derivative
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv

    def _initialize_network(self, weight_init_params, use_interneurons):
        """
        Set up voltages, weights, and their derivatives.
        """
        self._set_random_seed(self.rnd_seed)

        # states
        # setup voltage variable for neurons
        self.voltages = np.zeros(self.neuron_qty, dtype=self.dtype)
        self.voltages_deriv = np.zeros(self.neuron_qty, dtype=self.dtype)

        # setup bias variables for neurons
        self.weights = self._create_initial_weights(self.weight_mask, **weight_init_params)
        self.weights_deriv = self.weight_mask * 0.

        if use_interneurons:
            # setup weights from Pyr to IN
            self.weights_IP = self._create_initial_weights(self.weight_mask, **weight_init_params)
            self.weights_IP_deriv = 0.

            # setup weights from IN to Pyr
            self.weights_PI_transp = self._create_initial_weights(self.weight_mask, **weight_init_params)
            self.weights_PI_deriv = self.weight_mask * 0.

            self.weights_B_transp = self._create_initial_weights(self.weight_mask, **weight_init_params)
            self.weights_B_deriv = self.weight_mask * 0.
        # # else:  weight copying

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Create randomly initialized weight matrix.
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), -clip, clip) * (weight_mask > 0)

    def _test_simulation_run(self):
        """
        Test network run. Estimates the average time used to perform a time/integration step.
        """
        sample_size = 50.0  # number of samples to calculate the average integration time of

        sample_input = np.ones(self.input_size)  # input voltages set to 1
        sample_output = np.ones(self.target_size)  # output voltages set to 1

        # test prediction
        init_time = time.time()  # initial time
        for i in range(sample_size):
            self.update_network(sample_input, sample_output, False)  # run sample_size prediction updates of the network

        time_per_prediction_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        # test training
        init_time = time.time()
        for i in range(sample_size):
            self.update_network(sample_input, sample_output, True) # run sample_size training updates of the network

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
        print('Network architecture recurrent: ', self.is_recurrent)
        print('Use inter-neurons: ', self.use_interneurons)
        print('Activation function: ', self.activation_function)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Learning rate: ', self.arg_lrate_W)
        print('Beta (nudging parameter): ', self.beta)
        print('Membrane time constant (Tau): ', self.tau)
        print('Timestep: ', self.dt)
        print('Time per prediction step in test run: ', self.time_per_prediction_step, 's')
        print('Time per training step in test run: ', self.time_per_train_step, 's')
        print('------------------')
        print("Simulation framework: Numpy ", np.__version__)
        print('Simulation running on :cpu')
        print('------------------')

    # PERFORM UPDATE STEP OF NETWORK DYNAMICS
    ##############################
    def calculate_network_state(self, input_voltage, output_voltage, train_W=False, return_error=False):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

       (14.0) \tau \dot u_i = - u_i + W_i r_{i-1} +  e_i
       (14.1)           r_i = \bar r_i + \tau \dot{\bar r}_i
       (14.2)           e_i = \bar e_i + \tau \dot{\bar e}_i

       (14.3)     \bar e_i  =  \bar r'_i\odot [ W_{i+1}^\T  (u_{i+1} - W_{i+1} \bar r_i)]
       (14.4)      \bar e_N = \beta (u_N^{trg}(t) - u_N(t))
       """
        # prepare tensors which will be reused in the graph quite often
        # get current activities and derivatives + the inputs each neuron receives
        self._set_input_and_target(input_voltage, output_voltage)
        rho, rho_deriv, rho_scnd_deriv, l_input = self._get_activity_and_input(self.voltages, self.weights)
        weights_dr = self._dress_weights_with_rho_deriv(rho_deriv, self.weights)

        # CALCULATE WEIGHT AND BIAS DERIVATIVES
        # The weight update depends only on the current weights and voltages and can be updated first.
        if train_W:
            self.weights_deriv = self._update_weights(self.voltages, rho, l_input)
        else:
            self.weights_deriv = np.zeros((self.neuron_qty, self.neuron_qty))

        # calculate weight derivatives (with or without interneuron circuit)
        if self.use_interneurons:
            self.weights_B_deriv = self._update_weights_B(self.voltages, self.weights_B_transp)
            self.weights_PI_deriv = self._update_weights_PI(self.voltages, self.weights_B_transp, self.weights_PI_transp, self.weights_IP, rho)

            # precalculate weights with rho derivative
            weights_B_transp_dr = self._dress_weights_with_rho_deriv(rho_deriv, self.weights_B_transp)
            weights_PI_transp_dr = self._dress_weights_with_rho_deriv(rho_deriv, self.weights_PI_transp)
            weights_IP_dr = self._dress_weights_with_rho_deriv(rho_deriv, self.weights_IP)

            dendr_error = self._calculate_dendritic_error(self.voltages, rho, self.weights_B_transp, self.weights_PI_transp, self.weights_IP)
            err_matrix, err_vector = self._get_error_terms(self.voltages, rho_deriv, rho_scnd_deriv, weights_B_transp_dr, weights_PI_transp_dr, weights_IP_dr, dendr_error)
            err_wderiv, inp_wderiv = self._get_weight_derivative_terms(self.voltages, rho, rho_deriv, self.weights_deriv, self.weights_B_deriv, self.weights_PI_deriv,
                                                                       self.weights_IP_deriv, self.weights_PI_transp, self.weights_IP)
            inp_matrix, inp_vector = self._get_layer_input_terms(self.voltages, l_input, rho_deriv, weights_dr, self.weights_PI_transp)

        # without inter-neurons, we use weight transport as described in the theory.
        else:
            dendr_error = self._calculate_dendritic_error(self.voltages, rho, self.weights, self.weights, self.weights)
            err_matrix, err_vector = self._get_error_terms(self.voltages, rho_deriv, rho_scnd_deriv, weights_dr, weights_dr, weights_dr, dendr_error)
            err_wderiv, inp_wderiv = self._get_weight_derivative_terms(self.voltages, rho, rho_deriv, self.weights_deriv, self.weights_deriv, self.weights_deriv, self.weights_deriv,
                                                                       self.weights, self.weights)
            inp_matrix, inp_vector = self._get_layer_input_terms(self.voltages, l_input, rho_deriv, weights_dr, self.weights)

        self.voltages_deriv = solve(err_matrix + inp_matrix, err_vector + inp_vector + err_wderiv + inp_wderiv)

        err_, err = None, None
        if return_error:
            err_ = rho_deriv * dendr_error
            err_[-self.target_size:] += self.beta * (self.target - self.voltages[-self.target_size:])
            err = err_vector - np.dot(err_matrix, self.voltages_deriv) + err_wderiv
        return err_, err

    # FUNCTIONS OF SMALL CALCULATIONS NEEDED TO SOLVE ODE
    ########################################################
    def _get_activity_and_input(self, voltages, weights):
        """
        Return neural activity+derivatives thereof as well as the synaptic input of each neuron.
        Useful for computations u - Wr
        """
        rho = self.act_function(voltages)
        rho_deriv = self.act_func_deriv(voltages)
        rho_scnd_deriv = self.act_func_second_deriv(voltages)
        l_input = self._calculate_layerwise_inputs(rho, weights)
        return rho, rho_deriv, rho_scnd_deriv, l_input

    def _calculate_layerwise_inputs(self, rho, weights):
        """
        Returns external+network input for all neurons W * r + x.
        """
        inp = np.dot(weights, rho)
        inp[:self.input_size] += self.input
        return inp

    def _set_input_and_target(self, input_voltage, output_voltage):
        """
        Sets input and output and ensures proper transition of single sample and batch prediction
        Note: We need the input of the previous time step to approximate the derivatives of the inputs.
        """
        self.old_target = self.target
        self.target = output_voltage
        self.old_input = self.input
        self.input = input_voltage

    def _calculate_dendritic_error(self, voltages, rho, weights_B_transp, weights_PI_transp, weights_IP):
        """
        Calculates difference between what we can explain away and what is left as error:
            (B*u-W_PI*W_IP*r) = B * u-W_PI * U_I = e.  r_i' terms are already included
        """
        return np.dot(weights_B_transp.T, voltages) - np.dot(weights_PI_transp.T, self._calculate_layerwise_inputs(rho, weights_IP))

    @staticmethod
    def _dress_weights_with_rho_deriv(rho_deriv, weights):
        """
        Dress weights with the derivative of the act. function. See Eq. 17
        """
        return weights * rho_deriv

    # INDIVIDUAL PARTS OF THE ODE
    ###############################
    # The ODE is rewritten in the form matrix*voltages_deriv = vector.
    # Thus, we can solve for voltages_deriv by solving this linear system.
    def _get_error_terms(self, voltages, rho_deriv, rho_scnd_deriv, weights_B_transp_dr, weights_PI_transp_dr, weights_IP_dr, dendritic_error):
        """
        Calculates all terms originating from the error e = e_bar + tau*d(e_bar)/dt.
        e can be recovered by calculating matrix_comp*voltage_derivative+vector_comp, i.e., e was divided into a part depending
        on the voltage derivates and an independent part. M.u_derivative + b(u) (cf drawings)
        """
        matrix_comp = -self.tau * (weights_B_transp_dr.T - np.dot(weights_PI_transp_dr.T, weights_IP_dr))
        matrix_comp += -self.tau * np.diag(rho_scnd_deriv * dendritic_error)
        matrix_comp[-self.target_size:, -self.target_size:] += np.identity(self.target_size) * self.beta * self.tau

        vector_comp = rho_deriv * dendritic_error
        vector_comp[-self.target_size:] += self.beta * (self.target - voltages[-self.target_size:]) + self.tau * self.beta * (self.target - self.old_target) / self.dt
        vector_comp += -np.dot(weights_PI_transp_dr.T, np.append(self.tau / self.dt * (self.input - self.old_input), np.zeros(self.neuron_qty - self.input_size)))

        return matrix_comp, vector_comp

    def _get_weight_derivative_terms(self, voltages, rho, rho_deriv, weights_deriv, weights_B_deriv, weights_PI_deriv, weights_IP_deriv, weights_PI_transp, weights_IP):
        """
        Terms originating from taking the derivatives of the weights. Completely in vector form (i.e., independent of the derivative of the voltages).
        """
        error_comp = self.tau * rho_deriv * (np.dot(weights_B_deriv.T, voltages) - np.dot(weights_PI_deriv.T, np.dot(weights_IP, rho)) - np.dot(weights_PI_transp.T, np.dot(weights_IP_deriv, rho)))
        input_comp = self.tau * np.dot(weights_deriv, rho)

        return error_comp, input_comp

    def _get_layer_input_terms(self, voltages, linput, rho_deriv, weights_dr, weights_PI_transp):
        """
        Terms originating from external and synaptic inputs.
        """
        matrix_comp = self.tau * (np.identity(self.neuron_qty) - weights_dr)
        vector_comp = linput - voltages
        vector_comp[:self.input_size] += self.tau / self.dt * (self.input - self.old_input)

        return matrix_comp, vector_comp

    # WEIGHT DERIVATIVES
    ######################
    def _update_weights(self, voltages, rho, l_input):
        """
        Weight update of the network weights. See Eq. p18
        """
        return np.outer(voltages - l_input, rho) * self.weight_mask * self.arg_lrate_W

    def _update_weights_B(self, voltages, weights_B_transp):
        """
        Weight update of the generative weights of the interneuron circuit. See Eq. 31
        """
        return (np.outer(voltages - np.dot(weights_B_transp.T, voltages), voltages) * self.weight_mask.T * self.arg_lrate_B).T

    def _update_weights_PI(self, voltages, weights_B_transp, weights_PI_transp, weights_IP, rho):
        """
        Weight update of the interneuron to pyramidal weights (to learn a self-predicting state). See Eq. 34
        """
        interneuron_voltages = np.dot(weights_IP, rho)
        return (np.outer(np.dot(weights_B_transp.T, voltages) - np.dot(weights_PI_transp.T, interneuron_voltages), interneuron_voltages) * self.weight_mask.T * self.arg_lrate_PI).T

    # NETWORK INTEGRATOR
    ######################
    def update_network(self, input_voltage, output_voltage, train=False, return_error=False):
        """
        Perform a single integration step.
        """
        err_, err = self.calculate_network_state(input_voltage, output_voltage, train, return_error)
        self.voltages += self.dt * self.voltages_deriv
        self.weights += self.dt * self.weights_deriv
        if self.use_interneurons:
            self.weights_B_transp += self.dt * self.weights_B_deriv
            self.weights_PI_transp += self.dt * self.weights_PI_deriv
        return err_, err

    # GET and SET
    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)

    # HELPER FUNCTIONS IN NOTEBOOKS
    def get_rates(self, voltages, voltages_deriv):
        """
        Returns rate+rate lookahead for act. function. Purely used as a utility function!
        """
        rho = self.act_function(voltages)
        rho_deriv = self.act_func_deriv(voltages)
        rho_lahead = rho + self.tau * rho_deriv * voltages_deriv
        return rho, rho_lahead
