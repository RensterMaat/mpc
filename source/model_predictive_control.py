import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import minimize
from sympy import symbols, prod, simplify, diff, lambdify


class MPC:
    """
    Implements a model predictive controller.

    This implementation uses orthogonal collocation on finite elements to
    discretize the system dynamics. This means that the optimization problem is
    reformulated: instead of optimizing only the control inputs, the optimization
    problem is to find the control inputs and the state variables at the collocation
    nodes. Constraints are used to enforce that the collocation nodes are consistent
    with the supplied equations of motion, are consistent with the initial conditions
    and are continuous between nodes. The advantage of this approach is that the
    jacobian of the objective function with respect to the input variables can be
    computed analytically, which is not possible with the standard approach of
    discretizing the system dynamics with a Runge-Kutta method. This allows the
    optimization problem to be solved much faster.

    The collocation nodes are chosen to be orthogonal collocation nodes of the
    Gauss-Legendre quadrature rule. This optimizes the accuracy of the collocation
    method.


    Attributes:
        objective_equation: The symbolic equation to be minimized.
        ode_equation: The symbolic equations of motion.
        state_variables: The symbolic state variables.
        input_variables: The symbolic input variables.
        initial_state: The numerical initial state of the system.
        n_elements: The number of elements to use for the collocation method.
        time_per_element: The time to simulate per element.
        nodes_per_element: The number of nodes to use per element.

    Methods:
        setup(): Set up the numerical objective function and ode, collocation method,
        constraints and optimization problem.
        step(): (Re)calculate the optimal control inputs and state variables.
        shift_one_time_step(): Shift the optimal control inputs and state variables
            one time step forward. This is used for the initial guess of the next
            optimization step.
        get_collocation_node_times(): Returns the times of the collocation nodes.
        get_alphas(): Returns the alphas of the collocation nodes.
        objective(): Returns the value of the objective function at every node.
        objective_jacobian(): Returns the jacobian of the objective function with
            respect to the input variables.
        ode(): Returns the derivative of the state variables at every node.
        ode_constraints(): Returns the constraints enforcing that the collocation
            nodes are consistent with the equations of motion.
        ode_constraints_jacobian(): Returns the jacobian of the ode constraints with
            respect to the input variables.
        initial_condition_constraint(): Returns the constraint enforcing that the
            initial conditions are satisfied.
        setup_initial_condition_constraint_jacobian(): Precalculates the jacobian of
            the initial condition constraint with respect to the input variables. This
            can be done as the jacobian is not dependent on the state variables.
        continuity_constraint(): Returns the constraint enforcing that the state
            variables are continuous between elements.
        setup_continuity_constraint_jacobian(): Precalculates the jacobian of the continuity
            constraint with respect to the input variables. This can be done as the jacobian
            is not dependent on the state variables.

    """

    def __init__(
        self,
        objective_equation,
        ode_equation,
        state_variables,
        input_variables,
        initial_state,
        n_elements=6,
        time_per_element=0.5,
        nodes_per_element=4,
    ):
        self.objective_equation = objective_equation
        self.ode_equation = ode_equation
        self.state_variables = state_variables
        self.input_variables = input_variables
        self.initial_state = initial_state

        self.n_elements = n_elements
        self.time_per_element = time_per_element
        self.nodes_per_element = nodes_per_element

        self.log = []

        self.setup()

    def setup(self):
        """
        Set up the numerical objective function and ode, collocation method,
        constraints and optimization problem.
        """

        # Calculate the dimensions of the state and input vector for later reshaping operations
        self.state_vector_dimension = len(self.state_variables)
        self.input_vector_dimension = len(self.input_variables)

        # Create numerical functions for the objective and corresponding jacobian
        self.f_objective = lambdify(self.state_variables, self.objective_equation)
        self.f_objective_jacobian = lambdify(
            self.state_variables,
            [diff(self.objective_equation, var) for var in self.state_variables],
        )

        # Create numerical functions for the ode and corresponding jacobian
        state_and_input_variables = np.concatenate(
            [self.state_variables, self.input_variables]
        )
        self.f_ode = lambdify(state_and_input_variables, self.ode_equation)
        self.f_ode_jacobian = lambdify(
            state_and_input_variables,
            [diff(self.ode_equation, var) for var in state_and_input_variables],
        )

        # Set the initial guess to random values
        self.previous_optimum = np.random.normal(
            size=(
                self.n_elements * self.nodes_per_element * self.state_vector_dimension
                + self.n_elements * self.input_vector_dimension
            )
        )
        self.n_elements_completed = 0

        self.collocation_node_times = self.get_collocation_node_times()
        self.alphas = self.get_alphas()

        self.setup_initial_condition_constraint_jacobian()
        self.setup_continuity_constraint_jacobian()
        self.setup_gradient_polynomial_jacobian()

        # format the constraints for the optimization problem
        self.constraints = [
            {
                "type": "eq",
                "fun": self.collocation_node_constraint,
                "jac": self.collocation_node_constraint_jacobian,
            },
            {
                "type": "eq",
                "fun": self.continuity_constraint,
                "jac": lambda _: self.continuity_constraint_jacobian,
            },
            {
                "type": "eq",
                "fun": self.initial_condition_constraint,
                "jac": lambda _: self.initial_condition_constraint_jacobian,
            },
        ]

        # perform the first optimization step to get the initial control inputs
        self.step()

    def step(self):
        """
        (Re)calculate the optimal control inputs and state variables.
        """

        # calculate the initial guess for the optimization problem using the previous optimum
        approximate_optimum = self.shift_one_time_step(self.previous_optimum)

        # perform the optimization to get the optimal control inputs
        results = minimize(
            fun=self.objective,
            jac=self.objective_jacobian,
            x0=approximate_optimum,
            constraints=self.constraints,
        )

        # store the results of the optimization
        self.previous_optimum = results.x

        # extract the optimal control inputs and state variables
        _, inputs = self.parse_x(results.x)

        # the first element of the first input is the force that is returned
        # by the controller if called
        self.current_input = inputs[0, 0]

        self.n_elements_completed += 1

        self.log.append(results)

        # return results

    def __call__(self, t, x):
        """
        Return the current control input at time t and state x.

        If t is greater than the time of the next collocation node, the optimal
        control inputs and state variables are recalculated.

        Args:
            t (float): The current time.
            x (np.array): The current state.

        Returns:
            float: The optimal control input at the current time.
        """

        # if t is greater than the time of the next collocation node, recalculate the optimal control inputs
        if t >= self.time_per_element * self.n_elements_completed:
            self.initial_state = x
            self.step()

        return self.current_input

    def shift_one_time_step(self, x):
        """
        Shift the state and input vectors one time step forward.

        Is used to calculate the initial guess for the optimization problem.

        Args:
            x (np.array): The state and input vector.

        Returns:
            np.array: The state and input vector shifted one time step forward.

        """
        state, inputs = self.parse_x(x)
        inputs = inputs[:, 0]

        shifted_state = np.empty(state.shape)
        shifted_state[:-1] = state[1:]
        shifted_state[-1] = state[-1]

        shifted_inputs = np.empty(inputs.shape)
        shifted_inputs[:-1] = inputs[1:]
        shifted_inputs[-1] = inputs[-1]

        return np.concatenate([shifted_state.flatten(), shifted_inputs.flatten()])

    def objective(self, x):
        """
        Calculate the objective function value for the given state and input vector.

        Wraps f_objective to allow for vectorized calculations.

        Args:
            x (np.array): The state and input vector.

        Returns:
            float: The objective function value."""
        state, _ = self.parse_x(x)
        return np.apply_along_axis(
            lambda x: self.f_objective(*x), -1, state[:, -1]
        ).sum()

    def objective_jacobian(self, x):
        """
        Calculate the jacobian of the objective function for the given state and input vector.

        Wraps f_objective_jacobian to allow for vectorized calculations.

        Args:
            x (np.array): The state and input vector.

        Returns:
            np.array: The jacobian of the objective function.
        """
        state, inputs = self.parse_x(x)

        state_jacobian = np.zeros(state.shape)
        state_jacobian[:, -1] = np.apply_along_axis(
            lambda x: self.f_objective_jacobian(*x), -1, state[:, -1]
        )

        jacobian = np.concatenate(
            [state_jacobian.flatten(), np.zeros(inputs[:, 0].shape).flatten()]
        )
        return jacobian

    def ode(self, x):
        """
        Calculate the derivative of the state vector for the given state and input vector.

        Wraps f_ode to allow for vectorized calculations.

        Args:
            x (np.array): The state and input vector.

        Returns:
            np.array: The derivative of the state vector.
        """
        state, inputs = self.parse_x(x)
        state_and_inputs = np.concatenate([state, inputs], axis=-1)
        return np.apply_along_axis(
            lambda x: self.f_ode(*x).flatten(), -1, state_and_inputs
        )

    def get_collocation_node_times(self):
        """
        Calculate the collocation node times for the given number of collocation nodes.

        Collocation node times are chosen according to the roots of the Legendre to
        minimize the error of the collocation method.

        Returns:
            np.array: The collocation node times.
        """
        roots, _ = roots_legendre(self.nodes_per_element - 2)
        collocation_node_times = np.array([0, *(roots / 2 + 0.5), 1])

        return collocation_node_times

    def get_alphas(self):
        """
        Calculate the alphas for the collocation method.

        The alphas are the coefficients of the Lagrange interpolation polynomials
        that are used to interpolate the derivative of the state variables between
        the collocation nodes. These derivatives should be equal to the derivative
        calculated by the ODE function.

        These values are only dependent on the collocation node times and
        are therefore calculated once and stored in the class.
        """

        # tau is the time variable used for the interpolation polynomials and lies
        # between 0 and 1
        tau = symbols("tau")
        self.nodes_per_element = len(self.collocation_node_times)

        # calculate the alphas for the collocation method using the formula for
        # Lagrange interpolation polynomials
        alphas = np.empty((self.nodes_per_element, self.nodes_per_element))
        for j in range(self.nodes_per_element):
            # calculate the lagrange interpolation polynomial for the j-th collocation
            # node
            lagrange_interpolation_term = simplify(
                prod(
                    [
                        (tau - self.collocation_node_times[i])
                        / (
                            self.collocation_node_times[j]
                            - self.collocation_node_times[i]
                        )
                        for i in range(len(self.collocation_node_times))
                        if not i == j
                    ]
                )
            )

            # calculate the derivative of the lagrange interpolation polynomial
            lagrange_interpolation_term_dt = diff(lagrange_interpolation_term, tau)
            for k in range(self.nodes_per_element):
                alphas[j, k] = lagrange_interpolation_term_dt.subs(
                    tau, self.collocation_node_times[k]
                )

        return alphas

    def parse_x(self, x):
        """
        Parse the state and input vector from the combined state and input vector.

        The combined state and input vector has the shape (n_elements * nodes_per_element
        * state_vector_dimension + n_elements * input_vector_dimension). This vector is
        reshaped into the state vector and input vector. The state vector has shape
        (n_elements, nodes_per_element, state_vector_dimension) and the input vector has
        shape (n_elements, 1, input_vector_dimension). The input vector is then repeated
        nodes_per_element times along the second axis to match the shape of the state
        vector.

        Args:
            x (np.array): The flat, combined state and input vector.

        Returns:
            tuple: The state and input vector.
        """
        state_elements = (
            self.n_elements * self.nodes_per_element * self.state_vector_dimension
        )
        state, inputs = x[:state_elements], x[state_elements:]

        state = state.reshape(
            (self.n_elements, self.nodes_per_element, self.state_vector_dimension)
        )

        inputs = inputs.reshape((self.n_elements, 1, self.input_vector_dimension))
        inputs = inputs.repeat(self.nodes_per_element, 1)

        return state, inputs

    def collocation_node_constraint(self, x):
        """
        Calculate the collocation node constraints for the given state and input vector.

        The collocation node constraints are the constraints that the derivative of the
        state vector should obey the differential equation at the collocation nodes.
        These derivatives can be calculated using the precalculated alphas.

        SciPy requires equality constraints to return the value of the function that
        must equal zero. Therefore, the difference between the derivative calculated
        by the ODE function and the derivative calculated by the Lagrange interpolation
        polynomials is returned.

        Args:
            x (np.array): The state and input vector.

        Returns:
            np.array: The difference between the derivative calculated by the ODE
                function and the derivative calculated by the Lagrange interpolation
                polynomials.
        """
        state, _ = self.parse_x(x)
        gradient_polynomial = (
            np.einsum("jk...,ij...->ik...", self.alphas[:, 1:], state)
            / self.time_per_element
        )
        gradient_ode = self.ode(x)[:, 1:]

        return (gradient_polynomial - gradient_ode).flatten()

    def collocation_node_constraint_jacobian(self, x):
        """
        Calculate the Jacobian of the collocation node constraints for the given state and input vector.

        There are n_elements * nodes_per_element * state_vector_dimension collocation node constraints.
        There are n_elements * nodes_per_element * state_vector_dimension + n_elements * input_vector_dimension
        state and input variables. Therefore, the Jacobian has shape (n_elements * nodes_per_element * state_vector_dimension,
        n_elements * nodes_per_element * state_vector_dimension + n_elements * input_vector_dimension).

        Most elements in this Jacobian are zero. Non-zero elements only occur for "diagonal" elements, i.e. elements
        where the state variable is differentiated with respect to itself.

        The Jacobian is formed by the difference in Jacobian of the Lagrange interpolation polynomials and the Jacobian
        of the ODE function. The Jacobian of the Lagrange interpolation polynomials is calculated using the precalculated
        alphas. The Jacobian of the ODE function is symbolically derived in the setup method.

        Args:
            x (np.array): The state and input vector.

        Returns:
            np.array: The Jacobian of the collocation node constraints.
        """
        state, inputs = self.parse_x(x)
        state_and_inputs = np.concatenate([state, inputs], axis=-1)

        # calculate the value of the ode jacobian

        # only values where elements are differentiated with respect to themselves
        # are non-zero
        A = np.identity(self.n_elements)

        # add two axes for number of nodes per elements
        A = (
            A[:, :, np.newaxis, np.newaxis]
            .repeat(self.nodes_per_element, 2)
            .repeat(self.nodes_per_element, 3)
        )

        # only values where nodes are differentiated with respect to themselves
        # are non-zero
        A = A * np.identity(self.nodes_per_element)
        A = A[:, :, 1:]

        # add two axes for the state vector dimension
        A = (
            A[:, :, :, :, np.newaxis, np.newaxis]
            .repeat(self.state_vector_dimension, 4)
            .repeat(self.state_vector_dimension, 5)
        )
        A = A.transpose((0, 2, 1, 3, 4, 5))

        # calculate the jacobian of the ode function
        jacobian = np.apply_along_axis(
            lambda x: self.f_ode_jacobian(*x), -1, state_and_inputs
        )

        # mask the jacobian to only include have non-zero values for at positions
        # where the element/node/state is differentiated with respect to itself
        state_jacobian = A * jacobian[:, :, : -self.input_vector_dimension, :, 0]
        input_jacobian = (
            A[:, :, :, :, :1].repeat(self.input_vector_dimension, -1)
            * jacobian[:, :, -self.input_vector_dimension :, :, 0]
        )

        # reshape to match the shape of the output jacobian
        state_jacobian = state_jacobian.transpose(0, 1, 5, 2, 3, 4).reshape(
            self.n_elements
            * (self.nodes_per_element - 1)
            * self.state_vector_dimension,
            self.n_elements * self.nodes_per_element * self.state_vector_dimension,
        )

        input_jacobian = (
            input_jacobian.transpose(0, 1, 5, 2, 3, 4)
            .sum(axis=4)
            .reshape(
                self.n_elements
                * (self.nodes_per_element - 1)
                * self.state_vector_dimension,
                self.n_elements * self.input_vector_dimension,
            )
        )

        gradient_ode_jacobian = np.concatenate([state_jacobian, input_jacobian], axis=1)

        # the gradient of the Lagrange interpolation polynomials is constant
        return self.gradient_polynomial_jacobian - gradient_ode_jacobian

    def setup_gradient_polynomial_jacobian(self):
        """
        Calculate the Jacobian of the Lagrange interpolation polynomials.

        As the values of this Jacobian is only dependent on the alphas, it is calculated
        once in the setup method and stored for later use.

        There are n_elements * nodes_per_element * state_vector_dimension collocation node constraints.
        There are n_elements * nodes_per_element * state_vector_dimension + n_elements * input_vector_dimension
        state and input variables. Therefore, the Jacobian has shape (n_elements * nodes_per_element * state_vector_dimension,
        n_elements * nodes_per_element * state_vector_dimension + n_elements * input_vector_dimension).

        Most elements in this Jacobian are zero. Non-zero elements only occur for "diagonal" elements, i.e. elements
        where the state variable is differentiated with respect to itself.

        Returns:
            np.array: The Jacobian of the Lagrange interpolation polynomials.
        """

        # initialize the jacobian of the Lagrange interpolation polynomials
        jacobian = np.zeros(
            (
                self.n_elements,
                self.nodes_per_element - 1,
                self.state_vector_dimension,
                self.n_elements,
                self.nodes_per_element,
                self.state_vector_dimension,
            )
        )

        # the values of the Jacobian are set to the corresponding alpha values
        for i in range(self.n_elements):
            for j in range(self.nodes_per_element - 1):
                for k in range(self.state_vector_dimension):
                    jacobian[i, j, k, i, :, k] = (
                        self.alphas[:, 1:][:, j] / self.time_per_element
                    )

        # reshape to match the shape of the output jacobian
        jacobian = jacobian.reshape(
            self.n_elements
            * (self.nodes_per_element - 1)
            * self.state_vector_dimension,
            self.n_elements * self.nodes_per_element * self.state_vector_dimension,
        )

        # the values for the input variables are zero
        self.gradient_polynomial_jacobian = np.concatenate(
            [
                jacobian,
                np.zeros(
                    (jacobian.shape[0], self.n_elements * self.input_vector_dimension)
                ),
            ],
            axis=1,
        )

    def continuity_constraint(self, x):
        """
        Calculate the continuity constraint.

        The continuity constraint is the difference between the state at the end of one element
        and the state at the beginning of the next element. This should be zero for all elements.

        Args:
            x (np.array): The state and input variables.

        Returns:
            np.array: The difference in state between the end of one element and the beginning of the next element.
        """
        state, _ = self.parse_x(x)
        return (state[1:, 0] - state[:-1, -1]).flatten()

    def setup_continuity_constraint_jacobian(self):
        """
        Calculate the Jacobian of the continuity constraint.

        As the values of this Jacobian is only dependent on the alphas, it is calculated
        once in the setup method and stored for later use.

        There are n_elements - 1 continuity constraints. There are n_elements * nodes_per_element * state_vector_dimension
        state variables. Therefore, the Jacobian has shape (n_elements - 1, n_elements * nodes_per_element * state_vector_dimension).

        Elements of this Jacobian are either 0, 1 or -1. Almost all elements are zero. Values of 1 occur for
        elements where the state variable is differentiated with respect to itself and the element is the first
        element in the sequence. Values of -1 occur for elements where the state variable is differentiated with
        respect to itself and the element is the last element in the sequence.

        Returns:
            np.array: The Jacobian of the continuity constraint.
        """

        # initialize a negative diagonal matrix for the Jacobian of the negative part of the continuity
        # constraint, namely the subtraction of the state at the end of one element from the state at
        # the beginning of the next element
        A = -np.identity(self.n_elements)[:-1]
        A = (
            A[:, :, np.newaxis, np.newaxis]
            .repeat(self.nodes_per_element, 2)
            .repeat(self.state_vector_dimension, 3)
        )

        # all values are zero except for the last element in each row
        k = np.zeros((self.nodes_per_element, self.state_vector_dimension))
        k[-1] = 1

        # broadcast the values of k to the jacobian matrix
        A = A * k

        # initialize a positive diagonal matrix for the Jacobian of the positive part of the continuity
        # constraint, namely the state of the first node in the sequence of the second to last element
        B = np.identity(self.n_elements - 1)
        B = (
            B[:, :, np.newaxis, np.newaxis]
            .repeat(self.nodes_per_element, 2)
            .repeat(self.state_vector_dimension, 3)
        )

        # all values are zero except for the first element in each row
        k = np.zeros((self.nodes_per_element, self.state_vector_dimension))
        k[0] = 1

        B = B * k

        # add the matrices for the positive and negative part
        A[:, 1:] = A[:, 1:] + B

        # the derivatives are the same for every state variable
        A = A.repeat(self.state_vector_dimension, 0)

        # add a dimension for the state vector dimension and set all off-diagonal elements to zero
        i = np.identity(self.state_vector_dimension)
        C = np.stack([np.stack([i] * self.nodes_per_element)] * self.n_elements)
        C[0, 0] = 0
        C[:, 1:-1] = 0
        C[-1, -1] = 0

        C = np.concatenate([C.transpose((3, 0, 1, 2))] * (self.n_elements - 1))

        # reshape to match the shape of the output jacobian
        state_jacobian = (A * C).reshape(
            ((self.n_elements - 1) * self.state_vector_dimension), -1
        )

        # add zeros for the input variables
        inputs_jacobian = np.zeros((state_jacobian.shape[0], self.n_elements))
        self.continuity_constraint_jacobian = np.concatenate(
            [state_jacobian, inputs_jacobian], axis=1
        )

    def initial_condition_constraint(self, x):
        """
        Calculate the initial condition constraint.

        The initial condition constraint is the difference between the initial state and the state at the
        beginning of the first element. This should be zero.

        Args:
            x (np.array): The state and input variables.

        Returns:
            np.array: The difference between the initial state and the state at the beginning of the first element.

        """
        state, _ = self.parse_x(x)
        return state[0, 0] - self.initial_state

    def setup_initial_condition_constraint_jacobian(self):
        """
        Calculate the Jacobian of the initial condition constraint.

        There is one constraint for each state variable. There are n_elements * nodes_per_element * state_vector_dimension
        state variables. Therefore, the Jacobian has shape (state_vector_dimension, n_elements * nodes_per_element * state_vector_dimension).

        Elements of this Jacobian are either 0 or 1. Almost all elements are zero. Values of 1 occur for
        the state variables in the first node of the first element, where the state variable is differentiated with
        respect to itself.
        """

        # initialize a zero matrix for the jacobian
        self.initial_condition_constraint_jacobian = np.zeros(
            (self.state_vector_dimension, len(self.previous_optimum))
        )

        # set the diagonal elements of the first node of the first element to 1
        self.initial_condition_constraint_jacobian[
            : self.state_vector_dimension, : self.state_vector_dimension
        ] = np.identity(self.state_vector_dimension)
