import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from time import time
from sympy import symbols, diff, cos, sin, Matrix, simplify, solve, lambdify
from sympy.physics.mechanics import dynamicsymbols
from abc import ABC


class System(ABC):
    """
    Abstract base class for all physical simulation systems.

    Attributes:
        y (list | np.array): The state vector of the system.
        t (float): The current time of the system.

    Methods:
        step(dt): Advances the system by dt seconds.

    """

    def step(self, dt):
        self.y = solve_ivp(self.dy_dt, [0, dt], self.y).y[:, -1]
        self.t += dt


class DoublePendulumOnCart(System):
    """
    System simulating a double pendulum on a cart.

    Attributes:
        y (list | np.array): The state vector of the system.
        t (float): The current time of the system.
        params (tuple): The parameters of the system.
        controller (Controller): The controller of the system.
        equations_of_motion (list): List of sympy expressions representing the equations of motion of the system.
        state_vector (list): Vector of sympy symbols representing the state of the system.
        force (sympy symbol): Symbol representing the force applied to the cart.

    Methods:
        derive_equations_of_motion(): Derives the equations of motion of the system.
        dy_dt(t, y): Returns the derivative of the state vector at time t to be used for integration.
        pendulum_plot_coordinates(): Returns the coordinates of the pendulum for plotting.
        cart_plot_coordinates(): Returns the coordinates of the cart for plotting.
        visualize(fps, duration): Visualizes the system for duration seconds at fps frames per second.
        init_plot_(): Initializes the plot.
        animate_(i): Animates the plot at frame i.
    """

    def __init__(self, y0, controller=None, m_cart=5, m1=1, m2=1, l1=1, l2=1, g=9.8):
        """
        Args:
            y0 (list | np.array): The initial state vector of the system.
            controller (Controller): The controller of the system.
            m_cart (float): The mass of the cart.
            m1 (float): The mass of the first pendulum.
            m2 (float): The mass of the second pendulum.
            l1 (float): The length of the first pendulum.
            l2 (float): The length of the second pendulum.
            g (float): The gravitational acceleration.
        """
        if y0 == "up":
            self.y = [0, 0, np.pi, 0, np.pi, 0]
        elif y0 == "down":
            self.y = [0, 0, 0, 0, 0, 0]
        else:
            self.y = y0
        self.params = (m_cart, m1, m2, l1, l2, g)
        self.controller = controller

        self.derive_equations_of_motion()

        self.t = 0

    def derive_equations_of_motion(self):
        """
        Derives the equations of motion of the system.

        The positions of the cart and both pendulums, along with their first and second derivatives, are
        represented as SymPy symbols. The Lagrangian and Hamiltonian of the system are derived symbolically and
        then lambdified for numerical evaluation. The equations of motion are derived symbolically and then
        simplified before being converted to a matrix form for numerical evaluation.
        """
        m_cart, m1, m2, l1, l2, g = self.params
        t = symbols("t")

        # setup symbolic state variables
        x_cart, theta1, theta2, force = dynamicsymbols("x_cart theta_1 theta_2 f")
        x_cartd, theta1d, theta2d = dynamicsymbols("x_cart theta_1 theta_2", 1)
        x_cartdd, theta1dd, theta2dd = dynamicsymbols("x_cart theta_1 theta_2", 2)

        self.state_vector = [x_cart, x_cartd, theta1, theta1d, theta2, theta2d]
        self.force = force

        # derive symbolic expressions for kinetic energy and potential energy
        K_cart = 0.5 * m_cart * diff(x_cart, t) ** 2

        x1 = x_cart + l1 * sin(theta1)
        y1 = -l1 * cos(theta1)

        K1 = 0.5 * m1 * (diff(x1, t) ** 2 + diff(y1, t) ** 2)
        U1 = m1 * g * y1

        x2 = x1 + l2 * sin(theta2)
        y2 = y1 - l2 * cos(theta2)

        K2 = 0.5 * m2 * (diff(x2, t) ** 2 + diff(y2, t) ** 2)
        U2 = m1 * g * y2

        # derive expressions for the lagrangian and hamiltonian
        self.L = K_cart + K1 + K2 - U1 - U2
        self.H = K_cart + K1 + K2 + U1 + U2

        # lambdify the lagrangian and hamiltonian for numerical evaluation
        self.lagrangian = lambdify(self.state_vector + [force], self.L)
        self.hamiltonian = lambdify(self.state_vector + [force], self.H)

        # derive the equations of motion
        LE1 = diff(self.L, x_cart) - diff(diff(self.L, x_cartd), t) - force
        LE2 = diff(self.L, theta1) - diff(diff(self.L, theta1d), t)
        LE3 = diff(self.L, theta2) - diff(diff(self.L, theta2d), t)

        solutions = simplify(solve([LE1, LE2, LE3], [x_cartdd, theta1dd, theta2dd]))

        self.equations_of_motion = Matrix(
            [
                x_cartd,
                simplify(solutions[x_cartdd]),
                theta1d,
                simplify(solutions[theta1dd]),
                theta2d,
                simplify(solutions[theta2dd]),
            ]
        )

        # lambdify the equations of motion for numerical evaluation
        self.dy_dt_f = lambdify(self.state_vector + [force], self.equations_of_motion)

    def dy_dt(self, t, y):
        """
        Returns the derivative of the state vector y at time t to be used for integration.

        If no controller is specified, the force is assumed to be zero.

        Args:
            t (float): The time.
            y (list | np.array): The state vector.

        Returns:
            np.array: The derivative of the state vector.
        """
        if self.controller:
            force = self.controller(self.t, y).tolist()
        else:
            force = [0]

        state_with_force = y.flatten().tolist() + force

        return self.dy_dt_f(*state_with_force).flatten()
        # x_cart, x_cartd, theta1,theta1d, theta2, theta2d = y

        # return np.array([x_cartd, self.x_cartdd(*y), theta1d, self.theta1dd(*y), theta2d, self.theta2dd(*y)])

    def pendulum_plot_coordinates(self):
        """
        Returns the x and y coordinates of the pendulums for plotting.

        Returns:
            list: The x coordinates of the pendulums.
            list: The y coordinates of the pendulums.
        """
        m_cart, m1, m2, l1, l2, g = self.params
        x_cart, x_cartd, theta1, theta1d, theta2, theta2d = self.y

        x1 = x_cart + l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        return [x_cart, x1, x2], [0, y1, y2]

    def cart_plot_coordinates(self):
        """
        Returns the x and y coordinates of the cart for plotting.

        Returns:
            list: The x coordinates of the top left corner of the cart.
        """
        x_cart, _, _, _, _, _ = self.y
        return x_cart - 1, 0

    def visualize(self, fps=30, time_span=10):
        """
        Visualizes the double pendulum and saves the animation.

        Args:
            fps (int, optional): The frames per second. Defaults to 30.
            time_span (int, optional): The time span of the animation in seconds. Defaults to 10.

        Returns:
            None
        """
        self.dt = 1 / fps

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.grid("off")
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        ax.set_box_aspect(1)

        self.cart = Rectangle((0, 0), 2, -1, color="b")
        ax.add_patch(self.cart)
        (self.pendulum_line,) = ax.plot(
            [], [], "o-", lw=2, label="Time: \nHamiltonian: ", c="black"
        )

        t0 = time()
        self.animate_(0)
        t1 = time()
        interval = self.dt * 1000 - (t1 - t0)

        self.anim = animation.FuncAnimation(
            fig,
            self.animate_,
            frames=time_span * fps,
            interval=interval,
            init_func=self.init_plot_,
        )
        self.anim.save(r"animations\\double_pendulum_on_cart.mp4", fps=fps)

    def init_plot_(self):
        """
        Returns the artists to be re-drawn at each frame.

        Used internally by matplotlib.animation.FuncAnimation.

        Returns:
            tuple: The pendulum line and the cart."""
        return self.pendulum_line, self.cart

    def animate_(self, i):
        """
        Updates the artists to be re-drawn at each frame.

        Used internally by matplotlib.animation.FuncAnimation.

        Args:
            i (int): The frame number.

        Returns:
            tuple: The pendulum line and the cart.
        """
        self.step(self.dt)

        self.pendulum_line.set_data(*self.pendulum_plot_coordinates())
        self.cart.set_xy(self.cart_plot_coordinates())

        return (self.pendulum_line,)
