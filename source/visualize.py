import numpy as np
import matplotlib.pyplot as plt
from mpc import MPC
from single_pendulum_on_cart import PendulumOnCart
from double_pendulum_on_cart import DoublePendulumOnCart

pc = DoublePendulumOnCart("down", l1=0.5, l2=0.5, m_cart=0.6, m1=0.2, m2=0.2)

mpc = MPC(
    objective_equation=pc.L,
    ode_equation=pc.equations_of_motion,
    state_variables=pc.state_vector,
    input_variables=[pc.force],
    initial_state=np.array(pc.y),
    time_per_element=0.1,
    nodes_per_element=3,
    n_elements=40,
)

pc.controller = mpc
pc.visualize()
