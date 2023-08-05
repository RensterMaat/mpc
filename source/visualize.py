import numpy as np
from model_predictive_control import MPC
from double_pendulum import DoublePendulumOnCart

# initialize the pendulum in the down position
pc = DoublePendulumOnCart("down", l1=0.5, l2=0.5, m_cart=0.6, m1=0.2, m2=0.2)

# set up the MPC controller
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

# attach the controller to the pendulum
pc.controller = mpc

# run the simulation
pc.visualize()
