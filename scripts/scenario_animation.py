from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose 
from vartools.dynamical_systems import ConstantValue
from vartools.dynamical_systems import LinearSystem

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot
from vartools.animator import Animator

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles


class LaserscanAnimator(Animator):
    def setup(self, robot, initial_dynamics, avoider, environment, x_lim=[-10, 10], y_lim=[-10, 10]):
        self.dimension = 2
        
        self.robot = robot

        self.initial_dynamics = initial_dynamics
        self.avoider = avoider
        self.environment = environment
        
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.positions = np.zeros((self.dimension, self.it_max))

        self.velocities_init = np.zeros((self.dimension, self.it_max))
        self.velocities_mod = np.zeros((self.dimension, self.it_max))

        # Create
        self.fig, self.ax = plt.subplots(figsize=(16, 10))

        self.velocity_command = np.zeros(self.dimension)


    def update_step(self, ii):
        self.positions[:, ii] = self.robot.pose.position

        data_points = self.environment.get_surface_points(
            center_position=self.robot.pose.position,
            null_direction=self.velocity_command,
        )
        
        self.avoider.update_reference_direction(data_points, in_robot_frame=False)

        # Store all
        velocity_init = self.initial_dynamics.evaluate(self.robot.pose.position)
        self.velocity_command = self.avoider.avoid(velocity_init)

        # Update step
        self.robot.pose.position = (
            self.robot.pose.position + self.velocity_command*self.dt_simulation
        )
        
        self.ax.clear()
        
        self.ax.plot(data_points[0, :], data_points[1, :], 'o', color='k')
        
        self.ax.plot(self.robot.pose.position[0], self.robot.pose.position[1], 'o', color='b')
        
        visualize_obstacles(self.environment, ax=self.ax)

        self.ax.plot(self.positions[0, :ii], self.positions[1, :ii], '--', color='b')
        
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        
def single_polygon_animator():
    qolo = QoloRobot(
        pose=ObjectPose(position=[0.0, -3.4], orientation=0)
    )

    # dynamical_system = ConstantValue(velocity=[0, 1])
    dynamical_system = LinearSystem(
        attractor_position=np.array([0, 3]), maximum_velocity=1.0)

    fast_avoider = SampledAvoider(
        robot=qolo,
        # evaluate_normal=False,
        evaluate_normal=True,
        weight_max_norm=1e4,
        weight_factor=2,
        weight_power=1.,
    )
    
    main_environment = ShapelySamplingContainer(n_samples=50)
    
    main_environment.add_obstacle(                 
        shapely.geometry.box(-5, -1, 2, 1)
        )

    my_animator = LaserscanAnimator(
        it_max=400,
        dt_simulation=0.1,
    )

    my_animator.setup(
        robot=qolo,
        initial_dynamics=dynamical_system,
        avoider=fast_avoider,
        environment=main_environment,
        x_lim=[-4, 4],
        y_lim=[-4, 4],
    )

    my_animator.run()

if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    single_polygon_animator()

        

    
