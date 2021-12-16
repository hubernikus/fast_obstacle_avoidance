"""
Script to evaluate the rosbag.
"""
# Author: Lukas Huber
# Created: 2021-21-14
# 

import sys 
import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd
import rosbag

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from fast_obstacle_avoidance.control_robot import ControlRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.utils import laserscan_to_numpy


class LaserScanAnimator(Animator):
    def setup(self, static_laserscan, initial_dynamics, robot, x_lim=[-3, 4], y_lim=[-3, 3]):
        self.robot = robot
        self.initial_dynamics = initial_dynamics
        self.fast_avoider = FastObstacleAvoider(robot=self.robot)
        self.static_laserscan = static_laserscan

        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        self.obstacle_color = np.array([177, 124, 124]) / 255.0

        self.x_lim = x_lim
        self.y_lim = y_lim
        
        dimension = 2
        self.position_list = np.zeros((dimension, self.it_max+1))

    def update_step(self, ii):
        """ Update robot and position."""
        initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        start = timer()
        self.fast_avoider.update_laserscan(self.static_laserscan)
        modulated_velocity = self.fast_avoider.evaluate(initial_velocity)
        end = timer()
        print("Time for modulation {}ms at it={}".format( np.round((end-start)*1000, 3), ii))
        
        # Update qolo
        self.robot.pose.position = self.robot.pose.position + self.dt_simulation*modulated_velocity
        # self.robot.pose.orientation += self.dt_simulation*modulated_velocity
        self.position_list[:, ii] = self.robot.pose.position

        self.ax.clear()
        # Plot
        self.ax.plot(self.robot.pose.position[0],
                     self.robot.pose.position[1], 'o', color='k')
        self.ax.plot(self.static_laserscan[0, :], self.static_laserscan[1, :], '.',
                     color=self.obstacle_color, zorder=-1)
        
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_aspect('equal')

        self.ax.arrow(self.robot.pose.position[0], self.robot.pose.position[1],
                      initial_velocity[0], initial_velocity[1], 
                      width=0.05, head_width=0.3,
                      color='g',
                      label="Initial")
    
        self.ax.arrow(self.robot.pose.position[0], self.robot.pose.position[1],
                      modulated_velocity[0], modulated_velocity[1],
                      width=0.05, head_width=0.3,
                      color='b',
                      label="Modulated")

        self.robot.plot2D(self.ax)
        self.ax.plot(self.initial_dynamics.attractor_position[0],
                     self.initial_dynamics.attractor_position[1], '*', color='black') 
        # print(tt)
        self.ax.grid()
        
    def has_converged(self):
        conv_margin = 1e-4
        if (self.position_list[:, ii]-self.position_list[:, ii-1]) < conv_margin:
            return True
        else:
            return False



def get_topics(rosbag_name):
    import bagpy
    # get the list of topics
    print(bagpy.bagreader(rosbag_name).topic_table)
    

def static_plot(allscan, qolo, dynamical_system, fast_avoider):
    initial_velocity = dynamical_system.evaluate(qolo.pose.position)
    modulated_velocity = fast_avoider.evaluate(initial_velocity, allscan)    

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(0, 0, 'o', color='k')
    # ax.plot(frontscan[0, :], frontscan[1, :], '.', color='r')
    # ax.plot(rearscan[0, :], rearscan[1, :], '.', color='b')
    ax.plot(allscan[0, :], allscan[1, :], '.', color='k')

    ax.set_xlim([-10, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    ax.arrow(qolo.pose.position[0], qolo.pose.position[1],
             initial_velocity[0], initial_velocity[1], 
             width=0.05, head_width=0.3,
             color='g',
             label="Initial")
    
    ax.arrow(qolo.pose.position[0], qolo.pose.position[1],
             modulated_velocity[0], modulated_velocity[1],
             width=0.05, head_width=0.3,
             color='b',
             label="Modulated")

    ax.legend()

    qolo.plot2D(ax)
    # print(tt)
    ax.grid()
    # for topic, msg, t in my_bag.read_messages(topics=[
         # '/front_lidar/scan',
         # '/rear_lidar/scan'
    # ]):
        # breakpoint()
    

def main():
    bag_dir = '/home/lukas/Code/data_qolo/'
    # bag_name = '2021-12-13-18-32-13.bag'
    # bag_name = '2021-12-13-18-32-42.bag'
    bag_name = '2021-12-13-18-33-06.bag'
    # bag_name = '2021-12-13-18-32-13.bag'
    rosbag_name = bag_dir + bag_name
    
    # my_bag = bagpy.bagreader(rosbag)
    # my_bag = bagpy.bagreader(rosbag_name)
    my_bag = rosbag.Bag(rosbag_name)

    frontscan = None
    rearscan = None
    
    # for tt in my_bag.topics:
    for topic, msg, t in my_bag.read_messages(topics=[
        '/front_lidar/scan',
        '/rear_lidar/scan'
        ]):
        if topic == '/front_lidar/scan':
            frontscan = laserscan_to_numpy(msg)
            
        elif topic == '/rear_lidar/scan':
            rearscan = laserscan_to_numpy(
                msg, delta_angle=np.pi, delta_position=[0.75, 0.0])
            
        if frontscan is not None and rearscan is not None:
            break

    allscan = np.hstack((rearscan, frontscan))

    # Downsample for DEBUGGING only
    # sample_freq = 20
    # allscan = allscan[:,  np.logical_not(np.mod(np.arange(allscan.shape[1]), sample_freq))]

    qolo = ControlRobot(
        control_points=np.array([[0, 0],
                                 # [0.5, 0],
                                 ]).T,
        
        control_radiuses=np.array([0.5,
                                   # 0.4,
                                   ]),
        pose = ObjectPose(position=[0.7, -0.7], orientation=30*np.pi/180)
    )
    
    fast_avoider = FastObstacleAvoider(robot=qolo)

    dynamical_system = LinearSystem(
        attractor_position=np.array([-2, 2]), maximum_velocity=0.8)

    # static_plot(allscan, qolo, dynamical_system, fast_avoider)
    # breakpoint()

    main_animator = LaserScanAnimator(it_max=160, dt_simulation=0.04)
    main_animator.setup(
        static_laserscan=allscan,
        initial_dynamics=dynamical_system,
        robot=qolo,
        )

    main_animator.run(save_animation=True)
    # main_animator.update_step(ii=0)

if (__name__) == "__main__":
    plt.close('all')
    plt.ion()

    main()
    pass
