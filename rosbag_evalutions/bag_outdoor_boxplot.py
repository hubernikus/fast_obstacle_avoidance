""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2022-05-03
# Github: hubernikus

import os
import logging

import numpy as np
from numpy import linalg as LA

import pandas as pd

import rosbag

from vartools.states import ObjectPose
from fast_obstacle_avoidance.control_robot import QoloRobot

import matplotlib.pyplot as plt
import seaborn as sns

from .bag_utils import boxplot_of_creator


class DataHandler:
    def __init__(
        self,
        bag_names: list = None,
        bag_dir: str = "../data_qolo/marketplace_lausanne_2022_01_28/",
        import_bag: bool = True,
    ):
        self.robot = QoloRobot(
            pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
        )

        if not import_bag:
            return

        if bag_names is None:
            bag_tuples = [
                # ("2022-01-28-13-50-27.bag", "Sampled slow"),  # mostlystanding
                ("2022-01-28-13-33-32.bag", "Sampled"),
                # ("2022-01-28-13-20-25.bag", "Sampled"),  # Very short -> nonusefull
                ("2022-01-28-13-20-46.bag", "Sampled"),
                ("2022-01-28-14-00-39.bag", "Sampled"),
                ("2022-01-28-14-03-04.bag", "Sampled"),
                ("2022-01-28-14-08-47.bag", "Sampled"),
            ]
            bag_names = [name for name, _ in bag_tuples]
            self.experiment_id = [name[-9:-7] + name[-6:-4] for name, _ in bag_tuples]

            self.detection_type = [dettype for _, dettype in bag_tuples]

        self.linear_vels = []
        self.ctrl_contributions = []
        self.closest_distances = []

        self.it_bag = 0

        self.data_list = []
        for bag_name in bag_names:
            new_bag = rosbag.Bag(bag_dir + bag_name)
            self.data_list.append(self.load_data(new_bag))

        # df_columns = ["Value", "BagId", "Disparate"]
        # self.linear_vels = pd.DataFrame(df_columns)
        # self.ctrl_contributions = pd.DataFrame(df_columns)
        # self.closest_distances = pd.DataFrame(df_columns)

        # Create all data-frames
        # self.closest_distances_df = pd.concat(
        # self.closest_distances,
        # keys=[ii for ii in range(len(self.closest_distances))],
        # )

    def load_data(self, new_bag):
        # Reset values
        self._last_front = None
        self._last_back = None

        self._last_user = None
        self._last_remote = None

        return self.load_velocity(new_bag)

    def load_velocity(self, my_bag):
        linear_vel = []
        ctrl_contribution = []
        closest_distance = []
        for topic, msg, t in my_bag.read_messages(
            topics=[
                "/front_lidar/scan",
                "/rear_lidar/scan",
                "/qolo/user_commands",  # -> input
                "/qolo/remote_commands",  # -> output, i.e. final command
                "/qolo/twist",  # Actual Velocity
            ]
        ):
            # Get Distance distribution
            if topic == "/front_lidar/scan":
                self._last_front = msg

            if topic == "/rear_lidar/scan":
                self._last_back = msg

            if self._last_front is not None and self._last_back is not None:
                n_min_values = 10

                self.robot.set_laserscan(
                    self._last_front,
                    topic_name="/front_lidar/scan",
                    save_intensity=True,
                )

                self.robot.set_laserscan(
                    self._last_back, topic_name="/rear_lidar/scan", save_intensity=True
                )
                laserscan = self.robot.get_allscan(in_robot_frame=True)

                dists = LA.norm(laserscan, axis=0)
                idx = np.argpartition(dists, n_min_values)
                min_dists = dists[idx[:n_min_values]]
                closest_distance.append(np.mean(min_dists))

                # Calculate minimum distance
                self._last_back = None
                self._last_front = None

            # Get Controller-Contribution
            if topic == "/qolo/user_commands":
                self._last_user = msg

            if topic == "/qolo/remote_commands":
                self._last_remote = msg

            if self._last_user is not None and self._last_remote is not None:
                remote_xy = np.array(
                    [
                        self._last_remote.data[0],
                        self.robot.control_point[0] * np.cos(self._last_remote.data[1]),
                    ]
                )
                user_xy = np.array(
                    [
                        self._last_user.data[0],
                        self.robot.control_point[0] * np.cos(self._last_user.data[1]),
                    ]
                )
                norm_user = LA.norm(user_xy)
                if not norm_user:
                    continue

                ctrl_contribution.append(LA.norm(remote_xy - user_xy) / norm_user)

                # norm_remote = LA.norm(remote_xy)
                # if not norm_remote:
                # continue

                # remote_xy = remote_xy / norm_remote
                # user_xy = user_xy / norm_user

                # ctrl_contribution.append(LA.norm(remote_xy - user_xy))

                # remote_xy = remote_xy / LA.norm(remote_xy)
                # user_xy = user_xy / LA.norm(user_xy)

                # divisior = np.maximum(LA.norm(user_xy), 1e-1)
                # ctrl_contribution.append(LA.norm(remote_xy - user_xy) / divisior)

                self._last_user = None
                self._last_remote = None

            # Get linear_velocity
            if topic == "/qolo/twist":

                # Velocity is added only above a minimal margin
                vel_magintude = LA.norm(msg.twist.linear.x)
                # if vel_magintude > (1e-1):
                if True:
                    linear_vel.append(vel_magintude)

        # self.closest_distances.append(pd.DataFrame({"Value": closest_distance}))

        # Make sure were getting the next bag
        self.it_bag += 1
        time_range = [0, t]

        return {
            "linear_vel": linear_vel,
            "ctrl_contribution": ctrl_contribution,
            "closest_distance": closest_distance,
            "time_range": time_range,
        }


def boxplot_of_creator(
    data_handler,
    data_key="linear_vel",
    data_label="Linear velocity [m / s]",
    figure_name=None,
):
    fig, ax_sb = plt.subplots(figsize=(5, 3))

    # Control contribution
    ctrl_contributions = []

    for ii, data in enumerate(data_handler.data_list):
        new_frame = pd.DataFrame(data[data_key], columns=[data_label])

        # new_frame["method"] = data_handler.method_names[ii]
        # new_frame["Detection"] = data_handler.detection_type[ii]
        new_frame["Experiment"] = data_handler.experiment_names[ii]

        ctrl_contributions.append(new_frame)

    pd_ctrl = pd.concat(
        ctrl_contributions, keys=[ii for ii in range(len(ctrl_contributions))]
    )

    # mdf = pd.melt(pd_ctrl, id_vars=["Experiment"])
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x=pd_ctrl["Experiment"],
        y=pd_ctrl[data_label],
        # hue=pd_ctrl["Detection"],
        # linewidth=2.5,
        # showfliers=False,
        flierprops=dict(markerfacecolor="0.50", markersize=2),
    )

    if figure_name is not None:
        fig.savefig(os.path.join("figures", figure_name))

    # ax_sb.set_xlim([0.5, ax_sb.get_xlim()[1]])
    return fig, ax


def plot_normalized_over_time(
    data_handler, data_key="linear_vel", data_label="Linear velocity [m / s]"
):
    # boxplot_of_creator(data_handler, "ctrl_contribution", "Control contribution")
    # fig, ax = boxplot_of_creator(data_handler, "linear_vel", "Linear velocity [m / s]")
    # data_handler, "closest_distance", "Closest distance [m]"
    fig, axs = plt.subplots(len(data_handler.data_list), 1, figsize=(14, 12))

    for ii, data in enumerate(data_handler.data_list):
        ax = axs[ii]
        # x_vals = np.linspace(0, 1, len(data[data_key]))
        # ax.plot(x_vals, data[data_key], label=f"{ii+1}")
        ax.plot(data[data_key], label=f"{ii+1}")

    axs[3].set_ylabel(data_label)
    # ax.set_xlim([0, 1])
    axs[3].legend()

    return fig, ax


def box_plotter(main_handler, save_figure=True):
    plt.close("all")
    fig, ax = boxplot_of_creator(data_handler, "linear_vel", "Linear velocity [m / s]")
    if save_figure:
        fig.savefig("figures/linear_velocity_boxplot.pdf")

    fig, ax = boxplot_of_creator(
        data_handler, "closest_distance", "Closest distance [m]"
    )
    ax.set_ylim([0.0, ax.get_ylim()[1]])
    ax.plot(ax.get_xlim(), [0.45, 0.45], "--", linewidth=0.5)
    if save_figure:
        fig.savefig("figures/boxplot_closest_distance.pdf")

    boxplot_of_creator(data_handler, "ctrl_contribution", "Control contribution")
    if save_figure:
        fig.savefig("figures/boxplot_ctrl_contribution.pdf")


if (__name__) == "__main__":
    logging.basicConfig(
        # level=logging.DEBUG,
        # format="%(message)s",
        # format="%(asctime) %(levelname)-8s %(threadName)s %(message)s",
    )

    plt.ion()
    test_store_one_only = False
    if test_store_one_only:
        # Get single bag
        # bag_name = "2022-01-28-13-33-32.bag"
        bag_name = "2022-01-28-13-33-32.bag"

        bag_dir = "../data_qolo/marketplace_lausanne_2022_01_28/"

        import_again = False
        if import_again or not "my_bag" in locals():
            print("Loading again.")
            my_bag = rosbag.Bag(bag_dir + bag_name)

        data_handler = DataHandler(import_bag=False)
        data = data_handler.load_data(my_bag)

    import_again = False
    if import_again or not "data_handler" in locals():
        data_handler = DataHandler()

    # create_boxplots(data_handler)
    # boxplot_of_ctrl_contribution(data_handler)
    # boxplot_of_distance(data_handler)
    # boxplot_of_velocity(data_handler)

    logging.info("Execution finished.")

    plt.close("all)")
    plot_normalized_over_time(data_handler, "linear_vel", "Linear velocity [m / s]")
    plot_normalized_over_time(data_handler, "closest_distance", "Closest distance [m]")
    plot_normalized_over_time(data_handler, "ctrl_contribution", "Control contribution")
