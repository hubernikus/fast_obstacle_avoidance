""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2022-05-03
# Github: hubernikus

import numpy as np
from numpy import linalg as LA

import pandas as pd

import rosbag

from vartools.states import ObjectPose
from fast_obstacle_avoidance.control_robot import QoloRobot

import matplotlib.pyplot as plt
import seaborn as sns


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
            bag_names = ["2022-01-28-13-33-32.bag"]
            self.method_names = ["Disparate"]

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
                n_min_values = 5
                # breakpoint()

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
                ctrl_contribution.append(LA.norm(remote_xy - user_xy))

                # remote_xy = remote_xy / LA.norm(remote_xy)
                # user_xy = user_xy / LA.norm(user_xy)

                # divisior = np.maximum(LA.norm(user_xy), 1e-1)
                # ctrl_contribution.append(LA.norm(remote_xy - user_xy) / divisior)

                self._last_user = None
                self._last_remote = None

            # Get linear_velocity
            if topic == "/qolo/twist":
                linear_vel.append(LA.norm(msg.twist.linear.x))

        # self.closest_distances.append(pd.DataFrame({"Value": closest_distance}))

        # Make sure were getting the next bag
        self.it_bag += 1

        return {
            "linear_vel": linear_vel,
            "ctrl_contribution": ctrl_contribution,
            "closest_distance": closest_distance,
        }


def create_boxplots(data_handler):
    fig, ax = plt.subplots(figsize=(5, 4))
    for data in data_handler.data_list:
        ax.boxplot(data["ctrl_contribution"])
    ax.set_ylabel("Control contribution")

    fig, ax = plt.subplots(figsize=(5, 4))
    for data in data_handler.data_list:
        ax.boxplot(data["linear_vel"])
    ax.set_ylabel("Linear velocity")

    fig, ax = plt.subplots(figsize=(5, 4))
    for data in data_handler.data_list:
        bplot = ax.boxplot(data["closest_distance"])
    ax.set_ylabel("Closest distance")

    colors = ["pink", "lightblue", "lightgreen"]
    # for bplot in (bplot1, bplot2):
    # for patch, color in zip(bplot["boxes"], colors):
    # patch.set_facecolor(color)


def box_plot_with_sns(data_handler):
    # Control contribution
    ctrl_contributions = []

    for ii, data in enumerate(data_handler.data_list):
        new_frame = pd.DataFrame(data["ctrl_contribution"])
        # new_frame["method"] = data_handler.method_names[ii]
        new_frame["method"] = "disparate"

        ctrl_contributions.append(new_frame)

    pd_ctrl = pd.concat(
        ctrl_contributions,
        keys=[ii for ii in range(len(ctrl_contributions))],
        names=["series_id", "it_count"],
    )

    mdf = pd.melt(pd_ctrl, id_vars=["method", "values"], var_name=["series_id"])
    # mdf = pd.melt(pd_ctrl, id_vars=["Trial"], var_name=["Number"])
    # breakpoint()

    # ax = sns.boxplot(x="keys", y="")

    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")
    ax = sns.boxplot(
        x="Trial",
        y="Number",
        data=mdf,
        # hue="method",
        linewidth=2.5,
    )

    # tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
    # ax = sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False)
    # breakpoint()


if (__name__) == "__main__":
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
    box_plot_with_sns(data_handler)

    print("Done.")
