""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2022-05-03
# Github: hubernikus

import os
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import pandas as pd

# import rosbag_pandas
# from rosbags.dataframe import get_dataframe
# from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

import rosbag

from vartools.states import ObjectPose
from fast_obstacle_avoidance.control_robot import QoloRobot

import seaborn as sns

# Set logger
# logger = logging.getLogger("personal")
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)

## Functions
def print_topics(bagname=None, bag_dir=None):
    if bag_dir is None:
        bag_dir = "../../data_qolo/marketplace_lausanne_2022_01_28"
    if bagname is None:
        bagname = os.path.join(bag_dir, "2022-01-28-13-20-46.bag")

    logging.info("Printing topics")

    # Print datatypes
    with Reader(bagname) as reader:
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)


# @dataclass
# class Recording:
# position_x: np.ndarray
# position_y: np.ndarray
# orientation: np.ndarray

# velocity_x: np.ndarray
# velocity_y: np.ndarray
# velocity_angular: np.ndarray


def get_bagvalues(bagname):
    robot = QoloRobot(
        pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
    )
    # Asign values
    pose_x = []
    pose_y = []
    pose_theta = []

    vel_x = []
    vel_y = []
    vel_angular = []

    ctrl_contribution = []
    closest_distance = []
    closest_distance_long = []

    with Reader(bagname) as reader:
        _last_user_msg = None
        _last_remote_msg = None
        _last_front_msg = None
        _last_back_msg = None

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == "/qolo/pose2D":
                msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )
                pose_x.append(msg.x)
                pose_y.append(msg.y)
                pose_theta.append(msg.theta)

            if connection.topic == "/qolo/twist":
                msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )
                vel_x.append(msg.twist.linear.x)
                vel_y.append(msg.twist.linear.y)
                vel_angular.append(msg.twist.angular.z)

            if connection.topic == "/qolo/remote_commands":
                _last_remote_msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )
            if connection.topic == "/qolo/user_commands":
                _last_user_msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )

            if _last_remote_msg is not None and _last_user_msg is not None:
                remote_xphi = _last_remote_msg.data[1:]
                remote_xy = np.array(
                    [remote_xphi[0], robot.control_point[0] * np.cos(remote_xphi[1])]
                )
                user_xphi = _last_user_msg.data[1:]
                user_xy = np.array(
                    [user_xphi[0], robot.control_point[0] * np.cos(user_xphi.data[1])]
                )

                norm_user = LA.norm(user_xy)
                if not norm_user < 0.1:
                    continue

                # print("remote", remote_xphi)
                # print("user", remote_xy)

                ctrl_contribution.append(LA.norm(remote_xy - user_xy) / norm_user)
                closest_distance_long.append(closest_distance[-1])

                _last_user_msg = None
                _last_remote_msg = None

            if connection.topic == "/front_lidar/scan":
                _last_front_msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )

            if connection.topic == "/rear_lidar/scan":
                _last_back_msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )

            if _last_front_msg is not None and _last_back_msg is not None:
                n_min_values = 10

                robot.set_laserscan(
                    _last_front_msg, topic_name="/front_lidar/scan", save_intensity=True
                )

                robot.set_laserscan(
                    _last_back_msg, topic_name="/rear_lidar/scan", save_intensity=True
                )
                laserscan = robot.get_allscan(in_robot_frame=True)

                dists = LA.norm(laserscan, axis=0)
                idx = np.argpartition(dists, n_min_values)
                min_dists = dists[idx[:n_min_values]]
                closest_distance.append(np.mean(min_dists))

                # Calculate minimum distance
                _last_back_msg = None
                _last_front_msg = None

    recording = {
        "position_x": np.array(pose_x),
        "position_y": np.array(pose_y),
        "orientation": np.array(pose_theta),
        "velocity_x": np.array(vel_x),
        "velocity_y": np.array(vel_y),
        "velocity_angular": np.array(vel_y),
        "ctrl_contribution": np.array(ctrl_contribution),
        "closest_distance": np.array(closest_distance),
        "closest_distance_long": np.array(closest_distance_long),
    }

    return recording


def boxplot_of_creator_range(
    record_list, run_dicts, data_key, data_label, figure_name=None
):
    fig, ax_sb = plt.subplots(figsize=(5, 3))

    # Control contribution
    ctrl_contributions = []

    for ii, rdict in enumerate(run_dicts):
        record = record_list[rdict["run"]]

        frac = len(record[data_key]) / len(record["position_x"])
        indmin = int(rdict["range"][0] * frac)
        indmax = int(rdict["range"][1] * frac)

        new_frame = pd.DataFrame(record[data_key][indmin:indmax], columns=[data_label])

        # new_frame["method"] = data_handler.method_names[ii]
        # new_frame["Detection"] = data_handler.detection_type[ii]
        new_frame["Experiment"] = str(ii + 1)

        ctrl_contributions.append(new_frame)

        # print("Experiment", str(ii + 1), len(record[data_key][indmin:indmax]))
        # print("num")

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
        flierprops=dict(markerfacecolor="0.45", markersize=2),
    )

    if figure_name is not None:
        fig.savefig(os.path.join("figures", figure_name))

    # ax_sb.set_xlim([0.5, ax_sb.get_xlim()[1]])
    return fig, ax


def plot_mean_variance(
    record_list, run_dict, data_keys, data_labels=None, save_figure=False
):
    # Setup
    sns.set_style("darkgrid")
    # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=11)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=11)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize
    plt.rc("font", size=11)  # controls default text sizes

    fig, ax = plt.subplots(figsize=(4.0, 3.0), tight_layout=True)

    log_mean_x = []
    mean_x_list = []
    mean_y_list = []

    colors = sns.color_palette("pastel")

    for ii, rdict in enumerate(run_dicts):
        record = record_list[rdict["run"]]

        frac = len(record[data_keys[0]]) / len(record["position_x"])
        indmin = int(rdict["range"][0] * frac)
        indmax = int(rdict["range"][1] * frac)
        data_x = record[data_keys[0]][indmin:indmax]

        frac = len(record[data_keys[1]]) / len(record["position_x"])
        indmin = int(rdict["range"][0] * frac)
        indmax = int(rdict["range"][1] * frac)
        data_y = record[data_keys[1]][indmin:indmax]

        mean_x = np.mean(data_x)
        mean_x_list.append(mean_x)

        mean_y = np.mean(data_y)
        mean_y_list.append(mean_y)

        # ax.scatter(mean_x, mean_y)
        # ax = sns.scatterplot(
        # x=[mean_x[-1]],
        # y=[mean_y[-1]],
        # palette=colors[ii],
        # edgecolors="black",
        # markers="s",
        # s=100,
        # )
        std_x = np.std(data_x)
        std_y = np.std(data_y)

        # ax.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y], color=colors[ii])

        dx = 0.11
        log_x = np.log(mean_x)
        log_x_up = log_x + dx
        x_up = np.exp(log_x_up)
        log_x_dow = log_x - dx
        x_dow = np.exp(log_x_dow)

        log_mean_x.append(log_x)

        ax.plot(
            [x_dow, x_up],
            [mean_y + std_y, mean_y + std_y],
            # color=colors[ii],
            color="black",
            zorder=1,
            # marker="-",
        )
        ax.plot(
            [x_dow, x_up],
            [mean_y - std_y, mean_y - std_y],
            # color=colors[ii],
            color="black",
            zorder=1,
            # marker="-",
        )

        ax.plot(
            [mean_x, mean_x],
            [mean_y - std_y, mean_y + std_y],
            # color=colors[ii],
            color="black",
            zorder=1,
            # marker="-",
        )

        ax.scatter(
            x=mean_x,
            y=mean_y,
            color=colors[ii],
            edgecolors="black",
            # marker="s",
            marker="o",
            s=80,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    x_lim = ax.get_xlim()

    AA = np.vstack([np.log(mean_x_list), np.ones(len(mean_x_list))]).T
    mm, cc = np.linalg.lstsq(AA, mean_y_list, rcond=None)[0]

    y_regr = [np.log(x_lim[0]) * mm + cc, np.log(x_lim[1]) * mm + cc]

    ax.plot(x_lim, [0.45, 0.45], "--", color="#8b0000", linewidth=2, label=r"$d=0.45$m")
    ax.plot(
        x_lim,
        y_regr,
        ":",
        color="black",
        alpha=0.5,
        linewidth=2,
        zorder=-1,
        label="Regression",
    )
    # ax.hlines(0.45, x_lim, ":", color="#8b0000", linewidth=2)
    ax.legend()
    ax.set_xlim(x_lim)

    print(r"Regression approximated as $\Delta^c = exp(((1/mm) D  - cc/mm)$")
    print(f"with values as: mm={np.round(1/mm, 3)}, cc/mm={np.round(cc/mm, 3)}")

    # ax.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y], color="black")
    # s=80, facecolors='none', edgecolors=colors[ii])

    # ax = sns.scatterplot(x=mean_x, y=mean_y, palette="pastel", s=60)
    # std_x = np.std(data_x)
    # ax.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y])

    # ax.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y])

    if data_labels is None:
        ax.set_xlabel(r"Control contribution $\Delta^c$ [log]")
        ax.set_ylabel(r"Closest distance $D^{\mathrm{min}}$ [m]")
    else:
        ax.set_xlabel(data_labels[0])
        ax.set_ylabel(data_labels[1])

    if save_figure:
        fig.savefig(os.path.join("figures", "comparison_distance_control.pdf"))

    return fig, ax


def compare_plots(
    record_list, run_dicts, data_keys, data_labels=None, figure_name=None
):
    # fig, ax = plt.subplots(figsize=(5, 3))

    for ii, rdict in enumerate(run_dicts):
        record = record_list[rdict["run"]]

        frac = len(record[data_keys[0]]) / len(record["position_x"])
        indmin = int(rdict["range"][0] * frac)
        indmax = int(rdict["range"][1] * frac)
        data_x = record[data_keys[0]][indmin:indmax]

        frac = len(record[data_keys[1]]) / len(record["position_x"])
        indmin = int(rdict["range"][0] * frac)
        indmax = int(rdict["range"][1] * frac)
        data_y = record[data_keys[1]][indmin:indmax]

        ax.scatter(data_x, data_y)

    if not data_labels is None:
        ax.set_xlabel(data_labels[0])
        ax.set_ylabel(data_labels[1])

    return fig, ax


if (__name__) == "__main__":
    plt.ion()
    # Bag tuples
    bag_dir = "../data_qolo/marketplace_lausanne_2022_01_28/"

    bag_tuples = [
        # Does one loop (to far away), slow velocity ~ 0.4, important is the start
        ("2022-01-28-13-50-27.bag", "Sampled slow"),
        # Goes back and forwards once
        ("2022-01-28-13-33-32.bag", "Sampled"),
        ("2022-01-28-13-20-25.bag", "Sampled"),  # Very short -> nonusefull
        ("2022-01-28-13-20-46.bag", "Sampled"),
        ("2022-01-28-14-00-39.bag", "Sampled"),
        ("2022-01-28-14-03-04.bag", "Sampled"),
        ("2022-01-28-14-08-47.bag", "Sampled"),
    ]
    bag_names = [name for name, _ in bag_tuples]
    experiment_id = [name[-9:-7] + name[-6:-4] for name, _ in bag_tuples]
    detection_type = [dettype for _, dettype in bag_tuples]

    if False:
        import_list = [0, 1, 3]
        # import_list = None
        logging.info("Default importing in progress...")

        ## Values
        logging.info("Start data-importing.")
        recording_list = []
        for ii, bagname in enumerate(bag_names):
            if not ii in import_list:
                recording_list.append(None)
                continue

            bagname_dir = os.path.join(bag_dir, bagname)

            recording_list.append(get_bagvalues(bagname_dir))

        logging.info("Done importing.")

    ## Plot close up
    if False:
        plt.close("all")

        plt.ion()
        ranges = [None for ii in range(len(recording_list))]

        ranges[0] = [0, 15000]
        ranges[1] = [0, 26000]
        ranges[3] = [0, 40000]

        for ii in [0, 1, 3]:
            record = recording_list[ii]
            fig, axs = plt.subplots(2, 1, figsize=(12, 9))

            indmin = ranges[ii][0]
            indmax = ranges[ii][1]

            axs[0].plot(record["position_x"][indmin:indmax])
            axs[0].plot(record["position_y"][indmin:indmax])

            frac = len(record["velocity_x"]) / len(record["position_x"])
            indmin = int(ranges[ii][0] * frac)
            indmax = int(ranges[ii][1] * frac)

            axs[1].plot(record["velocity_x"][indmin:indmax])
            axs[1].plot(record["velocity_angular"][indmin:indmax])
        plt.show()

    ## Plot different methods
    if False:
        # Setup
        sns.set_style("darkgrid")
        # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
        plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
        plt.rc("legend", fontsize=13)  # legend fontsize
        plt.rc("font", size=13)  # controls default text sizes

    if False:
        plt.close("all")
        run_dicts = [
            {"run": 0, "range": [760, 5900]},
            {"run": 0, "range": [5900, 10280]},
            {"run": 1, "range": [2640, 11240]},
            {"run": 1, "range": [11240, 19000]},
            {"run": 3, "range": [10250, 19100]},
            {"run": 3, "range": [24300, 33250]},
        ]

        fig, ax = boxplot_of_creator_range(
            recording_list,
            run_dicts,
            data_key="ctrl_contribution",
            data_label="Control contribution [ ]",
        )
        ax.set_ylim([-0.05, 1])

        fig, ax = boxplot_of_creator_range(
            recording_list,
            run_dicts,
            data_key="closest_distance",
            data_label="Closest distance [m]",
        )
        ax.plot(ax.get_xlim(), [0.45, 0.45], "-", color="red")

    if True:
        # plot_mean_variance(recording_list, run_dicts, data_key="closest_distance")
        fig, ax = plot_mean_variance(
            recording_list,
            run_dicts,
            data_keys=["ctrl_contribution", "closest_distance_long"],
            # data_labels=["Control contribution [log]", "Closest distance [m]"],
            save_figure=True,
        )

        # compare_plots(
        # recording_list,
        # run_dicts,
        # data_keys=["closest_distance_long", "ctrl_contribution"],
        # data_labels=["Closest distance [m]", "Control contribution [ ]"],
        # )
