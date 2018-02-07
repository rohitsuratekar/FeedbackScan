"""
All visualization related functions
"""

import json
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

from constants.namespace import *

primary_colors = ["#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
                  "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
                  "#8BC34A", "#CDDC39"]

second_colors = ["#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9", "#C5CAE9",
                 "#BBDEFB", "#B3E5FC", "#B2EBF2", "#B2DFDB", "#C8E6C9",
                 "#DCEDC8", "#F0F4C3"]


def get_lipid_from_index(ind: int):
    r_s = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG, I_ERPI]
    r_n = [L_PMPI, L_PI4P, L_PIP2, L_DAG, L_PMPA, L_ERPA, L_CDPDAG, L_ERPI]
    return r_n[r_s.index(ind)]


def plot_hist(nor, fed):
    plt.axvline(nor[0], linestyle="--")
    plt.hist(fed, alpha=0.7)


def plot_lipid_wise(points, recovery_time_point, enzyme):
    fed_recovery_positive = defaultdict(list)
    fed_recovery_negative = defaultdict(list)
    all_normal = []

    ind = points[0].recovery_time_points.index(recovery_time_point)
    for a in points:
        all_normal.append(a.normal_recovery[ind])
        if a.feedback_enzyme == enzyme:
            if a.type_of_feedback == FEEDBACK_POSITIVE:
                fed_recovery_positive[a.feedback_substrate_name].append(
                    a.feedback_recovery[ind])
            else:
                fed_recovery_negative[a.feedback_substrate_name].append(
                    a.feedback_recovery[ind])

    print(set(all_normal))

    gs = gridspec.GridSpec(3, 3)
    grid_count = 0
    for label in fed_recovery_positive:
        ax = plt.subplot(gs[grid_count])
        ax.hist(fed_recovery_negative[label], color=second_colors[
            grid_count])
        ax.hist(fed_recovery_positive[label], color=primary_colors[grid_count])
        ax.axvline(points[0].normal_recovery[ind], linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
        grid_count += 1


def plot_all_points(points, feedback_type):
    positive_points = defaultdict(list)
    negative_points = defaultdict(list)

    for a in points:
        if a.type_of_feedback == FEEDBACK_POSITIVE:
            for k in a.recovery_time_points:
                positive_points[k].append(a.feedback_recovery[
                                              a.recovery_time_points.index(k)])
        else:
            for k in a.recovery_time_points:
                negative_points[k].append(a.feedback_recovery[
                                              a.recovery_time_points.index(k)])

    c = 0

    if feedback_type == FEEDBACK_POSITIVE:
        p = positive_points
        title = "Positive Feedback"
    else:
        p = negative_points
        title = "Negative Feedback"

    for k in sorted(p, reverse=True):
        plt.hist(p[k], 30, color=primary_colors[c], label=str(k * 100) \
                                                          + "% "
                                                            "recovery",
                 alpha=0.8)
        ind = points[0].recovery_time_points.index(k)
        plt.axvline(points[0].normal_recovery[ind], linestyle="--", color="k")
        c += 1

    plt.title(title + "\n(shown only points between 0-100)")
    plt.xlabel("Recovery Time")
    plt.ylabel("Frequency")
    plt.xlim(0, 100)
    plt.legend(loc=0)
    plt.show()


def plot_enzyme_wise(points, recovery_time_point, enzyme):
    fed_recovery_positive = defaultdict(list)
    fed_recovery_negative = defaultdict(list)
    ind = points[0].recovery_time_points.index(recovery_time_point)
    all_normal = []
    for a in points:
        all_normal.append(a.normal_recovery[ind])
        if a.type_of_feedback == FEEDBACK_POSITIVE:
            fed_recovery_positive[a.feedback_enzyme].append(
                a.feedback_recovery[ind])
        else:
            fed_recovery_negative[a.feedback_enzyme].append(
                a.feedback_recovery[ind])

    print(set(all_normal))
    gs = gridspec.GridSpec(3, 4)
    grid_count = 0
    for label in fed_recovery_positive:
        ax = plt.subplot(gs[grid_count])
        ax.hist(fed_recovery_negative[label], color=second_colors[
            grid_count])
        ax.hist(fed_recovery_positive[label], color=primary_colors[grid_count])
        ax.axvline(points[0].normal_recovery[ind], linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
        grid_count += 1


class SimpleOutput:
    def __init__(self, data: dict):
        self.data = data
        self.recovery_time_points = data["Diff"]["val"]
        self.normal_recovery = data["Diff"]["nor"]
        self.feedback_recovery = data["Diff"]["feed"]
        self.sum = data["Diff"]["sum"]
        self.fed_para = data["fed_para"]
        self.type_of_feedback = data["fed_para"][F_TYPE_OF_FEEDBACK]
        self.hill_coefficient = data["fed_para"][F_HILL_COEFFICIENT]
        self.carrying_capacity = data["fed_para"][F_CARRYING_CAPACITY]
        self.multiplication_factor = data["fed_para"][F_MULTIPLICATION_FACTOR]
        self.feedback_enzyme = data["fed_para"][F_ENZYME]
        self.feedback_substrate_index = data["fed_para"][
            F_FEED_SUBSTRATE_INDEX]
        self.enzyme_data = data["Enzymes"]
        self.feedback_substrate_name = get_lipid_from_index(
            self.feedback_substrate_index)


def plot_simple(filename: str):
    all_data = []
    with open(filename) as f:
        for line in f:
            all_data.append(
                SimpleOutput(json.loads(line.split(":", 1)[1].strip())))

    recovery_dic_normal = defaultdict(list)
    recovery_dic_feedback = defaultdict(list)

    for a in all_data:
        for k in a.recovery_time_points:
            recovery_dic_normal[k].append(a.normal_recovery[
                                              a.recovery_time_points.index(k)])
            recovery_dic_feedback[k].append(a.feedback_recovery[
                                                a.recovery_time_points.index(
                                                    k)])

    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    # plot_lipid_wise(all_data, 0.9, E_CDS)
    plot_all_points(all_data, FEEDBACK_POSITIVE)
    plt.show()
