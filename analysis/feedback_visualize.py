"""
All visualization related plots
"""
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

from analysis.analysis_settings import *
from analysis.feedback_scaling import recovery_time
from analysis.helper import *

DIFF_INDEX = 4


class VisualizeSingle:
    def __init__(self, data: str, wf_para: list):
        self.raw_data = data
        self.wf_para = wf_para
        split_data = json.loads(data.split(":", 1)[1])
        self.enzymes = split_data["Enzymes"]
        self.feed_para = split_data["fed_para"]
        self.min_pi4p = split_data["min_pi4p"]
        self.normalized_pi4p_depletion = self.min_pi4p / wf_para[I_PI4P]
        self.pi4p_timings = split_data["pi4p_timings"]
        self.pip2_timings = split_data["pip2_timings"]
        self.ss_dif_pi4p = split_data["ss_dif_pi4p"]
        self.ss_dif_pip2 = split_data["ss_dif_pip2"]
        self.diff = np.asanyarray(wf_para) / np.asanyarray(self.pip2_timings)
        for k in self.feed_para:
            self.feedback_enzyme = k
        self.type_of_feedback = self.feed_para[self.feedback_enzyme][
            F_TYPE_OF_FEEDBACK]
        self.hill_coefficient = self.feed_para[self.feedback_enzyme][
            F_HILL_COEFFICIENT]
        self.carrying_capacity = self.feed_para[self.feedback_enzyme][
            F_CARRYING_CAPACITY]
        self.multiplication_factor = self.feed_para[self.feedback_enzyme][
            F_MULTIPLICATION_FACTOR]

        self.feedback_substrate_index = self.feed_para[self.feedback_enzyme][
            F_FEED_SUBSTRATE_INDEX]
        self.feedback_substrate_name = get_lipid_from_index(
            self.feedback_substrate_index)


def get_without_feed_para(enz, system) -> list:
    init_con = get_random_concentrations(1, system)
    init_time = np.linspace(0, 10000, 10000)
    no_feed_ss = odeint(get_equations(system), init_con, init_time,
                        args=(enz, None))[-1]
    stim = give_stimulus(no_feed_ss, PERCENTAGE_DEPLETION)
    recovery = odeint(get_equations(system), stim, recovery_time,
                      args=(enz, None))
    ar_pip2 = np.asarray(recovery[:, I_PIP2])
    pip2_timings = []
    for point in RECOVERY_POINTS:
        req_con = no_feed_ss[I_PIP2] * point / 100
        try:
            i = recovery_time[[x > req_con for x in ar_pip2].index(True)]
            if i == 0.0:
                pip2_timings.append(recovery_time[1])
            else:
                pip2_timings.append(i)
        except ValueError:
            pip2_timings.append(-1989)

    return pip2_timings


def get_parameters(filename: str, system: str) -> list:
    para = []
    without_feed = None
    with open(filename) as f:
        for line in f:
            if without_feed is None:
                enz = convert_to_enzyme(extract_enz_from_log(line))
                without_feed = get_without_feed_para(enz, system)
            para.append(VisualizeSingle(line.strip(), without_feed))

    return para


def general_core(output_file: str, system: str) -> None:
    """
    Plots histogram for Positive and Negative feedback
    """
    all_data = get_parameters(output_file, system)

    diff_positive = [x.diff[DIFF_INDEX] for x in all_data if
                     x.type_of_feedback == FEEDBACK_POSITIVE]
    diff_negative = [x.diff[DIFF_INDEX] for x in all_data if
                     x.type_of_feedback == FEEDBACK_NEGATIVE]
    plt.hist(diff_positive, 10, alpha=0.5, label="Positive interaction",
             color="b")
    plt.hist(diff_negative, 10, alpha=0.5, label="Negative interaction",
             color="r")
    plt.axvline(1, linestyle="--", color="k")
    plt.yscale("log")
    plt.xlabel("90% PIP$_2$ recovery (Without Feedback/With Feedback)")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend(loc=0)
    plt.savefig("general.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def depletion_plot(output_file: str, system: str) -> None:
    """
    Plots histogram for Positive and Negative feedback
    """
    all_data = get_parameters(output_file, system)

    diff_positive = [x.normalized_pi4p_depletion for x in all_data if
                     x.type_of_feedback == FEEDBACK_POSITIVE]
    diff_negative = [x.normalized_pi4p_depletion for x in all_data if
                     x.type_of_feedback == FEEDBACK_NEGATIVE]
    plt.hist(diff_positive, 10, alpha=0.5, label="Positive interaction",
             color="b")
    plt.hist(diff_negative, 10, alpha=0.5, label="Negative interaction",
             color="r")
    plt.axvline(0.85, linestyle="--", color="k")
    plt.yscale("log")
    plt.xlabel("Depletion of PI4P with respect to its steady state")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend(loc=0)
    plt.savefig("pi4p_depletion.png", format='png', dpi=300,
                bbox_inches='tight')
    plt.show()


def pi4p_pip2_timing(output_file, system) -> None:
    """
    Plots histogram of PI4P depletion
    """
    all_data = get_parameters(output_file, system)
    diff_positive = [x.pi4p_timings[DIFF_INDEX] / x.pip2_timings[DIFF_INDEX]
                     for x in all_data if
                     x.type_of_feedback == FEEDBACK_POSITIVE]
    diff_negative = [x.pi4p_timings[DIFF_INDEX] / x.pip2_timings[DIFF_INDEX]
                     for x in all_data if
                     x.type_of_feedback == FEEDBACK_NEGATIVE]
    plt.hist(diff_positive, 10, alpha=0.5, label="Positive Feedback",
             color="b")
    plt.hist(diff_negative, 10, alpha=0.5, label="Negative Feedback",
             color="r")
    plt.yscale("log")
    plt.axvline(1, linestyle="--", color="k")
    plt.xlabel("time to 90% recovery (PI4P/PIP$_2$)")
    plt.ylabel("Frequency")
    plt.legend(loc=0)
    plt.savefig("pi4p_pip2_timing.png", format='png', dpi=300,
                bbox_inches='tight')
    plt.show()


def pi4p_to_pip2_all_depletion(output_file, system) -> None:
    """
    Plots lipid wise feedback distribution
    """
    all_data = get_parameters(output_file, system)
    lipid_wise_pos = defaultdict(list)
    lipid_wise_neg = defaultdict(list)
    for p in all_data:
        for m in RECOVERY_POINTS:
            b = p.pi4p_timings[RECOVERY_POINTS.index(m)] / p.pip2_timings[
                RECOVERY_POINTS.index(m)]
            if b > 0:
                if p.type_of_feedback == FEEDBACK_POSITIVE:
                    lipid_wise_pos[str(m)].append(b)
                else:
                    lipid_wise_neg[str(m)].append(b)

    gs = gridspec.GridSpec(3, 2)
    grid_count = 0

    for m in lipid_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(lipid_wise_pos[m], 10, alpha=0.5, color="b")
        ax.hist(lipid_wise_neg[m], 10, alpha=0.5, color="r")
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_title("%s %% of Steady State" % m)
        # ax.set_xlim(0, 2)
        grid_count += 1

    plt.savefig("lipid_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_lipid_wise(output_file, system) -> None:
    """
    Plots lipid wise feedback distribution
    """
    all_data = get_parameters(output_file, system)
    lipid_wise_pos = defaultdict(list)
    lipid_wise_neg = defaultdict(list)
    for p in all_data:
        if p.type_of_feedback == FEEDBACK_POSITIVE:
            lipid_wise_pos[p.feedback_substrate_name].append(
                p.diff[DIFF_INDEX])
        else:
            lipid_wise_neg[p.feedback_substrate_name].append(p.diff[
                                                                 DIFF_INDEX])

    gs = gridspec.GridSpec(3, 3)
    grid_count = 0

    for m in lipid_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(lipid_wise_pos[m], 10, alpha=0.5, color="b")
        ax.hist(lipid_wise_neg[m], 10, alpha=0.5, color="r")
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_title(m)
        # ax.set_xlim(0, 2)
        grid_count += 1

    plt.savefig("lipid_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_enzyme_wise(output_file, system) -> None:
    """
    Plots enzyme-wise feedback distribution
    """

    all_data = get_parameters(output_file, system)
    enzyme_wise_pos = defaultdict(list)
    enzyme_wise_neg = defaultdict(list)
    for p in all_data:
        if p.type_of_feedback == FEEDBACK_POSITIVE:
            enzyme_wise_pos[p.feedback_enzyme].append(p.diff[DIFF_INDEX])
        else:
            enzyme_wise_neg[p.feedback_enzyme].append(p.diff[DIFF_INDEX])

    gs = gridspec.GridSpec(4, 3)
    grid_count = 0
    for m in enzyme_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(enzyme_wise_pos[m], 10, alpha=0.5, color="b")
        ax.hist(enzyme_wise_neg[m], 10, alpha=0.5, color="r")
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_title(m)
        grid_count += 1

    plt.savefig("enzyme_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize(output_file: str, system: str):
    # general_core(output_file, system)
    # pi4p_pip2_timing(output_file, system)
    # pi4p_to_pip2_all_depletion(output_file, system)
    # depletion_plot(output_file, system)
    # check_lipid_wise(output_file, system)
    check_enzyme_wise(output_file, system)
