from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from analysis.analysis_settings import *
from analysis.feedback_multiple import get_recovery_points
from analysis.helper import *

primary_colors = ["#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
                  "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
                  "#8BC34A", "#CDDC39"]

second_colors = ["#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9", "#C5CAE9",
                 "#BBDEFB", "#B3E5FC", "#B2EBF2", "#B2DFDB", "#C8E6C9",
                 "#DCEDC8", "#F0F4C3"]


def get_lipid_from_index(ind: int) -> str:
    """
    Returns name of lipid based on standard index of lipid
    Standard index values are presented in "constants" package
    :param ind: standard index of lipid
    :return: name of lipid
    """
    r_s = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG, I_ERPI]
    r_n = [L_PMPI, L_PI4P, L_PIP2, L_DAG, L_PMPA, L_ERPA, L_CDPDAG, L_ERPI]
    return r_n[r_s.index(ind)]


class VisualClass:
    """
    Simple class to hold raw data
    """

    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.data = json.loads(raw_data.split(":", 1)[1])
        self.enzyme_data = self.data["Enzymes"]
        self.fed_para = self.data["fed_para"]
        self.feed_enzymes = []
        self.feed_substrates = []
        self.feedback_type = []
        for k in self.fed_para:
            self.feed_enzymes.append(k)
            self.feed_substrates.append(
                self.fed_para[k][F_FEED_SUBSTRATE_INDEX])
            self.feedback_type.append(self.fed_para[k][F_TYPE_OF_FEEDBACK])

        self.min_pi4p = self.data["min_pi4p"]
        self.pi4p_timings = self.data["pi4p_timings"]
        self.pip2_timings = self.data["pip2_timings"]
        self.ss_dif_pi4p = self.data["ss_dif_pi4p"]
        self.ss_dif_pip2 = self.data["ss_dif_pip2"]
        if self.ss_dif_pip2 < 0:
            print(raw_data)

    @property
    def enzymes(self):
        return convert_to_enzyme(self.enzyme_data)

    def get_pip2_ratio(self, nf_data: dict):
        return np.asanyarray(
            nf_data["pip2_timings"]) / np.asanyarray(self.pip2_timings)


def get_para_sets(filename: str) -> list:
    """
    Extracts parameters from the file and converts into VisualClass
    :param filename: Name of input file
    :return: list of VisualClass elements
    """
    all_sets = []
    with open(filename) as f:
        for line in f:
            all_sets.append(VisualClass(line.strip()))

    return all_sets


def convert_into_data(recovery_array, ss_lipids) -> dict:
    ar_pip2 = np.asarray(recovery_array[:, I_PIP2])
    ar_pi4p = np.asarray(recovery_array[:, I_PI4P])

    pip2_timings = []
    pi4p_timings = []
    pi4p_depletion = min(recovery_array[:, I_PI4P])

    for point in RECOVERY_POINTS:
        req_con = ss_lipids[I_PIP2] * point / 100
        try:
            i = recovery_time[[x > req_con for x in ar_pip2].index(True)]
            if i == 0.0:
                pip2_timings.append(ar_pip2[1])
            else:
                pip2_timings.append(i)
        except ValueError:
            pip2_timings.append(-1989)  # Too Slow or numerical error

    for point in RECOVERY_POINTS:
        req_con = ss_lipids[I_PI4P] * point / 100
        try:
            i = recovery_time[[x > req_con for x in ar_pi4p].index(True)]
            if i == 0.0:
                pi4p_timings.append(ar_pi4p[1])
            else:
                pi4p_timings.append(i)
        except ValueError:
            pi4p_timings.append(-1989)  # Too Slow or numerical error

    return {
        "pip2_timings": pip2_timings,
        "pi4p_timings": pi4p_timings,
        "min_pi4p": pi4p_depletion,
        "ss_dif_pip2": recovery_array[-1][I_PIP2] / ss_lipids[I_PIP2],
        "ss_dif_pi4p": recovery_array[-1][I_PI4P] / ss_lipids[I_PI4P]
    }


def single_feedback(filename: str):
    all_para = get_para_sets(filename)
    nf_recovery = get_recovery_points(all_para[0].enzymes, None)
    nf_data = convert_into_data(nf_recovery, nf_recovery[-1])
    pip2_ratio_positive = []
    pip2_ratio_negative = []
    for p in all_para:
        para = p  # type:VisualClass
        # r = para.get_pip2_ratio(nf_data)
        r = para.ss_dif_pip2
        if 0.9 < r < 1.1:

            if para.feedback_type[0] == FEEDBACK_POSITIVE:
                pip2_ratio_positive.append(para.get_pip2_ratio(nf_data))
            else:
                pip2_ratio_negative.append(para.get_pip2_ratio(nf_data))

    pip2_ratio_positive = np.asanyarray(pip2_ratio_positive)
    pip2_ratio_negative = np.asanyarray(pip2_ratio_negative)

    plt.hist(pip2_ratio_positive[:, 3], alpha=0.5)
    plt.hist(pip2_ratio_negative[:, 3], alpha=0.5)
    plt.axvline(1, color="k", linestyle="--")
    plt.yscale("log")
    plt.show()


def sanity_check_ss_diff(filename: str) -> None:
    """
    Plots lipid wise feedback distribution
    """
    all_data = get_para_sets(filename)
    checked_para = []
    for para in all_data:
        checked_para.append(para.ss_dif_pip2)

    checked_para = [x for x in checked_para if str(x) != 'nan']
    plt.hist(checked_para, 100)
    # plt.savefig("lipid_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_lipid_wise(filename: str) -> None:
    """
    Plots lipid wise feedback distribution
    """
    all_data = get_para_sets(filename)
    nf_recovery = get_recovery_points(all_data[0].enzymes, None)
    nf_data = convert_into_data(nf_recovery, nf_recovery[-1])
    lipid_wise_pos = defaultdict(list)
    lipid_wise_neg = defaultdict(list)
    for para in all_data:
        if para.feedback_type[0] == FEEDBACK_POSITIVE:
            lipid_wise_pos[para.feed_substrates[0]].append(
                para.get_pip2_ratio(nf_data)[3])
        else:
            lipid_wise_neg[para.feed_substrates[0]].append(
                para.get_pip2_ratio(nf_data)[3])

    gs = gridspec.GridSpec(2, 4)
    grid_count = 0

    for m in lipid_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(lipid_wise_pos[m], alpha=0.5, color=second_colors[
            grid_count])
        ax.hist(lipid_wise_neg[m], alpha=0.5, color=primary_colors[
            grid_count])
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_title(get_lipid_from_index(m))
        grid_count += 1

    # plt.savefig("lipid_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_enzyme_wise(filename: str) -> None:
    """
    Plots lipid wise feedback distribution
    """
    all_data = get_para_sets(filename)
    nf_recovery = get_recovery_points(all_data[0].enzymes, None)
    nf_data = convert_into_data(nf_recovery, nf_recovery[-1])
    enzyme_wise_pos = defaultdict(list)
    enzyme_wise_neg = defaultdict(list)
    for para in all_data:

        if para.feedback_type[0] == FEEDBACK_POSITIVE:
            enzyme_wise_pos[para.feed_enzymes[0]].append(
                para.get_pip2_ratio(nf_data)[3])
        else:
            enzyme_wise_neg[para.feed_enzymes[0]].append(
                para.get_pip2_ratio(nf_data)[3])

    gs = gridspec.GridSpec(3, 4)
    grid_count = 0

    for m in enzyme_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(enzyme_wise_pos[m], alpha=0.5, color=second_colors[
            grid_count])
        ax.hist(enzyme_wise_pos[m], alpha=0.5, color=primary_colors[
            grid_count])
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yscale("log")
        ax.set_yticks([])
        ax.set_title(m)
        grid_count += 1

    # plt.savefig("enzyme_wise.png", format='png', dpi=300,
    # bbox_inches='tight')
    plt.show()


def visualize(filename: str):
    sanity_check_ss_diff(filename)
    # single_feedback(filename)
    # check_lipid_wise(filename)
    # check_enzyme_wise(filename)
