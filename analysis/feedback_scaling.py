"""
All feedback related scaling functions can go in this file
"""

from collections import defaultdict
from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

from analysis.helper import *
from utils.functions import update_progress
from utils.log import *

primary_colors = ["#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
                  "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
                  "#8BC34A", "#CDDC39"]

second_colors = ["#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9", "#C5CAE9",
                 "#BBDEFB", "#B3E5FC", "#B2EBF2", "#B2DFDB", "#C8E6C9",
                 "#DCEDC8", "#F0F4C3"]

# Initial setup
system = S_OPEN_2

ss_end_time = 4000
ss_recovery_time = 4000

depletion_percentage = 15
desired_recovery_percentage = 90

initial_time = np.linspace(0, ss_end_time, ss_end_time * 10)
recovery_time = np.linspace(0, ss_recovery_time, ss_recovery_time * 10)

# feedback parameters
range_hill = [0.5, 1, 2]
range_feed_type = [FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE]
range_carry = np.linspace(0.1, 10, 10)
range_multi = np.linspace(1, 10, 10)
range_substrate = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG,
                   I_ERPI]
range_enz = [E_PITP, E_PI4K, E_PIP5K, E_PLC, E_DAGK, E_LAZA, E_PATP, E_CDS,
             E_PIS, E_SINK, E_SOURCE]


def get_ss(ini_cond, enzymes, feed_para=None):
    return odeint(get_equations(system), ini_cond,
                  initial_time, args=(enzymes, feed_para))[-1]


def give_stimulus(ini_cond):
    new_cond = [x for x in ini_cond]
    amount = ini_cond[I_PIP2] * depletion_percentage / 100
    new_cond[I_DAG] = new_cond[I_DAG] + new_cond[I_PIP2] - amount
    new_cond[I_PIP2] = amount
    return new_cond


def get_parameter(data: str):
    return convert_to_enzyme(extract_enz_from_log(data))


def get_recovery_time(ini_cond, enzymes, feed_para=None):
    return odeint(get_equations(system), ini_cond, recovery_time,
                  args=(enzymes, feed_para))


def get_desired_recovery_time(con_array, ss_pip2):
    ar = np.asarray(con_array[:, I_PIP2])
    req_conc = ss_pip2 * desired_recovery_percentage / 100
    try:
        return recovery_time[[x > req_conc for x in ar].index(True)]
    except ValueError:
        return recovery_time[-1]  # too much delay or bistable system


def do_scaling(filename: str):
    log_data = {
        "UID": CURRENT_JOB,
        "system": system,
        "Analysis": "Feedback Scan with Scaling",
        "desired_recovery": desired_recovery_percentage,
        "depletion_percentage": depletion_percentage,
        "version": "3.0"}
    LOG.info(json.dumps(log_data, sort_keys=True))

    with open(filename) as f:
        enzymes = get_parameter(f.read())
        init_conc = get_random_concentrations(200, system)
        starting_ss = get_ss(init_conc, enzymes)
        plc_base = enzymes[E_PLC].v
        for e in enzymes:
            if e != E_SOURCE:
                enzymes[e].k = enzymes[e].k / sum(starting_ss)
                enzymes[e].v = enzymes[e].v / plc_base
            else:
                enzymes[e].k = enzymes[e].k / plc_base

        ss_lipids = get_ss(init_conc, enzymes)
        ss_pip2 = ss_lipids[I_PIP2]
        stim = give_stimulus(ss_lipids)

        progress_counter = 0
        total_size = len(list(product(
            *[range_hill, range_carry, range_multi, range_feed_type,
              range_substrate, range_enz])))

        for hill, carry, multi, fed_type, sub_ind, enz in product(
                *[range_hill, range_carry, range_multi, range_feed_type,
                  range_substrate, range_enz]):
            update_progress(progress_counter / total_size)
            progress_counter += 1

            fed_para = {
                F_HILL_COEFFICIENT: hill,
                F_FEED_SUBSTRATE_INDEX: sub_ind,
                F_TYPE_OF_FEEDBACK: fed_type,
                F_CARRYING_CAPACITY: carry,
                F_MULTIPLICATION_FACTOR: multi,
                F_ENZYME: enz
            }

            without_feed_re_time = get_recovery_time(stim, enzymes)
            without_feed = get_desired_recovery_time(without_feed_re_time,
                                                     ss_pip2)
            with_feed_re_time = get_recovery_time(stim, enzymes, fed_para)
            with_feed = get_desired_recovery_time(with_feed_re_time, ss_pip2)

            data = {
                "Enzymes": {e: enzymes[e].properties for e in enzymes},
                "fed_para": fed_para,
                "without_feed": without_feed,
                "with_feed": with_feed
            }
            OUTPUT.info(json.dumps(data, sort_keys=True))


def get_lipid_from_index(ind: int):
    r_s = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG, I_ERPI]
    r_n = [L_PMPI, L_PI4P, L_PIP2, L_DAG, L_PMPA, L_ERPA, L_CDPDAG, L_ERPI]
    return r_n[r_s.index(ind)]


class VisualClass:
    def __init__(self, data):
        self.data = json.loads(data)
        self.enzymes = self.data["Enzymes"]
        self.feed_para = self.data["fed_para"]
        self.with_feed = self.data["with_feed"]
        self.without_feed = self.data["without_feed"]
        self.diff = self.without_feed / self.with_feed
        self.type_of_feedback = self.feed_para[F_TYPE_OF_FEEDBACK]
        self.hill_coefficient = self.feed_para[F_HILL_COEFFICIENT]
        self.carrying_capacity = self.feed_para[F_CARRYING_CAPACITY]
        self.multiplication_factor = self.feed_para[F_MULTIPLICATION_FACTOR]
        self.feedback_enzyme = self.feed_para[F_ENZYME]
        self.feedback_substrate_index = self.feed_para[
            F_FEED_SUBSTRATE_INDEX]
        self.feedback_substrate_name = get_lipid_from_index(
            self.feedback_substrate_index)


def get_data():
    all_data = []
    with open("output/output.log") as f:
        for line in f:
            all_data.append(VisualClass(line.split(":", 1)[1].strip()))

    return all_data


def general_core():
    all_data = get_data()

    diff_positive = [x.diff for x in all_data if
                     x.type_of_feedback == FEEDBACK_POSITIVE]
    diff_negative = [x.diff for x in all_data if
                     x.type_of_feedback == FEEDBACK_NEGATIVE]
    plt.hist(diff_positive, alpha=0.5, label="Positive Feedback")
    plt.hist(diff_negative, alpha=0.5, label="Negative Feedback")
    plt.axvline(1, linestyle="--", color="k")
    plt.xlabel("Without Feedback/With Feedback (PIP2 recovery)")
    plt.ylabel("Frequency")
    plt.legend(loc=0)
    plt.savefig("general.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_lipid_wise():
    all_data = get_data()
    lipid_wise_pos = defaultdict(list)
    lipid_wise_neg = defaultdict(list)
    for p in all_data:
        if p.type_of_feedback == FEEDBACK_POSITIVE:
            lipid_wise_pos[p.feedback_substrate_name].append(p.diff)
        else:
            lipid_wise_neg[p.feedback_substrate_name].append(p.diff)

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
        ax.set_yticks([])
        ax.set_title(m)
        grid_count += 1

    plt.savefig("lipid_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def check_enzyme_wise():
    all_data = get_data()
    enzyme_wise_pos = defaultdict(list)
    enzyme_wise_neg = defaultdict(list)
    for p in all_data:
        if p.type_of_feedback == FEEDBACK_POSITIVE:
            enzyme_wise_pos[p.feedback_enzyme].append(p.diff)
        else:
            enzyme_wise_neg[p.feedback_enzyme].append(p.diff)

    gs = gridspec.GridSpec(3, 4)
    grid_count = 0
    for m in enzyme_wise_pos:
        ax = plt.subplot(gs[grid_count])
        ax.hist(enzyme_wise_pos[m], alpha=0.5, color=second_colors[
            grid_count])
        ax.hist(enzyme_wise_neg[m], alpha=0.5, color=primary_colors[
            grid_count])
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(m)
        grid_count += 1

    plt.savefig("enzyme_wise.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def single_enzyme(enzyme_name):
    all_data = get_data()
    lipid_wise_pos = defaultdict(list)
    lipid_wise_neg = defaultdict(list)
    for p in all_data:
        if p.feedback_enzyme == enzyme_name:
            if p.type_of_feedback == FEEDBACK_POSITIVE:
                lipid_wise_pos[p.feedback_substrate_name].append(p.diff)
            else:
                lipid_wise_neg[p.feedback_substrate_name].append(p.diff)

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
        ax.set_yticks([])
        ax.set_title(m)
        grid_count += 1

    plt.savefig("%s.png" % enzyme_name, format='png', dpi=300,
                bbox_inches='tight')
    plt.show()


def visualize():
    # general_core()
    # check_lipid_wise()
    # check_enzyme_wise()
    single_enzyme(E_PI4K)
