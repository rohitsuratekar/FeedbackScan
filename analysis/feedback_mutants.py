"""
All mutant related analysis will go in this file.
This will not scan for new parameters. It will take input from given file
and then use parameter set from that file to do mutant analysis

Use 'feedback_scaling.py" to do initial analysis
"""

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

from analysis.helper import *
from utils.functions import update_progress
from utils.log import *

primary_colors = ["#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
                  "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
                  "#8BC34A", "#CDDC39"]

# Initial setup
system = S_OPEN_2

ss_end_time = 4000
ss_recovery_time = 4000

depletion_percentage = 15
desired_recovery_percentage = 90

initial_time = np.linspace(0, ss_end_time, ss_end_time * 10)
recovery_time = np.linspace(0, ss_recovery_time, ss_recovery_time * 10)

# First recovery point is 16 which is immediately after 15% depletion
# 76.5 is 90% of whatever is left after 15% depletion
recovery_points = [16, 30, 50, 76.5, 90, 95]  # Get data at these points


def get_initial_ss(ini_cond, enzymes, feed_para):
    """
    :param ini_cond: Initial condition array
    :param enzymes: Dictionary of Enzymes
    :param feed_para: Feedback parameters if any
    :return: array of numbers after solving ODE system
    """
    return odeint(get_equations(system), ini_cond,
                  initial_time, args=(enzymes, feed_para))[-1]


def get_recovery_time(ini_cond, enzymes, feed_para):
    """
    :param ini_cond: Initial Conditions
    :param enzymes: Dictionary of enzymes
    :param feed_para: Feedback parameters if any
    :return: array of numbers after solving ODE system
    """
    return odeint(get_equations(system), ini_cond, recovery_time,
                  args=(enzymes, feed_para))


def get_parameter_sets(filename: str) -> list:
    """
    :param filename: Name of file from which parameters are extracted
    :return: List of tuple with [0] Enzyme dictionary [1] Feedback Para
    """
    all_para = []
    with open(filename) as f:
        for line in f:
            enz = convert_to_enzyme(extract_enz_from_log(line))
            fed = json.loads(line.strip().split(":", 1)[1])["fed_para"]
            all_para.append((enz, fed))

    return all_para


def give_stimulus(ini_cond):
    """
    Note: This is approximate method and do not consider changes in any
    other lipid except PIP2 and DAG. Use this method only in initial
    analysis as this will cut down HUGE computational power
    :param ini_cond: Initial conditions
    :return: Lipid concentrations after stimulation
    """
    new_cond = [x for x in ini_cond]
    amount = ini_cond[I_PIP2] * depletion_percentage / 100
    new_cond[I_DAG] = new_cond[I_DAG] + new_cond[I_PIP2] - amount
    new_cond[I_PIP2] = amount
    return new_cond


def get_desired_recovery_time(con_array, lipid_index, ss_lipid) -> \
        list:
    """
    :param con_array: Recovery array
    :param lipid_index: Standard Index of Lipid
    :param ss_lipid: Steady state of the same lipid before stimulation
    :return: list containing recovery timing at given percentage recovery
    """
    ar = np.asarray(con_array[:, lipid_index])

    recovery_array = []

    for point in recovery_points:
        req_con = (ss_lipid * point / 100)
        try:
            recovery_array.append(
                recovery_time[[x > req_con for x in ar].index(True)])
        except ValueError:
            recovery_array.append(recovery_time[-1])  # too much delay or
            # bistable system

    return recovery_array


def do_mutant_analysis(filename: str):
    """
    Main function for mutant analysis
    :param filename: Name of file from which parameter sets are taken
    """

    # Log the analysis details
    log_data = {
        "UID": CURRENT_JOB,
        "system": system,
        "Analysis": "Mutant scan in Feedback",
        "desired_recovery": desired_recovery_percentage,
        "depletion_percentage": depletion_percentage,
        "recovery_points": recovery_points,
        "version": "3.0"}
    LOG.info(json.dumps(log_data, sort_keys=True))

    # Sanity Check
    if filename.strip().lower() == "output/output.log":
        raise Exception("You are using same input and output file")

    progress_counter = 0
    all_para = get_parameter_sets(filename)
    for para in all_para:
        update_progress(progress_counter / len(all_para))
        enzymes = para[0]
        feed_para = para[1]

        # Beware to add scaled enzyme parameters
        init_con = get_random_concentrations(1, system)

        # Wild Type
        wt_starting_ss = get_initial_ss(init_con, enzymes, feed_para)
        wt_stm_ss = give_stimulus(wt_starting_ss)
        wt_rec_array = get_recovery_time(wt_stm_ss, enzymes, feed_para)
        wt_points = get_desired_recovery_time(wt_rec_array, I_PIP2,
                                              wt_starting_ss[I_PIP2])

        # Generate Mutant
        enzymes[E_PIP5K].v *= 0.1 * enzymes[E_PIP5K].v

        # Mutant
        mt_starting_ss = get_initial_ss(init_con, enzymes, feed_para)
        mt_stm_ss = give_stimulus(mt_starting_ss)
        mt_rec_array = get_recovery_time(mt_stm_ss, enzymes, feed_para)
        mt_points = get_desired_recovery_time(mt_rec_array, I_PIP2,
                                              mt_starting_ss[I_PIP2])

        # Revert Enzyme parameter
        enzymes[E_PIP5K].v *= 10 * enzymes[E_PIP5K].v

        data = {
            "Enzymes": {e: enzymes[e].properties for e in enzymes},
            "fed_para": feed_para,
            "wt_recovery": wt_points,
            "mt_recovery": mt_points
        }
        OUTPUT.info(json.dumps(data, sort_keys=True))
        progress_counter += 1


class VisualModel:
    def __init__(self, data):
        self.data = json.loads(data)
        self.enzymes_data = self.data["Enzymes"]
        self.mt_recovery = self.data["mt_recovery"]
        self.fed_para = self.data["fed_para"]
        self.wt_recovery = self.data["wt_recovery"]


def mutant_vis():
    all_data = []
    with open("output/output.log") as f:
        for line in f:
            all_data.append(VisualModel(line.strip().split(":", 1)[1]))

    wt_points = []
    mt_points = []
    for a in all_data:
        wt_points.append(a.wt_recovery)
        mt_points.append(a.mt_recovery)

    wt_points = np.asanyarray(wt_points)
    mt_points = np.asanyarray(mt_points)

    nor_array = mt_points / wt_points

    gs = gridspec.GridSpec(2, 3)
    grid_count = 0
    for m in recovery_points:
        ax = plt.subplot(gs[grid_count])
        ax.hist(nor_array[:, grid_count], alpha=0.5,
                color=primary_colors[grid_count])
        ax.axvline(1, linestyle="--", color="k")
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yscale("log")
        ax.set_title("%s %% recovery" % m)
        grid_count += 1

    plt.show()
