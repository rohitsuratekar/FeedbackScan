"""
Generalized framework to analyse the multiple feedbacks
"""

from itertools import product, combinations

from analysis.helper import *
from utils.functions import update_progress
from utils.log import LOG, OUTPUT, CURRENT_JOB

SYSTEM = S_OPEN_2
PERCENTAGE_DEPLETION = 85

# First recovery point is 16 which is immediately after 15% depletion
# 76.5 is 90% of whatever is left after 15% depletion
RECOVERY_POINTS = [20, 30, 50, 76.5, 90, 100]  # Get data at these points

# Initialization and some base constants and their ranges
RANGE_HILL_COEFFICIENT = [0.5, 1, 2]
RANGE_MULTIPLICATION_FACTOR = np.linspace(1, 10, 10)
RANGE_CARRY = np.linspace(0.1, 10, 10)
RANGE_FEED_TYPE = [FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE]
RANGE_SUBSTRATE = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG,
                   I_ERPI]
RANGE_ENZYMES = [E_PITP, E_PI4K, E_PIP5K, E_PLC, E_DAGK, E_LAZA, E_PATP, E_CDS,
                 E_PIS, E_SINK, E_SOURCE]

# Timings
recovery_time = np.linspace(0, 100, 2000)


def get_parameter_set(filename: str):
    """
    Extracts parameter set from file and normalize it

    :param filename: File name with Single Parameter set in standard format
    :return: Dictionary of enzymes
    """
    with open(filename) as f:
        enzymes = convert_to_enzyme(extract_enz_from_log(f.read()))

    init_con = get_random_concentrations(200, SYSTEM)
    # Time to get approx  steady state
    initial_time = np.linspace(0, 4000, 10000)
    ss = odeint(get_equations(SYSTEM), init_con, initial_time, args=(enzymes,
                                                                     None))[-1]
    # Normalize all parameter values
    plc_base = enzymes[E_PLC].v
    for e in enzymes:
        if e != E_SOURCE:
            enzymes[e].k = enzymes[e].k / sum(ss)
            enzymes[e].v = enzymes[e].v / plc_base
        else:
            enzymes[e].k = enzymes[e].k / plc_base
    return enzymes


def get_recovery_points(enzymes, feed_factors):
    init_con = get_random_concentrations(1, SYSTEM)
    initial_time = np.linspace(0, 1000, 3000)
    pre_sim_ss = odeint(get_equations(SYSTEM), init_con, initial_time,
                        args=(enzymes, feed_factors))[-1]

    # Stimulation
    sim_ss = [x for x in pre_sim_ss]
    amount = pre_sim_ss[I_PIP2] * PERCENTAGE_DEPLETION / 100
    sim_ss[I_DAG] = sim_ss[I_DAG] + sim_ss[I_PIP2] - amount
    sim_ss[I_PIP2] = amount

    # Recovery
    return odeint(get_equations(SYSTEM), sim_ss, recovery_time,
                  args=(enzymes, feed_factors))


def save_data(enzymes, feed_para, recovery_array, ss_lipids):
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
            pip2_timings.append(recovery_time[-1])

    for point in RECOVERY_POINTS:
        req_con = ss_lipids[I_PI4P] * point / 100
        try:
            i = recovery_time[[x > req_con for x in ar_pi4p].index(True)]
            if i == 0.0:
                pi4p_timings.append(ar_pi4p[1])
            else:
                pi4p_timings.append(i)
        except ValueError:
            pi4p_timings.append(recovery_time[-1])

    data = {
        "Enzymes": {e: enzymes[e].properties for e in enzymes},
        "fed_para": feed_para,
        "pip2_timings": pip2_timings,
        "pi4p_timings": pi4p_timings,
        "min_pi4p": pi4p_depletion,
        "ss_dif_pip2": recovery_array[-1][I_PIP2] / ss_lipids[I_PIP2],
        "ss_dif_pi4p": recovery_array[-1][I_PI4P] / ss_lipids[I_PI4P]
    }
    OUTPUT.info(json.dumps(data, sort_keys=True))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def multi_feedback(no_of_feedback: int, filename: str):
    # Log the analysis details
    log_data = {
        "UID": CURRENT_JOB,
        "system": SYSTEM,
        "Analysis": "Multi-Feedback Scan with Scaling",
        "recovery_points": RECOVERY_POINTS,
        "number_of_feedback": no_of_feedback,
        "depletion_percentage": PERCENTAGE_DEPLETION,
        "version": "3.0"}
    LOG.info(json.dumps(log_data, sort_keys=True))

    enzymes = get_parameter_set(filename)

    # Get steady states of lipids without feedback
    nf_ss = get_recovery_points(enzymes, None)[-1]

    # Create combination of Enzyme-Substrate pairs
    enz_sub_comb = []
    for enzyme_set in combinations(RANGE_ENZYMES, no_of_feedback):
        for substrate_set in product(RANGE_SUBSTRATE, repeat=no_of_feedback):
            enz_sub_comb.append([enzyme_set, substrate_set])

    progress_counter = 0
    # Start main iteration
    for c in enz_sub_comb:
        update_progress(progress_counter / len(enz_sub_comb))

        # Initiate new dictionary for feedback parameters
        fed_para = {}

        for i in range(len(c[0])):
            fed_para[c[0][i]] = {
                F_FEED_SUBSTRATE_INDEX: c[1][i],
            }

        # Create combinations of feedback
        for fc in product(
                *[RANGE_HILL_COEFFICIENT, RANGE_CARRY,
                  RANGE_MULTIPLICATION_FACTOR, RANGE_FEED_TYPE],
                repeat=no_of_feedback):
            feed_comb = []
            for a in chunks(fc, 4):
                feed_comb.append(a)

            # Python 3.6 + maintains dictionary order so we cn ideally
            # assign all variables by index
            ind = 0
            for e in fed_para:
                fed_para[e][F_HILL_COEFFICIENT] = feed_comb[ind][0]
                fed_para[e][F_CARRYING_CAPACITY] = feed_comb[ind][1]
                fed_para[e][F_MULTIPLICATION_FACTOR] = feed_comb[ind][2]
                fed_para[e][F_TYPE_OF_FEEDBACK] = feed_comb[ind][3]

            f_fed_wt = get_recovery_points(enzymes, fed_para)
            save_data(enzymes, fed_para, f_fed_wt, nf_ss)

        progress_counter += 1
