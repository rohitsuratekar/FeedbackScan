"""
All feedback related scaling functions can go in this file
Related visualization will also go in the same file


IMPORTANT: We removed all feedback parameter which were not giving steady
state close to the steady state without feedback
"""

from itertools import product

from analysis.analysis_settings import *
from analysis.helper import *
from utils.functions import update_progress
from utils.log import *

recovery_time = np.linspace(0, 100, 3000)


def get_scaled_enzymes(filename: str, system: str) -> dict:
    with open(filename) as f:
        enzymes = convert_to_enzyme(extract_enz_from_log(f.read()))
    init_con = get_random_concentrations(200, system)
    initial_time = np.linspace(0, 2000, 5000)
    ss = odeint(get_equations(system), init_con, initial_time,
                args=(enzymes, None))[-1]
    plc_base = enzymes[E_PLC].v
    for e in enzymes:
        if e != E_SOURCE:
            enzymes[e].k = enzymes[e].k / sum(ss)
            enzymes[e].v = enzymes[e].v / plc_base
        else:
            enzymes[e].k = enzymes[e].k / plc_base
    return enzymes


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
                pip2_timings.append(recovery_time[1])
            else:
                pip2_timings.append(i)
        except ValueError:
            pip2_timings.append(-1989)

    for point in RECOVERY_POINTS:
        req_con = ss_lipids[I_PI4P] * point / 100
        try:
            i = recovery_time[[x > req_con for x in ar_pi4p].index(True)]
            if i == 0.0:
                pi4p_timings.append(recovery_time[1])
            else:
                pi4p_timings.append(i)
        except ValueError:
            pi4p_timings.append(-1989)

    pip2_diff = recovery_array[-1][I_PIP2] / ss_lipids[I_PIP2]
    pi4p_diff = recovery_array[-1][I_PI4P] / ss_lipids[I_PI4P]

    data = {
        "Enzymes": {e: enzymes[e].properties for e in enzymes},
        "fed_para": feed_para,
        "pip2_timings": pip2_timings,
        "pi4p_timings": pi4p_timings,
        "min_pi4p": pi4p_depletion,
        "ss_dif_pip2": pip2_diff,
        "ss_dif_pi4p": pi4p_diff
    }
    OUTPUT.info(json.dumps(data, sort_keys=True))


def scan_single_feedback(filename: str, system: str):
    # Log the analysis details
    log_data = {
        "UID": CURRENT_JOB,
        "system": system,
        "Analysis": "Feedback Scan with Scaling",
        "sub_version": "2.0",
        "number_of_feedback": 1,
        "recovery_points": RECOVERY_POINTS,
        "depletion_percentage": PERCENTAGE_DEPLETION,
        "version": "3.0"}
    LOG.info(json.dumps(log_data, sort_keys=True))

    total_size = len(list(product(
        *[RANGE_HILL_COEFFICIENT, RANGE_CARRY, RANGE_MULTIPLICATION_FACTOR,
          RANGE_FEED_TYPE, RANGE_SUBSTRATE, RANGE_ENZYMES])))
    progress_counter = 0
    enzymes = get_scaled_enzymes(filename, system)
    init_con = get_random_concentrations(1, system)
    init_time = np.linspace(0, 10000, 10000)

    no_feed_ss = odeint(get_equations(system), init_con, init_time,
                        args=(enzymes, None))[-1]

    for hill, carry, multi, fed_type, sub_ind, enz in product(
            *[RANGE_HILL_COEFFICIENT, RANGE_CARRY, RANGE_MULTIPLICATION_FACTOR,
              RANGE_FEED_TYPE, RANGE_SUBSTRATE, RANGE_ENZYMES]):
        update_progress(progress_counter / total_size)
        progress_counter += 1
        feed_para = {
            enz: {
                F_HILL_COEFFICIENT: hill,
                F_FEED_SUBSTRATE_INDEX: sub_ind,
                F_TYPE_OF_FEEDBACK: fed_type,
                F_CARRYING_CAPACITY: carry,
                F_MULTIPLICATION_FACTOR: multi,
            }
        }

        # Correct enzyme Vmax for feedback
        # This ensures same steady state with feedback
        reg = 1 + pow((no_feed_ss[sub_ind] / carry), hill)
        fed = 1 + multi * pow((no_feed_ss[sub_ind] / carry), hill)
        fed_factor = 1

        # Following multiplication should be opposite to feedback.
        if fed_type == FEEDBACK_NEGATIVE:
            fed_factor = fed / reg
        elif fed_type == FEEDBACK_POSITIVE:
            fed_factor = reg / fed

        enzymes[enz].v *= fed_factor
        init_ss = odeint(get_equations(system), init_con, init_time,
                         args=(enzymes, feed_para))[-1]

        # Roughly check if steady state values are same as without feedback
        if round(sum(no_feed_ss / init_ss)) == 8:
            # Give stimulus
            stim = give_stimulus(init_ss, PERCENTAGE_DEPLETION)
            recovery = odeint(get_equations(system), stim, recovery_time,
                              args=(enzymes, feed_para))
            # Change enzyme values back to original
            enzymes[enz].v /= fed_factor
            save_data(enzymes, feed_para, recovery, init_ss)
        else:
            # Change enzyme values back to original
            enzymes[enz].v /= fed_factor
