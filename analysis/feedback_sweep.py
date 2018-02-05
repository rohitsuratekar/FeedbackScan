"""
Analysis related to sweeping feedback
"""

from itertools import product

from analysis.helper import *
from utils.functions import update_progress
from utils.log import *

ss_end_time = 3000
ss_stimulus_end_time = 10
ss_recovery_time = 1000

initial_time = np.linspace(0, ss_end_time, ss_end_time * 10)
stimulus_time = np.linspace(0, ss_stimulus_end_time, ss_stimulus_end_time * 10)
recovery_time = np.linspace(0, ss_recovery_time, ss_recovery_time * 10)

stimulus_factor = 10

range_hill = [1]
range_feed_type = [FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE]
range_carry = np.linspace(1, 15, 2)
range_multi = np.linspace(1, 15, 2)
range_substrate = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG,
                   I_ERPI]
range_enz = [E_PITP, E_PI4K, E_PIP5K, E_PLC, E_DAGK, E_LAZA, E_PATP, E_CDS,
             E_PIS, E_SINK, E_SOURCE]


def get_ss(system, ini_cond, enzymes, feed_para=None):
    return odeint(get_equations(system), ini_cond,
                  initial_time, args=(enzymes, feed_para))[-1]


def get_stimulus(system, ini_cond, enzymes, feed_para=None):
    enzymes[E_PLC].stimulate(stimulus_factor)
    v = odeint(get_equations(system), ini_cond, stimulus_time,
               args=(enzymes, feed_para))
    enzymes[E_PLC].stimulate(1 / stimulus_factor)
    return v


def get_recovery(system, ini_cond, enzymes, feed_para=None):
    return odeint(get_equations(system), ini_cond, recovery_time,
                  args=(enzymes, feed_para))


def get_difference(nor_ss_pip2, nor_recovery, fed_ss_pip2, fed_recovery):
    nor_array = np.asarray(nor_recovery) / nor_ss_pip2
    fed_array = np.asarray(fed_recovery) / fed_ss_pip2
    diff = nor_array - fed_array

    val_to_check = [0.3, 0.5, 0.7, 0.9]
    normal = []
    feedback = []

    for k in val_to_check:
        normal.append(recovery_time[[x > k for x in nor_array].index(True)])
        feedback.append(recovery_time[[x > k for x in fed_array].index(True)])

    return {"sum": sum(diff), "val": val_to_check, "nor": normal,
            "feed": feedback}


def scan_for_feedback(enzymes: dict, system: str, total_lipid: float):
    initial_conditions = get_random_concentrations(total_lipid, system)
    progress_counter = 0
    total_size = len(list(product(
        *[range_hill, range_carry, range_multi, range_feed_type,
          range_substrate, range_enz])))
    for hill, carry, multi, fed_type, sub_ind, enz in product(
            *[range_hill, range_carry, range_multi, range_feed_type,
              range_substrate, range_enz]):
        update_progress(progress_counter / total_size)
        progress_counter += 1

        # Without Feedback
        initial_ss = get_ss(system, initial_conditions, enzymes)
        st_phase = get_stimulus(system, initial_ss, enzymes)
        re_phase = get_recovery(system, st_phase[-1], enzymes)

        fed_para = {
            F_HILL_COEFFICIENT: hill,
            F_FEED_SUBSTRATE_INDEX: sub_ind,
            F_TYPE_OF_FEEDBACK: fed_type,
            F_CARRYING_CAPACITY: carry,
            F_MULTIPLICATION_FACTOR: multi,
            F_ENZYME: enz
        }

        # With Feedback
        initial_fed_ss = get_ss(system, initial_conditions, enzymes, fed_para)
        st_fed_phase = get_stimulus(system, initial_fed_ss, enzymes, fed_para)
        re_fed_phase = get_recovery(system, st_fed_phase[-1], enzymes,
                                    fed_para)

        dif = get_difference(initial_ss[I_PIP2], re_phase[:, I_PIP2],
                             initial_fed_ss[I_PIP2], re_fed_phase[:, I_PIP2])

        fed_para.pop(F_FEEDBACK_SUBSTRATE)
        data = {
            "Enzymes": {e: enzymes[e].properties for e in enzymes},
            "fed_para": fed_para,
            "Diff": dif
        }
        OUTPUT.info(json.dumps(data, sort_keys=True))


def do_feedback_sweep(filename: str, system: str, total_lipid=1):
    log_data = {
        "UID": CURRENT_JOB,
        "system": system,
        "TotalLipid": total_lipid,
        "Analysis": "Feedback Scan",
        "version": "3.0"}
    LOG.info(json.dumps(log_data, sort_keys=True))
    all_sets = []
    with open(filename) as f:
        for line in f:
            all_sets.append(convert_to_enzyme(extract_enz_from_log(line)))

    for s in all_sets:
        scan_for_feedback(s, system, total_lipid)
