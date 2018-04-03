"""
All feedback related scaling functions can go in this file
"""

import matplotlib.pylab as plt

from analysis.helper import *

# Initial setup
system = S_OPEN_2

ss_end_time = 4000
ss_recovery_time = 1000

depletion_percentage = 15
desired_recovery_percentage = 90

initial_time = np.linspace(0, ss_end_time, ss_end_time * 10)
recovery_time = np.linspace(0, ss_recovery_time, ss_recovery_time * 10)


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
    ra = recovery_time[[x > req_conc for x in ar].index(True)]
    print(ra)
    plt.plot(recovery_time, con_array[:, I_PIP2])
    plt.xlim(0, 10)
    plt.show()


def do_scaling(filename: str):
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
        re_time = get_recovery_time(stim, enzymes)
        get_desired_recovery_time(re_time, ss_pip2)
