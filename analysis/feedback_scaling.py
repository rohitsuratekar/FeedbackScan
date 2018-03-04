"""
All feedback related scaling functions can go in this file
"""

from analysis.helper import *

ss_end_time = 4000
ss_stimulus_end_time = 5
ss_recovery_time = 1000

depletion_percentage = 15

initial_time = np.linspace(0, ss_end_time, ss_end_time * 10)
stimulus_time = np.linspace(0, ss_stimulus_end_time, ss_stimulus_end_time)
recovery_time = np.linspace(0, ss_recovery_time, ss_recovery_time * 10)


def get_ss(system, ini_cond, enzymes, feed_para=None):
    return odeint(get_equations(system), ini_cond,
                  initial_time, args=(enzymes, feed_para))[-1]


def give_stimulus(ini_cond):
    amount = ini_cond[I_PIP2] * depletion_percentage / 100
    ini_cond[I_DAG] = ini_cond[I_DAG] + ini_cond[I_PIP2] - amount
    ini_cond[I_PIP2] = amount
    return ini_cond


def get_parameter(data: str):
    return convert_to_enzyme(extract_enz_from_log(data))


def check_recovery_time():
    pass


def do_scaling(filename: str):
    # Initial setup
    system = S_OPEN_2

    with open(filename) as f:
        enzymes = get_parameter(f.read())
        init_conc = get_random_concentrations(200, system)
        starting_ss = get_ss(system, init_conc, enzymes)
        plc_base = enzymes[E_PLC].v
        for e in enzymes:
            if e != E_SOURCE:
                enzymes[e].k = enzymes[e].k / sum(starting_ss)
                enzymes[e].v = enzymes[e].v / plc_base
            else:
                enzymes[e].k = enzymes[e].k / plc_base
            if e == E_PLC:
                print(enzymes[e].v)

        print(sum(get_ss(system, init_conc, enzymes)))
