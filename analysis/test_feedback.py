import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from analysis.analysis_settings import *
from analysis.helper import *

SYSTEM = S_OPEN_2
BUFFER_TIME = 2

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


def get_scaled_parameter(filename: str):
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


feed = {'pip5k': {'a': 7.196856730011519, 'c': 0.13894954943731375, 'h': 0.5,
                  'i': 0, 's': 0.7874841424458127, 't': 2}}


def check(filename: str):
    feed_para = {}
    for f in feed:
        feed_para = {f: {
            F_HILL_COEFFICIENT: feed[f][F_HILL_COEFFICIENT],
            F_CARRYING_CAPACITY: feed[f][F_CARRYING_CAPACITY],
            F_MULTIPLICATION_FACTOR: feed[f][F_MULTIPLICATION_FACTOR],
            F_TYPE_OF_FEEDBACK: feed[f][F_TYPE_OF_FEEDBACK],
            F_FEED_SUBSTRATE_INDEX: feed[f][F_FEED_SUBSTRATE_INDEX]
        }}
    # feed_para = None
    enzymes = get_scaled_parameter(filename)

    init_con = get_random_concentrations(1, SYSTEM)
    initial_time = np.linspace(0, 1000, 3000)
    pre_sim_ss = odeint(get_equations(SYSTEM), init_con, initial_time,
                        args=(enzymes, feed_para))[-1]

    buffer_time = np.linspace(0, BUFFER_TIME, 30)
    buffer_con = odeint(get_equations(SYSTEM), pre_sim_ss, buffer_time,
                        args=(enzymes, feed_para))

    # Stimulation
    sim_ss = [x for x in pre_sim_ss]
    amount = pre_sim_ss[I_PIP2] * (1 - 0.85)
    sim_ss[I_DAG] = sim_ss[I_DAG] + sim_ss[I_PIP2] - amount
    sim_ss[I_PIP2] = amount

    rec = odeint(get_equations(SYSTEM), sim_ss, recovery_time,
                 args=(enzymes, feed_para))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(x - BUFFER_TIME))
    ax.xaxis.set_major_formatter(ticks_x)

    def lipid_plot(index):
        ax.plot(buffer_time, buffer_con[:, index], color=primary_colors[index])
        ax.plot(recovery_time + BUFFER_TIME, rec[:, index],
                color=primary_colors[index], label=get_lipid_from_index(index))

    lipid_plot(I_PIP2)
    lipid_plot(I_PI4P)
    ax.axvline(BUFFER_TIME, linestyle="--", color="k")
    ax.set_xlim(1, 5)
    ax.set_xlabel("time (arbitrary units)")
    ax.set_ylabel("amount (normalized to total lipid content)")
    ax.legend(loc=0)
    plt.show()
