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


def check(filename: str):
    feed_para = {E_PITP: {
        F_HILL_COEFFICIENT: 1,
        F_CARRYING_CAPACITY: 0.1,
        F_MULTIPLICATION_FACTOR: 6.3095,
        F_TYPE_OF_FEEDBACK: 2,
        F_FEED_SUBSTRATE_INDEX: 2
    }}
    #feed_para = None
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
                color=primary_colors[index])

    lipid_plot(I_PIP2)
    ax.axvline(BUFFER_TIME, linestyle="--", color="k")
    ax.set_xlim(1, 5)
    plt.show()
