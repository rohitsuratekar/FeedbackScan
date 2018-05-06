import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from analysis.analysis_settings import *
from analysis.helper import *


def get_enzymes(filename: str) -> dict:
    with open(filename) as f:
        return convert_to_enzyme(extract_enz_from_log(f.read()))


def get_scaled_enzymes(filename: str, system: str) -> dict:
    enzymes = get_enzymes(filename)
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


def get_real_value_enzymes(filename: str, system: str, total_pi: float):
    enzymes = get_enzymes(filename)
    init_con = get_random_concentrations(200, system)
    initial_time = np.linspace(0, 2000, 5000)
    ss = odeint(get_equations(system), init_con, initial_time,
                args=(enzymes, None))[-1]
    plc_base = enzymes[E_PLC].v

    total_lipid = 1.2767 * total_pi

    for e in enzymes:
        if e != E_SOURCE:
            enzymes[e].k = enzymes[e].k * total_lipid / sum(ss)
            enzymes[e].v = enzymes[e].v / plc_base
        else:
            enzymes[e].k = enzymes[e].k / plc_base
    return enzymes


def plot_without_feedback(filename: str, system: str):
    enzymes = get_real_value_enzymes(filename, system, 3.58)
    init_con = get_random_concentrations(1, system)
    initial_time = np.linspace(0, 200, 3000)
    ss = odeint(get_equations(system), init_con, initial_time,
                args=(enzymes, None))[-1]

    # Plot Buffer Time
    buffer_time = np.linspace(0, 2, 50)
    buffer = odeint(get_equations(system), ss, buffer_time,
                    args=(enzymes, None))

    # Do stimulation swap
    stim = give_stimulus(buffer[-1], PERCENTAGE_DEPLETION)

    # Recovery
    recovery_time = np.linspace(0, 10, 1000)
    recovery = odeint(get_equations(system), stim, recovery_time,
                      args=(enzymes, None))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(x - buffer_time[-1]))
    ax.xaxis.set_major_formatter(ticks_x)

    for i in [I_PIP2]:
        ax.plot(buffer_time, buffer[:, i], label=get_lipid_from_index(i),
                color=COLORS_PRIMARY[i])
        ax.plot(recovery_time + buffer_time[-1], recovery[:, i],
                color=COLORS_PRIMARY[i])

    ax.legend(loc=0)
    ax.axvline(buffer_time[-1], linestyle="--", color="k")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("concentration (pmoles)")
    ax.text(buffer_time[-1], ax.get_ylim()[0] + sum(ax.get_ylim()) / 2,
            'Light Stimulation', ha='center', va='center',
            rotation='vertical', backgroundcolor='white')
    plt.savefig("without_feedback.png", format='png', dpi=300,
                bbox_inches='tight')
    plt.show()


def plot(filename: str, system: str):
    plot_without_feedback(filename, system)
