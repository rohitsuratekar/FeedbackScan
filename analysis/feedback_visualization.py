import matplotlib.pyplot as plt

from analysis.analysis_settings import *
from analysis.feedback_multiple import get_recovery_points
from analysis.helper import *


class VisualClass:
    """
    Simple class to hold raw data
    """

    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.data = json.loads(raw_data.split(":", 1)[1])
        self.enzyme_data = self.data["Enzymes"]
        self.fed_para = self.data["fed_para"]
        self.min_pi4p = self.data["min_pi4p"]
        self.pi4p_timings = self.data["pi4p_timings"]
        self.pip2_timings = self.data["pip2_timings"]
        self.ss_dif_pi4p = self.data["ss_dif_pi4p"]
        self.ss_dif_pip2 = self.data["ss_dif_pip2"]

    @property
    def enzymes(self):
        return convert_to_enzyme(self.enzyme_data)

    def get_pip2_ratio(self, nf_data: dict):
        print(np.asanyarray(self.pip2_timings))
        print(np.asanyarray(nf_data["pip2_timings"]))
        print(np.asanyarray(self.pip2_timings) / np.asanyarray(
            nf_data["pip2_timings"]))
        return np.asanyarray(self.pip2_timings) / np.asanyarray(
            nf_data["pip2_timings"])


def get_para_sets(filename: str) -> list:
    """
    Extracts parameters from the file and converts into VisualClass
    :param filename: Name of input file
    :return: list of VisualClass elements
    """
    all_sets = []
    with open(filename) as f:
        for line in f:
            all_sets.append(VisualClass(line.strip()))

    return all_sets


def convert_into_data(recovery_array, ss_lipids) -> dict:
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

    return {
        "pip2_timings": pip2_timings,
        "pi4p_timings": pi4p_timings,
        "min_pi4p": pi4p_depletion,
        "ss_dif_pip2": recovery_array[-1][I_PIP2] / ss_lipids[I_PIP2],
        "ss_dif_pi4p": recovery_array[-1][I_PI4P] / ss_lipids[I_PI4P]
    }


def general_analysis(filename: str):
    all_para = get_para_sets(filename)
    nf_recovery = get_recovery_points(all_para[0].enzymes, None)
    nf_data = convert_into_data(nf_recovery, nf_recovery[-1])
    pip2_ratio = []
    for p in all_para:
        para = p  # type:VisualClass
        r = para.get_pip2_ratio(nf_data)
        if all(x < 1000 for x in r):
            pip2_ratio.append(r)

        break

    pip2_ratio = np.asanyarray(pip2_ratio)
    plt.hist(pip2_ratio[:, 4], 100)
    plt.yscale("log")
    plt.show()


def visualize(filename: str):
    general_analysis(filename)
