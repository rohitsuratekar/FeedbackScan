from analysis.feedback_scaling import scan_single_feedback
from analysis.feedback_visualize import visualize
from constants.namespace import S_OPEN_2
from test import plot, check_patp

CURRENT_FILE = "best_para.txt"


def scan_single():
    scan_single_feedback(CURRENT_FILE, S_OPEN_2)


def test():
    plot(CURRENT_FILE, S_OPEN_2)


def vis():
    visualize("output/output.log", S_OPEN_2)


if __name__ == "__main__":
    #check_patp(CURRENT_FILE, S_OPEN_2)
    vis()
