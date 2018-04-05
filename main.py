from analysis.feedback_mutants import do_mutant_analysis, mutant_vis
from analysis.feedback_scaling import do_scaling, visualize


def s():
    do_scaling("test.txt")


def v():
    visualize()


def d():
    do_mutant_analysis("output/sample.log")


def mv():
    mutant_vis()


mv()
