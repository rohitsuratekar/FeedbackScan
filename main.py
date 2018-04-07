from analysis.feedback_multiple import multi_feedback
from analysis.feedback_mutants import do_mutant_analysis, mutant_vis
from analysis.feedback_scaling import do_scaling, visualize, sort_for_mutant


def s():
    do_scaling("test.txt")


def v():
    visualize()


def d():
    do_mutant_analysis("output/mutant_sorted.log")


def st():
    sort_for_mutant()


def mv():
    mutant_vis()


multi_feedback(2, "test.txt")
