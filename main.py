from analysis.feedback_multiple import multi_feedback
from analysis.feedback_visualization import visualize
from analysis.test_feedback import check


def s():
    multi_feedback(1, "test.txt")


def v():
    visualize("output/single_output.log")


def c():
    check("test.txt")


c()
