from analysis.feedback_multiple import multi_feedback
from analysis.feedback_visualization import visualize


def s():
    multi_feedback(2, "test.txt")


def v():
    visualize("output/output.log")


v()
