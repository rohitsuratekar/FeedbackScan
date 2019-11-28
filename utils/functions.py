"""
General utility functions used in log of various scripts.
"""

import random
import string
import sys


def clamp(number, min_value, max_value):
    """
    clamp() : Clamps value between minimum and maximum
    :param number: Number
    :param min_value: Minimum allowed
    :param max_value: Maximum allowed
    :return: min_value if value is less than min, max_value if greater or
    same value

   """
    return sorted((number, min_value, max_value))[1]


def update_progress(progress, message=""):
    """
    update_progress() : Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%

    Code from : http://stackoverflow.com/questions/3160699/python-progress-bar

    """
    bar_length = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = "\rPercent: [{0}] {1}% {2} {3}".format(
        "#" * block + "-" * (bar_length - block), round(progress * 100, 2),
        status, message)
    sys.stdout.write(text)
    sys.stdout.flush()


def get_uid(n=10):
    """
    Creates random UID
    :return: unique string of length n (default) 10
    """
    char_set = string.ascii_uppercase + string.digits
    return ''.join(random.sample(char_set * n, n))
