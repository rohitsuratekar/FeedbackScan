"""
Copyright © 2017  Rohit Suratekar
Code from this file is released under MIT Licence 2017.
use "Log" for logging information and "OUTPUT" for saving information
"""
import logging
import os

from settings import *
from utils.functions import get_uid

# Creates UID for current job
CURRENT_JOB = get_uid()
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


class AppFilter(logging.Filter):
    """
    Adds custom field in log file
    """

    def filter(self, record):
        record.uid = CURRENT_JOB
        return True


LOG = logging.getLogger('log')
LOG.setLevel(logging.INFO)
LOG.addFilter(AppFilter())
if STORE_SCRIPT_LOG:
    log_file = logging.FileHandler(
        OUTPUT_FOLDER + "/" + NAME_OF_SCRIPT_LOG_FILE)
    log_file.setFormatter(
        logging.Formatter('%(uid)s %(asctime)s %(filename)s : %(message)s'))
    LOG.addHandler(log_file)
if PRINT_TO_CONSOLE:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(uid)s %(asctime)s %(filename)s : %(message)s')
    console.setFormatter(formatter)
    LOG.addHandler(console)

OUTPUT = logging.getLogger('output')
OUTPUT.setLevel(logging.INFO)
OUTPUT.addFilter(AppFilter())
output_file = logging.FileHandler(OUTPUT_FOLDER + "/" + NAME_OF_OUTPUT_FILE)
output_file.setFormatter(logging.Formatter('%(uid)s: %(message)s'))
OUTPUT.addHandler(output_file)
