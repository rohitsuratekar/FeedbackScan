"""
Helper methods used in all analysis
"""
import json

import numpy as np
from scipy.integrate import odeint

from models.biology import *
from models.systems.open2 import get_equations as open2


def get_parameter_set(filename) -> list:
    """
    Converts log file parameter to
    :param filename:
    :return:
    """
    parameters = []
    with open(filename, "r") as f:
        for line in f:
            parameters.append(extract_enz_from_log(line))
    return parameters


def get_equations(system: str):
    """
    Returns equation of specific system
    :param system: topology or known model
    :return: set of equation function
    """
    if system == S_OPEN_2:
        return open2
    else:
        raise Exception("No such system found :%s" % system)


def extract_enz_from_log(log_text: str):
    return json.loads(log_text.split(":", 1)[1])["Enzymes"]


def get_concentration_profile(system: str, initial_condition, parameters: dict,
                              ode_time: int, slices: int):
    """
    Solves ODE and returns output array
    :param system: topology or known model
    :param initial_condition: array of initial condition concentrations
    :param parameters: dict of parameter set
    :param ode_time: end time of ODE, initial time will be always 0
    :param slices: number of points to integrate
    :return: Output array
    """

    time = np.linspace(0, ode_time, slices)
    output = odeint(get_equations(system), initial_condition, time,
                    args=(parameters, 0))
    return output


def get_random_concentrations(total: float, system: str) -> list:
    """
    Creates random lipid distribution who's total is equal to given value
    :param system: Type of Cycle
    :param total: total amount of lipid
    :return: randomly distributing lipids
    """

    no_of_lipids = 8
    # if system == S_CLASSICAL_REVERSIBLE:
    #  We need extra concentration for IP3
    #     no_of_lipids = 9

    all_ratios = np.random.uniform(0, 1, no_of_lipids)
    all_concentration = []
    for i in range(no_of_lipids):
        all_concentration.append(all_ratios[i] * total / sum(all_ratios))

    return all_concentration


def convert_to_enzyme(data) -> dict:
    """
    Makes new random enzyme based on kinetics

    :param data: Dict of data properties
    :return: enzyme list
    """
    pitp = Enzyme.make(E_PITP, data[E_PITP])
    pi4k = Enzyme.make(E_PI4K, data[E_PI4K])
    pip5k = Enzyme.make(E_PIP5K, data[E_PIP5K])
    plc = Enzyme.make(E_PLC, data[E_PLC])
    dagk = Enzyme.make(E_DAGK, data[E_DAGK])
    laza = Enzyme.make(E_LAZA, data[E_LAZA])
    patp = Enzyme.make(E_PATP, data[E_PATP])
    cds = Enzyme.make(E_CDS, data[E_CDS])
    pis = Enzyme.make(E_PIS, data[E_PIS])
    sink = Enzyme.make(E_SINK, data[E_SINK])
    source = Enzyme.make(E_SOURCE, data[E_SOURCE])
    p4tase = Enzyme.make(E_P4TASE, data[E_P4TASE])
    p5tase = Enzyme.make(E_P5TASE, data[E_P5TASE])
    ip3tase = Enzyme.make(E_IP3_PTASE, data[E_IP3_PTASE])
    return {x.name: x for x in
            [pitp, pi4k, pip5k, plc, dagk, laza, patp, cds, pis, sink, source,
             p4tase, p5tase, ip3tase]}
