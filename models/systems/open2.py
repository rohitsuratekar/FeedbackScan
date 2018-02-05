from constants.namespace import *
from models.biology import Enzyme


def get_equations(concentrations: list, time: tuple, *args) -> list:
    assert len(concentrations) == 8, "You should provide all concentrations"
    enzyme_list = args[0]  # type: dict
    feed_para = args[1]
    if feed_para is not None:
        feed_para[F_FEEDBACK_SUBSTRATE] = concentrations[
            feed_para[F_FEED_SUBSTRATE_INDEX]]

    pitp = enzyme_list.get(E_PITP)  # type: Enzyme
    pip5k = enzyme_list.get(E_PIP5K)  # type: Enzyme
    plc = enzyme_list.get(E_PLC)  # type: Enzyme
    pi4k = enzyme_list.get(E_PI4K)  # type: Enzyme
    dagk = enzyme_list.get(E_DAGK)  # type: Enzyme
    laza = enzyme_list.get(E_LAZA)  # type: Enzyme
    patp = enzyme_list.get(E_PATP)  # type: Enzyme
    cds = enzyme_list.get(E_CDS)  # type: Enzyme
    pis = enzyme_list.get(E_PIS)  # type: Enzyme
    sink = enzyme_list.get(E_SINK)  # type: Enzyme
    source = enzyme_list.get(E_SOURCE)  # type: Enzyme

    pmpi, pi4p, pip2, dag, pmpa, erpa, cdpdag, erpi = concentrations

    d_pmpi = pitp.react_with(erpi, feed_para) - pi4k.react_with(pmpi,
                                                                feed_para)
    d_pi4p = pi4k.react_with(pmpi, feed_para) - pip5k.react_with(pi4p,
                                                                 feed_para)
    d_pip2 = pip5k.react_with(pi4p, feed_para) - plc.react_with(pip2,
                                                                feed_para)
    d_dag = plc.react_with(pip2, feed_para) - dagk.react_with(dag, feed_para) \
            + laza.react_with(pmpa, feed_para) - sink.react_with(dag,
                                                                 feed_para)
    d_pmpa = dagk.react_with(dag, feed_para) - laza.react_with(pmpa, feed_para
                                                               ) - patp.react_with(
        pmpa, feed_para)
    d_erpa = patp.react_with(pmpa, feed_para) - cds.react_with(erpa,
                                                               feed_para) + \
             source.react_with(
                 None, feed_para)  # equal to k value of source
    d_cdpdag = cds.react_with(erpa, feed_para) - pis.react_with(cdpdag,
                                                                feed_para)
    d_erpi = pis.react_with(cdpdag, feed_para) - pitp.react_with(erpi,
                                                                 feed_para)

    return [d_pmpi, d_pi4p, d_pip2, d_dag, d_pmpa, d_erpa, d_cdpdag, d_erpi]
