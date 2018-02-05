from constants.namespace import *


class Enzyme:
    """
    Biological enzymes modified for feedback reaction
    """

    def __init__(self, name: str):
        self.name = name
        self.v = None
        self.k = None
        self.kinetics = None

    @property
    def properties(self) -> dict:
        return {"v": self.v, "k": self.k, "kinetics": self.kinetics,
                "name": self.name}

    @classmethod
    def make(cls, name: str, data: dict):
        """
        Makes Enzyme class from data
        :param name: Name of Enzyme
        :param data: Dictionary of data
        :return: Returns class with properties assigned
        """
        temp = cls(name)
        temp.k = data["k"]
        temp.kinetics = data["kinetics"]
        try:
            temp.v = data["v"]
        except KeyError:
            pass
        return temp

    def react_with(self, substrate: float,
                   feedback_para: dict = None) -> float:
        """
        This method is modified for taking into account feedback
        :param substrate: Substrate on which enzyme is acting
        :param feedback_para: dictionary of feedback parameters
        this dictionary should have following values (Use constants for keys)
        t : type of feedback (1 for positive, 2 for negative)
        h : hill coefficient
        a : multiplying factor factor
        c : carrying capacity
        s : Concentration of component who is giving feedback
        :return: product amount
        """

        if feedback_para is None:
            # Regular reaction
            if substrate is not None:
                if self.kinetics == KINETIC_MASS_ACTION:
                    return self.k * substrate
                elif self.kinetics == KINETIC_MICHAELIS_MENTEN:
                    return (self.v * substrate) / (self.k + substrate)
            else:
                return self.k  # For source
        else:
            h = feedback_para[F_HILL_COEFFICIENT]
            a = feedback_para[F_MULTIPLICATION_FACTOR]
            c = feedback_para[F_CARRYING_CAPACITY]
            feedback_substrate = feedback_para[F_FEEDBACK_SUBSTRATE]
            t = feedback_para[F_TYPE_OF_FEEDBACK]
            e = feedback_para[F_ENZYME]

            reg = 1 + pow((feedback_substrate / c), h)
            fed = 1 + a * pow((feedback_substrate / c), h)
            fed_factor = 1
            if t == FEEDBACK_POSITIVE and e == self.name:
                fed_factor = fed / reg
            elif t == FEEDBACK_NEGATIVE and e == self.name:
                fed_factor = reg / fed

            if substrate is not None:
                if self.kinetics == KINETIC_MASS_ACTION:
                    return self.k * substrate * fed_factor
                elif self.kinetics == KINETIC_MICHAELIS_MENTEN:
                    return (self.v * substrate * fed_factor) / (
                            self.k + substrate)
            else:
                return self.k * fed_factor  # For source

    def stimulate(self, factor):
        if self.name is not E_PLC:
            raise Exception("Only PLC can be stimulated")
        if self.kinetics == KINETIC_MASS_ACTION:
            self.k *= factor
        elif self.kinetics == KINETIC_MICHAELIS_MENTEN:
            self.v *= factor
