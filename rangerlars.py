from .lookahead import *
from .ralamb import *


def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    radam = Ralamb(params, *args, **kwargs)
    return Lookahead(radam, alpha, k)
