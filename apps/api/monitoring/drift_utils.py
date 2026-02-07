import pandas as pd
import numpy as np

EPS = 1e-6

def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """
    PSI for categorical variables.
    expected: baseline distribution
    actual: current distribution
    """
    exp_dist = expected.value_counts(normalize=True)
    act_dist = actual.value_counts(normalize=True)

    categories = set(exp_dist.index).union(set(act_dist.index))

    psi = 0.0
    for c in categories:
        e = exp_dist.get(c, EPS)
        a = act_dist.get(c, EPS)
        psi += (a - e) * np.log(a / e)

    return float(psi)


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "drift"
