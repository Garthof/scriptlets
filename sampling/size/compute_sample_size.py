"""
Based on Shanti R Rao and Potluri M Rao, "Sample Size Calculator",
Raosoft Inc., 2009, http://www.raosoft.com/samplesize.html
"""

import math

def prob_critical_normal(p):
    """
    @param p The confidence level
    This implementation has been adapted from Numerical Recipes in Fortran
    """
    pn = [
            0
            , -0.322232431088
            , -1.0
            , -0.342242088547
            , -0.0204231210245
            , -0.453642210148e-4
        ]

    qn = [
            0
            , 0.0993484626060
            , 0.588581570495
            , 0.531103462366
            , 0.103537752850
            , 0.38560700634e-2
        ]

    pr = 0.5 - p / 2.       # One side significance

    if pr <= 1.e-8:
        return 6

    if pr == 0.5:
        return 0

    y = math.sqrt(math.log(1. / (pr * pr)))
    r1 = pn[5]
    r2 = qn[5]

    for i in xrange(4, 0, -1):
        r1 = r1 * y + pn[i]
        r2 = r2 * y + qn[i]

    return y + r1 / r2


def sample_size(margin=5, confidence=95, response=50, population=20000):
    """
    Returns the sample size
    @param margin Amount of error that can be tolerated
    @param confidence Amount of uncertainty that can be tolerated
    @param response Expected result
    @param population Size of the population
    """
    pcn = prob_critical_normal(confidence / 100.)
    d1 = pcn * pcn * response * (100.0 - response)
    d2 = (population - 1.0) * (margin * margin) + d1

    if d2 > 0.:
        return math.ceil(population * d1 / d2)
    else:
        return 0.


def error_margin(sample=100, confidence=95, response=50, population=20000):
    """
    Returns the error margin
    @param sample Size of the sample
    @param confidence Amount of uncertainty that can be tolerated
    @param response Expected result
    @param population Size of the population
    """
    pcn = prob_critical_normal(confidence / 100.)
    d1 = pcn * pcn * response * (100. - response)
    d2 = d1 * (population - sample) / (sample * (population - 1.))

    if d2 > 0.:
        return math.sqrt(d2)
    else:
        return 0.
