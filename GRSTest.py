#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Perform the Gibbons, Ross and Shanken (1989) Test
    - Gibbons, M., Ross, S., and Shanken, J., 1989, A Test of the Efficiency of A Given Portfolio,
      Econometrica 57, 1121-1152.
"""

import numpy as np
from scipy.stats import f


def grs_test(resid: np.ndarray, alpha: np.ndarray, factors: np.ndarray) -> tuple:
    """ Perform the Gibbons, Ross and Shaken (1989) test.
        :param resid: Matrix of residuals from the OLS of size TxK.
        :param alpha: Vector of alphas from the OLS of size Kx1.
        :param factors: Matrix of factor returns of size TxJ.
        :return Test statistic and pValue of the test statistic.
    """
    # Determine the time series and assets
    iT, iK = resid.shape

    # Determine the amount of risk factors
    iJ = factors.shape[1]

    # Input size checks
    assert alpha.shape == (iK, 1)
    assert factors.shape == (iT, iJ)

    # Covariance of the residuals, variables are in columns.
    mCov = np.cov(resid, rowvar=False)

    # Mean of excess returns of the risk factors
    vMuRF = np.nanmean(factors, axis=0)

    try:
        assert vMuRF.shape == (1, iJ)
    except AssertionError:
        vMuRF = vMuRF.reshape(1, iJ)

    # Duplicate this series for T timestamps
    mMuRF = np.repeat(vMuRF, iT, axis=0)

    # Test statistic
    mCovRF = (factors - mMuRF).T @ (factors - mMuRF) / (iT - 1)
    dTestStat = (iT / iK) * ((iT - iK - iJ) / (iT - iJ - 1)) * \
                (alpha.T @ (np.linalg.inv(mCov) @ alpha)) / \
                (1 + (vMuRF @ (np.linalg.inv(mCovRF) @ vMuRF.T)))

    pVal = 1 - f.cdf(dTestStat, iK, iT-iK-1)

    return dTestStat, pVal
