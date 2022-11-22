# GRS_Test
Script to perform the asset pricing test of Gibbons, Ross, and Shanken (1989)

## Inputs
The functions expects:
  - A TxK np.array of residuals from an OLS of assets on common risk factors,
  - A Kx1 np.array of intercepts from an OLS of assets on common risk factors,
  - A TxJ np.array of risk factors.
  
T: The time series dimension,
K: Assets,
J: Risk Factors.

## Output
A tuple consisting of the test statistic and the corresponding p-Value drawn from an F distribution.
