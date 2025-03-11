
import numpy as np
def variance(t, sigma_max, sigma_min, T):
    """
    Compute the variance as a function of time t.
    Variance decreases from sigma_max to sigma_min and then increases back to sigma_max.
    """
    if 0 <= t < T / 2:
        return sigma_max - (sigma_max - sigma_min) * (2 * t / T)
    elif T / 2 <= t <= T:
        return sigma_min + (sigma_max - sigma_min) * (2 * (t - T / 2) / T)
    else:
        return sigma_max  # Default value if outside defined range


def gaussian(x, t, sigma_max, sigma_min, T):
    """
    Compute the Gaussian function at time t.
    """
    sigma2 = variance(t, sigma_max, sigma_min, T)
    if sigma2 <= 0:
        return np.zeros_like(x)  # If variance is zero or negative, return zero array
    return 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-x ** 2 / (2 * sigma2))


def gaussian_derivative(x, t, sigma_max, sigma_min, T):
    """
    Compute the derivative of the Gaussian function at time t.
    """
    sigma2 = variance(t, sigma_max, sigma_min, T)
    if sigma2 <= 0:
        return np.zeros_like(x)  # If variance is zero or negative, return zero array
    sigma4 = sigma2 ** 2
    exp_term = np.exp(-x ** 2 / (2 * sigma2))
    derivative = (x ** 2 - sigma2) / (2 * sigma4) * exp_term
    return -1 / np.sqrt(2 * np.pi * sigma2) * derivative