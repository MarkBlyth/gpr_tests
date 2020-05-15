from .neuronmodels import *
__all__ = ["neuronmodels"]

DATASETS = {
    "HodgkinHuxley": neuronmodels.hodgkin_huxley,
    "FitzhughNagumo": neuronmodels.fitzhugh_nagumo,
    "HindmarshRose": neuronmodels.hindmarsh_rose,
    "HRFast": neuronmodels.hindmarsh_rose_fast,
}
