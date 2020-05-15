from . import neuronmodels
from . import datasets
__all__ = ["neuronmodels", "datasets"]

DATASETS = {
    "HodgkinHuxley": neuronmodels.hodgkin_huxley,
    "FitzhughNagumo": neuronmodels.fitzhugh_nagumo,
    "HindmarshRose": neuronmodels.hindmarsh_rose,
    "HRFast": neuronmodels.hindmarsh_rose_fast,
}
