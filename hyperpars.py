""" NOTE The log-likelihood can vary slightly, depending on the number
of datapoints used. atol and rtol are proxies for this. For the noisy
simulations I often used rtol=1e-6, instead of the default 1e-3, so
your found log-likelihoods might be slightly different to those stated
here, depending on the rtol value that was used. No rtol was set for
the Hodgkin Huxley simulations. """

NOISE_FITTED_HYPERPARS = {

    # Simga_n = 0.1. LL=-290.094 at rtol=1e-6 (Noise-free -456,778)
    ("FitzhughNagumo", "PeriodicKernel"): { 
        "sigma_f": 2.91128931,
        "l": 0.07422477,
        "period": 36.99430406,
    },  

    # Sigma_n = 0.1. LL = -288.997 at rtol=1e-6.
    ("FitzhughNagumo", "MySEKernel"): {
        "sigma_f":4.7271724,
        "l": 5.01367266,
    },

    # Sigma_n = 0.1. LL = -283 (Noise-free -457,451)
    ("FitzhughNagumo","ModuloKernel"):{
        "sigma_f": 3.8836687936535084,
        "l": 4.447273411719457,
        "period": 36.92512214463045,
    },

    # Sigma_n = 0.05. LL = -284 (noise-free -377,038)
    ("HRFast", "PeriodicKernel"):{
        "sigma_f":2.8480833530602627,
        "l":0.18851337537121246,
        "period": 4.368130551017679,
    },

    # Sigma_n = 0.05. LL = -287 at rtol=1e-6 (noise-free -145,659)
    ("HRFast", "ModuloKernel"): { 
        "sigma_f": 2.816513202867441,
        "l": 0.17130043951941454,
        "period": 4.368740126818106,
    },  

    # Sigma_n = 0.05. LL = -277 at rtol=1e-6 (noise-free -1,120,449,
    # bad fit)
    ("HRFast", "MySEKernel"): {
        "sigma_f":5.182069020807821,
        "l":0.27973214754870057,
    }, 

    # Attempted optimization with sigma_n = 2. Bad LL, but params
    # stopped changing. Possibly near a saddle point? LL=-6170
    # (noise-free -175,468,920)
    ("HodgkinHuxley", "MySEKernel"): {
        "sigma_f": 4.57346376e+02,
        "l": 5.33533781e-02,
    },  


    # Ran to optimization termination, sigma_n = 2. LL = -2657
    # (noise-free -39,117,518)
    ("HodgkinHuxley", "PeriodicKernel"): {
        "sigma_f": 4.07319588e+02,
        "l": 1.55770848e-03,
        "period": 16.4737292,
    }, 

    # Ran to optimization termination, sigma_n = 2. LL = -2690
    # (noise-free -39,092,086)
    ("HodgkinHuxley", "ModuloKernel"): {
        "sigma_f": 4.05645373e+02,
        "l": 2.12527567e-02,
        "period": 16.4741041,
    }, 

}


CLEAN_FITTED_HYPERPARS = {

    # OPTIMIZED on no-noise. LL=-8547
    ("HodgkinHuxley", "MySEKernel"): {
        "sigma_f": 4.97343652e+02,
        "l": 4.31450040e-03,
    },  

    # OPTIMIZED on no-noise. LL=-9952
    ("HodgkinHuxley", "ModuloKernel"): {
        "sigma_f": 5.90733143e+02,
        "l": 1.51545473e-07,
        "period": 9.24228872,
    }, 

    # OPTIMIZED on no-noise. LL=-7063
    ("HodgkinHuxley", "PeriodicKernel"): {
        "sigma_f": 4.85988689e+02,
        "l": 3.03975258e-06,
        "period": 16.4740305,
    }, 

    # OPTIMIZED on no-noise. LL=-135.48
    ("FitzhughNagumo", "MySEKernel"): {"sigma_f": 2.53347, "l": 1.14897},  

    # OPTIMIZED on no noise. LL= -178
    ("FitzhughNagumo", "ModuloKernel"): { 
        "sigma_f": 2.73057518,
        "l": 1.40443265e-03,
        "period": 38.2597690,
    },  

    # OPTIMIZED on no noise. LL= -170
    ("FitzhughNagumo", "PeriodicKernel"): { 
        "sigma_f": 3.25972023,
        "l": 1.47510418e-04,
        "period": 36.959
    },  

    # OPTIMIZED on no noise. LL = -114
    ("HRFast", "MySEKernel"): {
        "sigma_f": 1.9841870275260058,
        "l": 0.0938153427971645,
    }, 

    # OPTIMIZED on no noise. LL = -132
    ("HRFast", "PeriodicKernel"): {
        "sigma_f": 1.63489095,
        "l": 7.70769527e-04,
        "period": 4.37281924,
    },  

    # OPTIMIZED on no=noise. LL = -152
    ("HRFast", "ModuloKernel"): {
        "sigma_f": 1.68339210,
        "l": 1.85967897e-10,
        "period": 4.36874764,
    },  
}
