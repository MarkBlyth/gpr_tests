""" NOTE The log-likelihood can vary slightly, depending on the number
of datapoints used. atol and rtol are proxies for this. For the noisy
simulations I often used rtol=1e-6, instead of the default 1e-3, so
your found log-likelihoods might be slightly different to those stated
here, depending on the rtol value that was used. No rtol was set for
the Hodgkin Huxley simulations. """

_fitzhugh_nagumo_noisy = {
    # Sigma_n = 0.1. LL = -400.7 (Noise-free -342)
    ("FitzhughNagumo", "Matern32"): {
        "sigma_f": 6.440462523764936,
        "l": 6.315544053000552,
    },
    # Sigma_n = 0.1. LL = -374
    ("FitzhughNagumo", "PeriodicMatern32"): {
        "sigma_f": 4.608716985986971,
        "l": 0.45933697040210353,
        "period": 36.68702806545151,
    },
    # Sigma_n = 0.1. LL = -401 at rtol=1e-6 (noise-free -337)
    ("FitzhughNagumo", "Matern52"): {
        "sigma_f": 5.374279993215441,
        "l": 4.004628468563434,
    },
    # Simga_n = 0.1. LL=-290.094 at rtol=1e-6 (Noise-free -456,778)
    ("FitzhughNagumo", "PeriodicKernel"): {
        "sigma_f": 2.91128931,
        "l": 0.07422477,
        "period": 36.99430406,
    },
    # Sigma_n = 0.1. LL = -288.997 at rtol=1e-6.
    ("FitzhughNagumo", "SEKernel"): {
        "sigma_f": 4.7271724,
        "l": 5.01367266,
    },
    # Sigma_n = 0.1. LL = -283 (Noise-free -457,451)
    ("FitzhughNagumo", "ModuloKernel"): {
        "sigma_f": 3.8836687936535084,
        "l": 4.447273411719457,
        "period": 36.92512214463045,
    },
}


_hindmarsh_rose_fast_noisy = {
    # Sigma_n = 0.05. LL = -284 (noise-free -377,038)
    ("HRFast", "PeriodicKernel"): {
        "sigma_f": 2.8480833530602627,
        "l": 0.18851337537121246,
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
    ("HRFast", "SEKernel"): {
        "sigma_f": 5.182069020807821,
        "l": 0.27973214754870057,
    },
    # Sigma_n = 0.05. LL = -276 at rtol=1e-6 (noise-free -246)
    ("HRFast", "Matern32"): {
        "sigma_f": 5.972329059747012,
        "l": 1.113944194766012,
    },
    # Sigma_n = 0.05. LL = -255 at rtol=1e-6 (noise-free -642)
    ("HRFast", "PeriodicMatern32"): {
        "sigma_f": 4.519665163964117,
        "l": 0.8004769675268222,
        "period": 4.339166456489576,
    },
    # Sigma_n = 0.05. LL = -274 at rtol=1e-6 (noise-free -236)
    ("HRFast", "Matern52"): {
        "sigma_f": 5.699425486440175,
        "l": 0.7962712867444136,
    },
}


_hodgkin_huxley_noisy = {
    # Attempted optimization with sigma_n = 2. Bad LL, but params
    # stopped changing. Possibly near a saddle point? LL=-6170
    # (noise-free -175,468,920)
    ("HodgkinHuxley", "SEKernel"): {
        "sigma_f": 4.57346376e02,
        "l": 5.33533781e-02,
    },
    # sigma_n = 2
    ("HodgkinHuxley", "Matern32"): {
        "sigma_f": 798.0901792870231,
        "l": 0.6985381186697813,
    },
    # sigma_n = 2. LL = -3069  at rtol=1e-6 (noise-free -5219)
    ("HodgkinHuxley", "Matern52"): {
        "sigma_f": 618.4726293241634,
        "l": 0.39905731896925706,
    },
    # sigma_n = 2. LL = -1768 at rtol=1e-6
    ("HodgkinHuxley", "PeriodicMatern32"): {
        "sigma_f": 722.4990014822946,
        "l": 0.12114279055587815,
        "period": 16.473304332154413,
    },
    # Ran to optimization termination, sigma_n = 2. LL = -2657
    # (noise-free -39,117,518)
    ("HodgkinHuxley", "PeriodicKernel"): {
        "sigma_f": 4.07319588e02,
        "l": 1.55770848e-03,
        "period": 16.4737292,
    },
    # Ran to optimization termination, sigma_n = 2. LL = -2690
    # (noise-free -39,092,086)
    ("HodgkinHuxley", "ModuloKernel"): {
        "sigma_f": 4.05645373e02,
        "l": 2.12527567e-02,
        "period": 16.4741041,
    },
}


_fitzhugh_nagumo_clean = {
    # OPTIMIZED
    ("FitzhughNagumo", "PeriodicMatern32"): {
        "sigma_f": 4.543321785067146,
        "l": 0.45178443851923694,
        "period": 36.70592638526368,
    },
    # OPTIMIZED on no-noise.
    ("FitzhughNagumo", "Matern32"): {
        "sigma_f": 11.175296817511112,
        "l": 7.044808744925122,
    },
    # OPTIMIZED
    ("FitzhughNagumo", "Matern52"): {
        "sigma_f": 9.585586305709052,
        "l": 3.667510267272261,
    },
    # OPTIMIZED on no-noise. LL=-135.48
    ("FitzhughNagumo", "SEKernel"): {
        "sigma_f": 2.53347,
        "l": 1.14897
    },
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
        "period": 36.959,
    },
}

_hodgkin_huxley_clean = {
    # Optimized no noise.
    ("HodgkinHuxley", "Matern32"): {
        "sigma_f": 599.3747172611621,
        "l": 0.5296389516518929,
    },
    # OPTIMIZED
    ("HodgkinHuxley", "Matern52"): {
        "sigma_f": 472.4963849977704,
        "l": 0.2495895442270998,
    },
    # OPTIMIZED on no-noise. LL=-8547
    ("HodgkinHuxley", "SEKernel"): {
        "sigma_f": 4.97343652e02,
        "l": 4.31450040e-03,
    },
    # OPTIMIZED on no-noise. LL=-9952
    ("HodgkinHuxley", "ModuloKernel"): {
        "sigma_f": 5.90733143e02,
        "l": 1.51545473e-07,
        "period": 9.24228872,
    },
    # OPTIMIZED on no-noise. LL=-7063
    ("HodgkinHuxley", "PeriodicKernel"): {
        "sigma_f": 4.85988689e02,
        "l": 3.03975258e-06,
        "period": 16.4740305,
    },
    # OPTIMIZED
    ("HodgkinHuxley", "PeriodicMatern32"): {
        "sigma_f": 745.8564657654453,
        "l": 0.11131822480079424,
        "period": 16.473677419209384,
    },
}


_hindmarsh_rose_fast_clean = {
    # OPTIMIZED on no noise. LL = -114
    ("HRFast", "SEKernel"): {
        "sigma_f": 1.9841870275260058,
        "l": 0.0938153427971645,
    },
    # OPTIMIZED
    ("HRFast", "Matern32"): {
        "sigma_f": 11.4211272201263,
        "l": 1.4790190433448793,
    },
    # OPTIMIZED
    ("HRFast", "PeriodicMatern32"): {
        "sigma_f": 5.975274077867351,
        "l": 0.41647014071979827,
        "period": 4.337715784818653,
    },
    # OPTIMIZED
    ("HRFast", "Matern52"): {
        "sigma_f": 13.036740146551901,
        "l": 0.9196207118937813
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


CLEAN_FITTED_HYPERPARS = {
    **_hodgkin_huxley_clean,
    **_fitzhugh_nagumo_clean,
    **_hindmarsh_rose_fast_clean,
}

NOISE_FITTED_HYPERPARS = {
    **_hodgkin_huxley_noisy,
    **_fitzhugh_nagumo_noisy,
    **_hindmarsh_rose_fast_noisy,
}
