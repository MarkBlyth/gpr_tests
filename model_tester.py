#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import datagenerator as dg

import SingleCellCBC.gpr.gpr as mygpr
import SingleCellCBC.gpr.kernels as mykernels


GPR_SCHEMES = ["MySEKernel", "ModuloKernel", "CosineKernel", None]

DEFAULT_HYPERPARS = {
    ("HodgkinHuxley", "MySEKernel"): {
        "sigma_f": 5.90466936e+01,
        "l": 3.86948099e-02,
    },  # Optimized, but possibly not very well

    ("FitzhughNagumo", "MySEKernel"): {"sigma_f": 2.53347, "l": 1.14897},

    ("HindmarshRose", "MySEKernel"): {}, #TODO

    ("HRFast", "MySEKernel"): {}, # TODO  

    ("HodgkinHuxley", "ModuloKernel"): { # TODO
        "sigma_f": 5.90466936e+01,
        "l": 0.05,
        "period": 18,
    }, 

    ("FitzhughNagumo", "ModuloKernel"): { # Optimization was unsuccessful for this one
        "sigma_f": 2.6,
        "l": 0.3,
        "period": 40
    },  

    ("HindmarshRose", "ModuloKernel"): { # TODO
        "period": 450
    },  

    ("HRFast", "ModuloKernel"): {"period": 4.5},  # TODO

    ("HodgkinHuxley", "CosineKernel"): { # TODO
        "sigma_f": 5.90466936e+01,
        "period": 18,
    }, 

    ("FitzhughNagumo", "CosineKernel"): { # TODO Unoptimized, but a fair guess
        "sigma_f": 2.53347,
        "l": 1.14887,
        "period": 40
    },  

    ("HindmarshRose", "CosineKernel"): { # TODO
        "period": 450
    },  

    ("HRFast", "CosineKernel"): {"period": 4.5},  # TODO
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment using a data generator and a GPR scheme."
    )
    parser.add_argument(
        "--data",
        "-d",
        help="Dataset to use",
        choices=list(dg.DATASETS.keys()),
        required=True,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="GPR model to use. Plot raw data if no GPR model is chosen.",
        default=None,
        choices=GPR_SCHEMES,
    )
    parser.add_argument(
        "--noise", "-n", help="Observation noise variance", default=0, type=float,
    )
    parser.add_argument(
        "--atol",
        "--a",
        help="Solver absolute error tolerance",
        default=1e-6,
        type=float,
    )
    parser.add_argument(
        "--rtol",
        "--r",
        help="Solver relative error tolerance",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--optimize",
        "-o",
        help="Optimize GPR model",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


def build_my_gpr(data_x, data_y, kernel, optimize):
    # Build and fit model
    ts = np.linspace(min(data_x), max(data_x), 10 * len(data_x))
    if not optimize:
        gpr_model = mygpr.GPR(data_x, data_y, kernel)
        print("Log likelihood = {0}".format(gpr_model.log_likelihood))
        return ts, gpr_model(ts)

    kerneltype = mykernels.KERNEL_NAMES[kernel.name]
    sigma_n = kernel.sigma_n
    if kernel.name in ["PeriodicSEKernel", "CosineKernel"]:
        # Optimize a periodic kernel
        # Could probably reuse some code with the other optimizer
        initial = np.sqrt(np.array([kernel.sigma_f, kernel.l[0], kernel._period]))

        def objective(pars):
            sigma_f, l, T = pars ** 2
            newkernel = kerneltype(T, sigma_n, sigma_f, l)
            model = mygpr.GPR(data_x, data_y, newkernel)
            print(pars ** 2, model.log_likelihood)
            return -model.log_likelihood

        optimized = scipy.optimize.minimize(objective, initial, tol=1e-3)
        print("Optimization success: {0}. {1}".format(optimized.success, optimized.message))
        sigma_f, l, period = optimized.x ** 2
        print(optimized.x **2)
        print(
            "Fitted model: period: {0}, sigma_f: {1}, l: {2}".format(period, sigma_f, l)
        )
        newkernel = kerneltype(period, sigma_n, sigma_f, l)

    else:
        # Optimize a non-periodic kernel
        initial = np.sqrt(np.array([kernel.sigma_f, kernel.l[0]]))

        def objective(pars):
            sigma_f, l = pars ** 2
            newkernel = kerneltype(sigma_n, sigma_f, l)
            model = mygpr.GPR(data_x, data_y, newkernel)
            print(pars ** 2, model.log_likelihood)
            return -model.log_likelihood

        optimized = scipy.optimize.minimize(objective, initial, tol=1e-3)
        sigma_f, l = optimized.x ** 2
        print("Fitted model: sigma_f: {0}, l: {1}".format(sigma_f, l))
        newkernel = kerneltype(sigma_n, sigma_f, l)

    model = mygpr.GPR(data_x, data_y, newkernel)
    print("Log likelihood = {0}".format(model.log_likelihood))
    return ts, model(ts)


def main():
    args = parse_args()

    # Generate some neuron training data
    ts, ys = dg.simple_data_generator(
        dg.DATASETS[args.data],
        observation_noise=args.noise,
        atol=args.atol,
        rtol=args.rtol,
    )
    print("Working with {0} datapoints".format(len(ts)))

    # If we want one of the my-code GPRs...
    if args.model in ["MySEKernel", "ModuloKernel", "CosineKernel"]:
        hyperpars = {
            **DEFAULT_HYPERPARS[(args.data, args.model)],
            "sigma_n": args.noise,
        }
        if args.model == "MySEKernel":
            kernel = mykernels.SEKernel(**hyperpars)
        elif args.model == "ModuloKernel":
            kernel = mykernels.PeriodicSEKernel(**hyperpars)
        elif args.model == "CosineKernel":
            kernel = mykernels.CosinePeriodicKernel(**hyperpars)
        gpr_ts, gpr_ys = build_my_gpr(ts, ys, kernel, args.optimize)

    # If we want a GPR but it's not one I've set up yet...
    elif args.model is not None:
        raise NotImplementedError

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(ts, ys, label="Model simulation")
    if args.model is not None:
        ax.plot(gpr_ts, gpr_ys, label="GPR fit")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
