#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.optimize
import scipy.io

import datagenerator as dg
import hyperpars as hp
import spectral_TEST as fkl

import SingleCellCBC.gpr.gpr as mygpr
import SingleCellCBC.gpr.kernels as mykernels


GPR_SCHEMES = ["FKL", "MySEKernel", "ModuloKernel", "PeriodicKernel", None]


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
        "--hypers",
        "-p",
        help="Hyperparameter set to use.",
        default="NoiseFitted",
        choices=["NoiseFitted", "CleanFitted"],
    )
    parser.add_argument(
        "--noise", "-n", help="Observation noise variance", default=0, type=float,
    )
    parser.add_argument(
        "--atol",
        "-a",
        help="Solver absolute error tolerance",
        default=1e-6,
        type=float,
    )
    parser.add_argument(
        "--rtol",
        "-r",
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
    parser.add_argument(
        "--save",
        "-s",
        help="Save training data to a .mat file of specified filename",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--transients",
        "-t",
        help="Pre-integrate for specified time, to allow transients to settle",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--validate",
        "-v",
        help="If set, one third of the datapoints will be removed, and used to compute the mean square prediction error",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--niters",
        "-i",
        help="Number of training iterations for functional kernel learning",
        default=20,
        type=int,
    )
    return parser.parse_args()


def build_my_gpr(data_x, data_y, kernel, optimize):
    # Build and fit model
    if not optimize:
        gpr_model = mygpr.GPR(data_x, data_y, kernel)
        print("Log likelihood = {0}".format(gpr_model.log_likelihood))
        return gpr_model

    kerneltype = mykernels.KERNEL_NAMES[kernel.name]
    sigma_n = kernel.sigma_n
    if kernel.name in ["PeriodicSEKernel", "PeriodicKernel"]:
        # Optimize a periodic kernel
        # Could probably reuse some code with the other optimizer
        initial = np.sqrt(
            np.array([kernel.sigma_f, kernel.l[0], kernel._period]))

        def objective(pars):
            sigma_f, l, T = pars ** 2
            newkernel = kerneltype(T, sigma_n, sigma_f, l)
            model = mygpr.GPR(data_x, data_y, newkernel)
            print(pars ** 2, model.log_likelihood)
            return -model.log_likelihood

        optimized = scipy.optimize.minimize(objective, initial, tol=1e-3)
        print(
            "Optimization success: {0}. {1}".format(
                optimized.success, optimized.message
            )
        )
        sigma_f, l, period = optimized.x ** 2
        print(optimized.x ** 2)
        print(
            "Fitted model: period: {0}, sigma_f: {1}, l: {2}".format(
                period, sigma_f, l)
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
    return model


def get_data(args):
    # Generate some neuron training data
    ts, ys = dg.simple_data_generator(
        dg.DATASETS[args.data],
        atol=args.atol,
        rtol=args.rtol,
        transients=args.transients,
    )

    ts_test, ys_test = None, None
    if args.validate:
        indices = np.arange(len(ys))
        test_indices = (np.mod(indices, 3) == 0)
        ts_test, ys_test = ts[test_indices], ys[test_indices]
        ts, ys = ts[np.logical_not(test_indices)
                    ], ys[np.logical_not(test_indices)]

    ys += np.random.normal(0, args.noise, ys.shape)
    return ts, ys, ts_test, ys_test


def main():
    args = parse_args()

    if args.data == "HindmarshRose":
        raise NotImplementedError("HindmarshRose is not implemented here.")

    ts, ys, ts_test, ys_test = get_data(args)
    gpr_ts = np.linspace(min(ts), max(ts), 10 * len(ts))
    print("Working with {0} datapoints".format(len(ts)))

    # Find hyperparameters
    hyperset = (
        hp.NOISE_FITTED_HYPERPARS
        if args.hypers == "NoiseFitted"
        else hp.CLEAN_FITTED_HYPERPARS
    )

    # If we want one of the my-code GPRs...
    if args.model in ["MySEKernel", "ModuloKernel", "PeriodicKernel"]:
        hyperpars = {
            **hyperset[(args.data, args.model)],
            "sigma_n": args.noise,
        }
        if args.model == "MySEKernel":
            kernel = mykernels.SEKernel(**hyperpars)
        elif args.model == "ModuloKernel":
            kernel = mykernels.PeriodicSEKernel(**hyperpars)
        elif args.model == "PeriodicKernel":
            kernel = mykernels.PeriodicKernel(**hyperpars)
        model = build_my_gpr(ts, ys, kernel, args.optimize)
        gpr_ys = model(gpr_ts)

    if args.model == "FKL":
        model = fkl.run(ts, ys, ts_test, ys_test,n_iters=args.niters)
        gpr_ys = model(gpr_ts)

    # If we want a GPR but it's not one I've set up yet...
    elif args.model is not None:
        raise NotImplementedError

    if args.save is not None:
        scipy.io.savemat(args.save + ".mat", dict(v=ys, t=ts))

    if ts_test is not None:
        gpr_test_ys = model(ts_test)
        MSPE = np.mean((gpr_test_ys - ys_test)**2)
        rMSPE = np.mean(((gpr_test_ys - ys_test)/ys_test)**2)
        print("MSPE: {0}, on {1} datapoints".format(MSPE, len(gpr_test_ys)))
        print("rMSPE: {0}, on {1} datapoints".format(rMSPE, len(gpr_test_ys)))

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(ts, ys, label="Model simulation")
    if args.model is not None:
        ax.plot(gpr_ts, gpr_ys, label="GPR fit")
        ax.legend()
    if args.validate:
        ax.scatter(ts_test, gpr_test_ys, label="Predicted test points")
        ax.scatter(ts_test, ys_test, c="red", marker="X",
                   label="Actual test points")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
