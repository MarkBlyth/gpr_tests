#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.optimize
import scipy.io
import os.path
import warnings

import datagenerator as dg
import hyperpars as hp
import spectralrunner as fkl
import BarsNWrapper.wrapper as wrapper

import SingleCellCBC.gpr.gpr as mygpr
import SingleCellCBC.gpr.kernels as mykernels


GPR_SCHEMES = [
    "FKL",
    "SEKernel",
    "ModuloKernel",
    "PeriodicKernel",
    "Matern32",
    "PeriodicMatern32",
    "Matern52",
    "BARS",
    None,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Run experiment
using a data generator and a GPR scheme. The chosen data can be
either a model, which will be simulated, or a datafile. Datafiles
must be a saved np array, with the first row being the sample
times, and the second row being the recorded data. Models to
available to simulate are {0}""".format(
            dg.DATASETS.keys()
        )
    )
    parser.add_argument(
        "--data", "-d", help="Dataset to use", required=True,
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
        help="Save resulting plot with this name, instead of showing it",
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
        "-T",
        "--tests",
        help="Number of latent testpoints to evaluate the GP at",
        default=400,
        type=int,
    )
    parser.add_argument(
        "--niters",
        "-i",
        help="Number of training iterations for functional kernel learning",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--downsample",
        "-D",
        help="Downsample the training data using [::D]",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--period",
        "-P",
        help="Override signal period with this value, for periodic kernels",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sigmaf",
        "-f",
        help="Override sigma_f with this value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lengthscale",
        "-l",
        help="Override l with this value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--var", "-V", help="Plot variance bands, where available", action="store_true",
    )
    parser.add_argument(
        "--tmin", help="Lower-bound time to plot experimental data from", type=float, default=None
    )
    parser.add_argument(
        "--tmax", help="Upper-bound time to plot experimental data to", type=float, default=None
    )
    parser.add_argument(
        "--noplot", help="Skip plotting step", action="store_true", default=False
    )
    return parser.parse_args()


def get_hypers(args):
    # Find hyperparameters
    if args.hypers == "NoiseFitted":
        hyperset = hp.NOISE_FITTED_HYPERPARS
    else:
        hyperset = hp.CLEAN_FITTED_HYPERPARS
    try:
        hyperpars = hyperset[(args.data, args.model)]
    except KeyError:
        hyperpars = {}
        warnings.warn("No hyperparameters found, falling back to args")
    if args.sigmaf is not None:
        hyperpars["sigma_f"] = args.sigmaf
    if args.period is not None:
        hyperpars["period"] = args.period
    if args.lengthscale is not None:
        hyperpars["l"] = args.lengthscale
    if "l" not in hyperpars or "sigma_f" not in hyperpars or "period" not in hyperpars:
        warnings.warn(
            "One of l, sigma_f, period is not set. Will produce errors if a model requires them"
        )
    hyperpars["sigma_n"] = args.noise
    return hyperpars


def build_my_gpr(data_x, data_y, kernel, optimize):
    # Build and fit model
    kerneltype = mykernels.KERNEL_NAMES[kernel.name]
    print("Using kernel {0}".format(kerneltype))
    if not optimize:
        gpr_model = mygpr.GPR(data_x, data_y, kernel)
        print("Log likelihood = {0}".format(gpr_model.log_likelihood))
        return gpr_model

    kerneltype = mykernels.KERNEL_NAMES[kernel.name]
    sigma_n = kernel.sigma_n
    if kernel.name in ["PeriodicSEKernel", "PeriodicKernel", "PeriodicMatern32"]:
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
        print(
            "Optimization success: {0}. {1}".format(
                optimized.success, optimized.message
            )
        )
        sigma_f, l, period = optimized.x ** 2
        print(optimized.x ** 2)
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
    return model


def get_data(args, noise, to_validate):
    if args.data not in dg.DATASETS.keys():
        # Using a datafile instead of a model
        if not os.path.isfile(args.data):
            raise FileNotFoundError(
                "data not a specified model and not an existing datafile"
            )
        warnings.warn(
            "Noise level sigma_n is not a trainable parameter in this script. Appropriate values must be estimated when working with real data."
        )
        ts, ys = np.load(args.data)
        if args.tmin is None:
            args.tmin = np.min(ts)
        if args.tmax is None:
            args.tmax = np.max(ts)
        ys = ys[np.logical_and(ts>=args.tmin, ts<=args.tmax)]
        ts = ts[np.logical_and(ts>=args.tmin, ts<=args.tmax)]
    else:
        # Generate some neuron training data
        ts, ys = dg.simple_data_generator(
            dg.DATASETS[args.data],
            atol=args.atol,
            rtol=args.rtol,
            transients=args.transients,
        )
        ys += np.random.normal(0, noise, ys.shape)

    # Downsample training data, if needed
    if args.downsample is not None:
        ts = ts[:: args.downsample]
        ys = ys[:: args.downsample]

    # Split up into test and training data, if needed
    ts_test, ys_test = None, None
    if to_validate:
        indices = np.arange(len(ys))
        test_indices = np.mod(indices, 3) == 0
        ts_test, ys_test = ts[test_indices], ys[test_indices]
        ts, ys = ts[np.logical_not(test_indices)], ys[np.logical_not(test_indices)]

    # Points to evaluate a model at
    gpr_ts = np.linspace(min(ts), max(ts), args.tests)

    return ts, ys, ts_test, ys_test, gpr_ts


def main():
    args = parse_args()
    hyperpars = get_hypers(args)
    ts, ys, ts_test, ys_test, gpr_ts = get_data(args, args.noise, args.validate)
    print("Working with {0} datapoints".format(len(ts)))

    # If we want one of the my-code GPRs...
    if args.model in mykernels.KERNEL_NAMES:
        kernel = mykernels.KERNEL_NAMES[args.model](**hyperpars)
        model = build_my_gpr(ts, ys, kernel, args.optimize)
        print("Evaluating at {0} latent points".format(len(gpr_ts)))
        gpr_ys = model(gpr_ts)

    elif args.model == "FKL":
        model = fkl.run(ts, ys, ts_test, ys_test, n_iters=args.niters)
        gpr_ys = model(gpr_ts)

    elif args.model == "BARS":
        model = wrapper.barsN(ts, ys, burnin=5000, prior_param=(0, 80), iknots=80)
        gpr_ys = model(gpr_ts)

    if ts_test is not None:
        gpr_test_ys = model(ts_test)
        MSPE = np.mean((gpr_test_ys - ys_test) ** 2)
        rMSPE = np.mean(((gpr_test_ys - ys_test) / ys_test) ** 2)
        print("MSPE: {0}, on {1} datapoints".format(MSPE, len(gpr_test_ys)))
        print("rMSPE: {0}, on {1} datapoints".format(rMSPE, len(gpr_test_ys)))

    if not args.noplot:
        print("Generating plot")
        # Plot results
        fig, ax = plt.subplots()
        # Generate and plot noise-free data, if working with noised simulations
        if args.noise != 0 and args.data in dg.DATASETS.keys() and args.model is not None:
            clean_ts, clean_ys, _, _, _ = get_data(args, noise=0, to_validate=False)
            ax.plot(clean_ts, clean_ys, "k--", label="Noise-free signal", alpha=0.5)
        # Plot (possibly noised) simulation
        ax.plot(ts, ys, label="Model simulation")
        # Plot variance bands, if appropriate
        try:
            if isinstance(model, mygpr.GPR) and args.var:
                print("Finding variance")
                variance = np.diag(model.get_variance(gpr_ts))
                twosigma = 2 * np.sqrt(np.abs(variance))
                ax.fill_between(
                    gpr_ts,
                    gpr_ys - twosigma,
                    gpr_ys + twosigma,
                    label=r"$2\sigma$ bounds",
                    alpha=0.5,
                )
        except NameError:
            # No model defined
            pass
        # Plot validation points, if appropriate
        if args.validate:
            ax.scatter(ts_test, gpr_test_ys, label="Predicted test points")
            ax.scatter(ts_test, ys_test, c="red", marker="X", label="Actual test points")
            ax.legend()
        # Plot model fit, if appropriate
        if args.model is not None:
            ax.plot(gpr_ts, gpr_ys, label="{0} fit".format(args.model))
            ax.legend()
        if args.save is not None:
            plt.savefig(args.save)
        else:
            plt.show()


if __name__ == "__main__":
    main()
