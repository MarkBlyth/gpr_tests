#!/usr/bin/env python3

"""Cannibalised from the run-experiments.py script, in
nssm_gp/experiments.
"""


from nssm_gp.nssm_gp.initializers import init_gsm, init_spectral, init_neural
from nssm_gp.nssm_gp.spectral_kernels import SMKernel

import tensorflow as tf
import gpflow
from gpflow import settings
import gpflow.training.monitor as mon

import numpy as np
import sklearn.utils
import shutil


float_type = settings.dtypes.float_type

# Model / inference hyperparameters
NUM_INDUCING_POINTS = 500  # Uses min(this, actual number of datapoints)
LEARNING_RATE = 5e-4  # default 1e-4
EPOCHS = 2000         # default 500
BATCH_SIZE = 512      # default 512
N_INITS = 3           # default 5
Q = 3               # default 3
MAX_FREQ = 0.001    # default 20
MAX_LEN = 10        # default 10
LATENT_ELL = 0.05   # default 1.0


def simple_model(x, y, likelihood, noise, M=NUM_INDUCING_POINTS, bs=BATCH_SIZE, ARD=True):
    # Gaussian kernel
    ell = np.std(x)
    var = np.std(y)
    kern_rbf = gpflow.kernels.RBF(x.shape[1], variance=var, lengthscales=LATENT_ELL, ARD=ARD)
    # Randomly select inducing point locations among given inputs
    idx = np.random.randint(len(x), size=M)
    Z = x[idx, :].copy()
    # Create SVGP with Gaussian likelihood
    m_rbf = gpflow.models.SVGP(x, y, kern_rbf, likelihood, Z, minibatch_size=bs)
    return m_rbf




def run(
    xs,
    ys,
    kernel,  # gsm, sm, rbf, neural
    noise=0,
    ard=True,  # Automatic relevance determination
    bs=BATCH_SIZE,  # Minibatch size for training
    lr=LEARNING_RATE,  # Learning rate for traininig
    m=NUM_INDUCING_POINTS,  # Number of inducing points for SVGP
    q=Q,  # Number of spectral mixture components
    freq=None,
    ell=None,
):

    # TODO preprocess: remove mean using linear regression
    y_mean = np.mean(ys)
    y_energy = np.var(ys)
    ys = (ys-y_mean)/y_energy
    
    m = min(NUM_INDUCING_POINTS, len(xs))
    xs, ys = xs.reshape((-1, 1)), ys.reshape((-1, 1))
    xs, ys = sklearn.utils.shuffle(xs, ys)

    latent_ell = ell if ell is not None else LATENT_ELL
    max_freq = freq if freq is not None else MAX_FREQ

    # Select model type
    likelihood = gpflow.likelihoods.Gaussian(0.1 ** 2)

    model_functions = {
        "gsm": lambda x, y, likelihood, noise, q=Q, ARD=False, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS: init_gsm(
            x,
            y,
            M=M,
            max_freq=max_freq,
            Q=q,
            minibatch_size=bs,
            ell=latent_ell,
            n_inits=N_INITS,
            noise_var=noise,
            ARD=ARD,
            likelihood=likelihood,
        ),
        "sm": lambda x, y, likelihood, noise, q=Q, ARD=False, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS: init_spectral(
            x,
            y,
            kern=SMKernel,
            n_inits=N_INITS,
            M=M,
            Q=q,
            minibatch_size=bs,
            noise_var=noise,
            ARD=ARD,
            likelihood=likelihood,
        ),
        "rbf": simple_model,
        "neural": lambda x, y, likelihood, noise, q=Q, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS, ARD=None: init_neural(
            x,
            y,
            n_inits=N_INITS,
            M=M,
            Q=q,
            minibatch_size=bs,
            likelihood=likelihood,
            hidden_sizes=(32, 32),
        ),
    }

    if kernel == "rbf":
        model = model_functions[kernel](
            xs, ys, noise=noise, ARD=ard, likelihood=likelihood, bs=int(bs), M=int(m)
        )
    else:
        model = model_functions[kernel](
            xs, ys, noise=noise, q=int(q), ARD=ard, likelihood=likelihood, bs=int(bs), M=int(m),
        )

    # Create monitoring
    session = model.enquire_session()
    global_step = mon.create_global_step(session)
    model_name = "nssm-gp"
    tensorboard_dir = "tensorboard/" + model_name
    shutil.rmtree(tensorboard_dir, ignore_errors=True)
    with mon.LogdirWriter(tensorboard_dir) as writer:
        tensorboard_task = (
            mon.ModelToTensorBoardTask(writer, model, only_scalars=False)
            .with_name("tensorboard")
            .with_condition(mon.PeriodicIterationCondition(50))
            .with_exit_condition(True)
        )
        print_task = (
            mon.PrintTimingsTask()
            .with_name("print")
            .with_condition(mon.PeriodicIterationCondition(250))
        )

        # Create optimizer
        epoch_steps = max(len(xs) // int(bs), 1)
        learning_rate = tf.train.exponential_decay(
            float(lr), global_step, decay_steps=epoch_steps, decay_rate=0.99
        )
        optimizer = gpflow.train.AdamOptimizer(learning_rate)

        maxiter = epoch_steps * EPOCHS
        print("Optimizing model (running {} iterations)...".format(maxiter))
        with mon.Monitor(
            [tensorboard_task, print_task], session, global_step, print_summary=True
        ) as monitor:
            optimizer.minimize(
                model, maxiter=maxiter, step_callback=monitor, global_step=global_step,
            )
    print(model)

    def new_model(x):
        newx = x.reshape((-1, 1))
        mean, _ = model.predict_y(newx)
        return (y_energy*mean) + y_mean

    return new_model
