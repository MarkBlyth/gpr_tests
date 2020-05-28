#!/usr/bin/env python3

"""
Uses the functional kernel learning method of Benton et al., see
https://arxiv.org/abs/1910.13565 . This code is based entirely on
theirs, a fork of which can be found in the included spectralgp 
submodule.
"""

import torch
import gpytorch

import spectralgp.spectralgp as spectralgp

torch.set_default_dtype(torch.float64)


def run(
    train_x,
    train_y,
    test_x,
    test_y,
    n_omegas=100,   # Number of datapoints used to construct spectrum
    omega_max=1,    # Largest omega (Fourier spectrum frequency)
    n_iters=100,    # Number of elliptic slice sampling iterations to run
    ess_iters=20,   # Number of elliptic slice sampling samples per iteration
    optim_iters=1,  # Number of optimization iterations
):
    train_y = torch.tensor(train_y.astype(float)).view(-1)
    train_x = torch.tensor(train_x.astype(float)).view(-1)
    if test_y is not None:
        test_y = torch.tensor(test_y.astype(float)).view(-1)
        test_x = torch.tensor(test_x.astype(float)).view(-1)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        train_x, train_y, test_x, test_y = (
            train_x.cuda(),
            train_y.cuda(),
            test_x.cuda(),
            test_y.cuda(),
        )

    ###########################################
    #  set up the spectral and latent models  #
    ###########################################
    data_lh = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.SmoothedBoxPrior(1e-8, 1e-3)
    )
    data_mod = spectralgp.models.SpectralModel(
        train_x,
        train_y,
        data_lh,
        normalize=False,
        symmetrize=False,
        num_locs=n_omegas,
        spacing="random",
        omega_max=omega_max,
    )
    data_lh.raw_noise = torch.tensor(-3.5)

    ################################
    #  set up alternating sampler  #
    ################################

    alt_sampler = spectralgp.samplers.AlternatingSampler(
        [data_mod],
        [data_lh],
        spectralgp.sampling_factories.ss_factory,
        [spectralgp.sampling_factories.ess_factory],
        totalSamples=n_iters,
        numInnerSamples=ess_iters,
        numOuterSamples=optim_iters,
    )
    alt_sampler.run()

    data_mod.eval()
    if test_y is not None:
        test_mspe = torch.mean(torch.pow(data_mod(test_x).mean - test_y, 2))
        test_rmspe = torch.mean(
            torch.pow((data_mod(test_x).mean - test_y) / test_y, 2))
        print("Test MSPE: ", test_mspe)
        print("Test rMSPE: ", test_rmspe)

    latent_mod = data_mod.covar_module.latent_mod
    print(list(data_mod.named_parameters()))
    print(list(latent_mod.named_parameters()))

    def nice_model(xs):
        xs = torch.tensor(xs.astype(float)).view(-1)
        last_samples = min(10, alt_sampler.gsampled[0].size(1))
        # preprocess the spectral samples #
        out_samples = alt_sampler.gsampled[0][0, :, -last_samples:].detach()

        pred_data = torch.zeros(len(xs), last_samples)

        data_mod.eval()
        with torch.no_grad():
            for ii in range(last_samples):
                data_mod.covar_module.set_latent_params(out_samples[:, ii])
                data_mod.set_train_data(train_x, train_y)
                out = data_mod(xs)
                pred_data[:, ii] = out.mean
        return pred_data[:, 0].detach().cpu().numpy()

    return nice_model
