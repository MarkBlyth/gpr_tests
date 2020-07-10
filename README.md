
# Table of Contents

1.  [Summary](#orgf479852)
2.  [Compiling free-knot splines](#orgf840b78)
3.  [Models](#orgbe9aa9f)
4.  [Data](#orge3accea)
5.  [Hyperparameters](#org84276a1)
6.  [Dependencies](#org06c00af)
7.  [Examples](#org9c1e99d)
8.  [References](#orgec141fe)


<a id="orgf479852"></a>

# Summary

These codes are designed for testing out a range of non-parametric regression methods on neuronal data. 

    usage: model_tester.py [-h] --data DATA
    		       [--model {FKL,SEKernel,ModuloKernel,PeriodicKernel,Matern32,PeriodicMatern32,Matern52,BARS,None}]
    		       [--hypers {NoiseFitted,CleanFitted}] [--noise NOISE] [--atol ATOL] [--rtol RTOL]
    		       [--optimize] [--save SAVE] [--transients TRANSIENTS] [--validate] [-T TESTS]
    		       [--niters NITERS] [--downsample DOWNSAMPLE] [--period PERIOD] [--sigmaf SIGMAF]
    		       [--lengthscale LENGTHSCALE] [--var] [--tmin TMIN] [--tmax TMAX] [--noplot]
    
    Run experiment using a data generator and a GPR scheme. The chosen data can be either a model, which will be
    simulated, or a datafile. Datafiles must be a saved np array, with the first row being the sample times, and the
    second row being the recorded data. Models to available to simulate are dict_keys(['HodgkinHuxley',
    'FitzhughNagumo', 'HindmarshRose', 'HRFast'])
    
    optional arguments:
      -h, --help            show this help message and exit
      --data DATA, -d DATA  Dataset to use
      --model {FKL,SEKernel,ModuloKernel,PeriodicKernel,Matern32,PeriodicMatern32,Matern52,BARS,None}, -m {FKL,SEKernel,ModuloKernel,PeriodicKernel,Matern32,PeriodicMatern32,Matern52,BARS,None}
    			GPR model to use. Plot raw data if no GPR model is chosen.
      --hypers {NoiseFitted,CleanFitted}, -p {NoiseFitted,CleanFitted}
    			Hyperparameter set to use.
      --noise NOISE, -n NOISE
    			Observation noise variance
      --atol ATOL, -a ATOL  Solver absolute error tolerance
      --rtol RTOL, -r RTOL  Solver relative error tolerance
      --optimize, -o        Optimize GPR model
      --save SAVE, -s SAVE  Save resulting plot with this name, instead of showing it
      --transients TRANSIENTS, -t TRANSIENTS
    			Pre-integrate for specified time, to allow transients to settle
      --validate, -v        If set, one third of the datapoints will be removed, and used to compute the mean square
    			prediction error
      -T TESTS, --tests TESTS
    			Number of latent testpoints to evaluate the GP at
      --niters NITERS, -i NITERS
    			Number of training iterations for functional kernel learning
      --downsample DOWNSAMPLE, -D DOWNSAMPLE
    			Downsample the training data using [::D]
      --period PERIOD, -P PERIOD
    			Override signal period with this value, for periodic kernels
      --sigmaf SIGMAF, -f SIGMAF
    			Override sigma_f with this value
      --lengthscale LENGTHSCALE, -l LENGTHSCALE
    			Override l with this value
      --var, -V             Plot variance bands, where available
      --tmin TMIN           Lower-bound time to plot experimental data from
      --tmax TMAX           Upper-bound time to plot experimental data to
      --noplot              Skip plotting step

The data and model flags are the most interesting, see below.


<a id="orgf840b78"></a>

# Compiling free-knot splines

Free-knot regression splines are currently implemented through a
wrapper of Wallstrom et al.'s C implementation, see [1]. This requires
a compiled binary. To make it, follow the instructions in the README
and INSTRUCTIONS file in the BarsNWrapper submodule. The rest of the
code will run fine without this having been built, so this step is
only necessary if you want to test BARS.


<a id="orgbe9aa9f"></a>

# Models

The following regression models are implemented:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Model</th>
<th scope="col" class="org-left">Name</th>
<th scope="col" class="org-left">Notes</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">GP: Functional kernel learning</td>
<td class="org-left">FKL</td>
<td class="org-left">See [2]</td>
</tr>


<tr>
<td class="org-left">GP: Square-exponential kernel</td>
<td class="org-left">SEKernel</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">GP: Modulo-based psuedokernel</td>
<td class="org-left">ModuloKernel</td>
<td class="org-left">I made this method up, not recommended</td>
</tr>


<tr>
<td class="org-left">GP: Periodic SE kernel</td>
<td class="org-left">PeriodicKernel</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">GP: Matern 3/2 kernel</td>
<td class="org-left">Matern32</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">GP: Periodic Matern 3/2 kernel</td>
<td class="org-left">PeriodicMatern32</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">GP: Matern 5/2 kernel</td>
<td class="org-left">Matern52</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">Bayesian free-knot splines</td>
<td class="org-left">BARS</td>
<td class="org-left">See [1], [3]</td>
</tr>
</tbody>
</table>

To choose a method, put its name after the model flag, eg.

    ./model_tester.py -d [DATA] -m SEKernel
    ./model_tester.py -d [DATA] -m BARS


<a id="orge3accea"></a>

# Data

Three datasets are provided

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Name</th>
<th scope="col" class="org-left">Dataset</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">HRFast</td>
<td class="org-left">Hindmarsh Rose fast subsystem</td>
</tr>


<tr>
<td class="org-left">HodgkinHuxley</td>
<td class="org-left">A Hodgkin Huxley neuron</td>
</tr>


<tr>
<td class="org-left">FitzhughNagumo</td>
<td class="org-left">A Fitzhugh Nagumo neuron</td>
</tr>
</tbody>
</table>

These are selected using the data flag, eg.

    ./model_tester.py -d FitzhughNagumo

If the specified dataset isn't one of the above, it is assumed to be the filename of some experimental data.
Datafiles should be a saved np array, where the first row is sample times, and the second row is sample values, eg.

    ./model_tester.py -d experimental_data.np


<a id="org84276a1"></a>

# Hyperparameters

The Gaussian process models which rely on hyperparameters have some optimised versions pre-coded.
Two sets are provided; clean-fitted hyperparameters are those optimised on a noise-free signal; noise-fitted hyperparameters are those optimised on a noisy signal.
Noise-fitted hyperparameters generally give better results.
To select the hyperparameters, use the hypers flag.

    ./model_tester.py [...] -p NoiseFitted
    ./model_tester.py [...] --hypers CleanFitted


<a id="org06c00af"></a>

# Dependencies

Tested by installing

-   Python3.6 virtualenv
-   torch 1.5.0
-   gpytorch 1.1.1
-   sklearn 0.23.1

Full package list: (note some of these may be redundant now!)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Package</th>
<th scope="col" class="org-right">Version</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">astor</td>
<td class="org-right">0.8.1</td>
</tr>


<tr>
<td class="org-left">attrs</td>
<td class="org-right">19.3.0</td>
</tr>


<tr>
<td class="org-left">cycler</td>
<td class="org-right">0.10.0</td>
</tr>


<tr>
<td class="org-left">future</td>
<td class="org-right">0.18.2</td>
</tr>


<tr>
<td class="org-left">gpytorch</td>
<td class="org-right">1.1.1</td>
</tr>


<tr>
<td class="org-left">importlib-metadata</td>
<td class="org-right">1.6.0</td>
</tr>


<tr>
<td class="org-left">joblib</td>
<td class="org-right">0.15.1</td>
</tr>


<tr>
<td class="org-left">kiwisolver</td>
<td class="org-right">1.2.0</td>
</tr>


<tr>
<td class="org-left">matplotlib</td>
<td class="org-right">3.2.1</td>
</tr>


<tr>
<td class="org-left">more-itertools</td>
<td class="org-right">8.3.0</td>
</tr>


<tr>
<td class="org-left">numpy</td>
<td class="org-right">1.18.5</td>
</tr>


<tr>
<td class="org-left">packaging</td>
<td class="org-right">20.4</td>
</tr>


<tr>
<td class="org-left">pip</td>
<td class="org-right">20.1.1</td>
</tr>


<tr>
<td class="org-left">pip-autoremove</td>
<td class="org-right">0.9.1</td>
</tr>


<tr>
<td class="org-left">pluggy</td>
<td class="org-right">0.13.1</td>
</tr>


<tr>
<td class="org-left">py</td>
<td class="org-right">1.8.1</td>
</tr>


<tr>
<td class="org-left">pyparsing</td>
<td class="org-right">2.4.7</td>
</tr>


<tr>
<td class="org-left">pytest</td>
<td class="org-right">5.4.2</td>
</tr>


<tr>
<td class="org-left">python-dateutil</td>
<td class="org-right">2.8.1</td>
</tr>


<tr>
<td class="org-left">scikit-learn</td>
<td class="org-right">0.23.1</td>
</tr>


<tr>
<td class="org-left">scipy</td>
<td class="org-right">1.4.1</td>
</tr>


<tr>
<td class="org-left">setuptools</td>
<td class="org-right">47.1.1</td>
</tr>


<tr>
<td class="org-left">six</td>
<td class="org-right">1.15.0</td>
</tr>


<tr>
<td class="org-left">sklearn</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">threadpoolctl</td>
<td class="org-right">2.1.0</td>
</tr>


<tr>
<td class="org-left">torch</td>
<td class="org-right">1.5.0</td>
</tr>


<tr>
<td class="org-left">wcwidth</td>
<td class="org-right">0.1.9</td>
</tr>


<tr>
<td class="org-left">zipp</td>
<td class="org-right">3.1.0</td>
</tr>
</tbody>
</table>


<a id="org9c1e99d"></a>

# Examples

Simulate and plot a Hodgkin Huxley neuron

    ./model_tester.py -d HodgkinHuxley

Fit a square-exponential GP to a Fitzhugh Nagumo simulation, with \(\sigma^2 = 0.1\) observation noise

    ./model_tester.py -d FitzhughNagumo -m SEKernel -n 0.1

Find the test validation score of a Matern32 kernel, on the Hindmarsh Rose fast subsystem

    ./model_tester.py -d HRFast -m Matern32 -v

Improve the simulation accuracy, thus increasing the number of datapoints

    ./model_tester.py -d HRFast -m Matern32 -v -r 1e-9


<a id="orgec141fe"></a>

# References

[1] Wallstrom, Garrick, Jeffrey Liebner, and Robert E. Kass. "An
implementation of Bayesian adaptive regression splines (BARS) in C
with S and R wrappers." Journal of Statistical Software 26.1
(2008): 1.

[2] Benton, Gregory, et al. "Function-space distributions over
kernels." Advances in Neural Information Processing Systems. 2019.

[3] DiMatteo, Ilaria, Christopher R. Genovese, and Robert E. Kass. "Bayesian curve‐fitting with free‐knot splines." Biometrika 88.4 (2001): 1055-1071.

