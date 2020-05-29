import scipy.integrate
import numpy as np


def hodgkin_huxley(x, gk=80, Ek=-100, gna=100, Ena=50, gl=0.1, El=-67, I=1.75, C=1):
    """Right-hand side for the Hodgkin-Huxley equation.

        x: ndarray
            np array of the system state, of form [v,n,m,h]

        **kwargs:
            Optional parameters. The parameters that can be modified,
            and their default values, are as follows:
                gk : 36; potassium channel conductance
                gna : 120; sodium channel conductance
                gl : 0.3; leak channel conductance
                Ek : -12; potassium Nernst equilibrium
                Ena : 120; sodium Nernst equilibrium
                El : 10.6; leak Nernst equilibrium
                I : 0; applied current
                C : 1; membrane capacitance

    Returns the derivatives [v_dot, n_dot, m_dot, h_dot]
    """
    v, n, m, h = x

    # Ion channel activations
    def alpha_m(v): return 0.32 * (v + 54) / (1 - np.exp(-(v + 54) / 4))
    def beta_m(v): return 0.28 * (v + 27) / (np.exp((v + 27) / 5) - 1)
    def alpha_h(v): return 0.128 * np.exp(-(50 + v) / 18)
    def beta_h(v): return 4 / (1 + np.exp(-(v + 27) / 5))
    def alpha_n(v): return 0.032 * (v + 52) / (1 - np.exp(-(v + 52) / 5))
    def beta_n(v): return 0.5 * np.exp(-(57 + v) / 40)

    # Ion currents
    IK = gk * (n ** 4) * (v - Ek)
    INa = gna * (m ** 3) * h * (v - Ena)
    Il = gl * (v - El)

    # ODEs
    v_dot = (I - IK - INa - Il) / C
    n_dot = alpha_n(v) * (1 - n) - beta_n(v) * n
    m_dot = alpha_m(v) * (1 - m) - beta_m(v) * m
    h_dot = alpha_h(v) * (1 - h) - beta_h(v) * h

    return np.array([v_dot, n_dot, m_dot, h_dot])


def fitzhugh_nagumo(x, I=1):
    """
    Defines the RHS of the FH model
        x: ndarray
            np array of the system state, of form [v, w]

        **kwargs:
            Optional parameters. The parameters that can be modified,
            and their default values, are as follows:
                I : 0; applied current

    Returns the derivatives [v_dot, w_dot]
    """
    v, w = x  # Unpack state
    v_dot = v - (v ** 3) / 3 - w + I
    w_dot = 0.08 * (v + 0.7 - 0.8 * w)
    return np.array([v_dot, w_dot])


def hindmarsh_rose(x, a=1, b=3, c=1, d=5, s=4, xr=-8 / 5, r=0.001, I=2):
    """
    Defines the RHS of the Hindmarsh-Rose model
        x: ndarray
            np array of the system state, of form [u,v,w]

        **kwargs:
            Optional parameters. The parameters that can be modified,
            and their default values, are as follows:
                xr : -8/5; dimensionless parameter
                I : 2; current-like parameter
                r : 0.001; timescale separation
                a : 1; dimensionless parameter
                b : 3; dimensionless parameter
                c : 1; dimensionless parameter
                d : 5; dimensionless parameter
                s : 4; dimensionless parameter

    Returns the derivatives [u_dot, v_dot, w_dot]
    """
    u, v, w = x

    u_dot = v - a * u ** 3 + b * u ** 2 - w + I
    v_dot = c - d * u ** 2 - v
    w_dot = r * (s * (u - xr) - w)
    return np.array([u_dot, v_dot, w_dot])


def hindmarsh_rose_fast(x, a=1, b=3, c=1, d=5, z=0, I=2):
    """
    Defines the RHS of the Hindmarsh-Rose fast subsystem
        x: ndarray
            np array of the system state, of form [u,v]

        **kwargs:
            Optional parameters. The parameters that can be modified,
            and their default values, are as follows:
                I : 2; current-like parameter
                a : 1; dimensionless parameter
                b : 3; dimensionless parameter
                c : 1; dimensionless parameter
                d : 5; dimensionless parameter
                z : 0; slow subsystem psuedoparameter

    Returns the derivatives [u_dot, v_dot]
    """
    u, v = x
    w = z * np.ones(u.shape)

    new_x = np.array([u, v, w])

    return hindmarsh_rose(new_x, a, b, c, d, I=I)[:-1]


def simple_data_generator(model, observation_noise=0, transients=0, **kwargs):
    """Model runner for simulating neurons. Default parameter values
    are used, unless set in **kwargs. Example initial conditions and
    integration bounds are used, unless set in **kwargs.

        model : function
            The defining RHS function for the neuron model of
            interest.

        observation_noise : float>0
            Variance of a normally distributed random variable, added
            to the output retrospectively, to simulate observation
            noise.

        transients : float>0
            How long to pre-integrate for, to allow transients to die
            out

        kwargs :
            Any desired arguments. Arguments that match an argument
            available to the model function will be extracted and
            passed to the function, so model parameters can be set in
            this way. All remaining arguments are then passed to the
            solver. Some interesting kwarg to set include:

                I: model applied current
                y0: model initial condition
                rtol: integration relative error tolerance
                atol: integration absolute error tolerance
                t_span: integration period

    Returns a tuple (ts, xs), where ts are timepoints, and xs are the
    first dimension of the solution vectors.

    """
    # Extract the kwargs that were meant for the solver, and model
    # function
    model_kwargs = {k: kwargs[k]
                    for k in kwargs if k in model.__code__.co_varnames}
    solver_kwargs = {k: kwargs[k] for k in kwargs if k not in model_kwargs}
    # Define some default initial conditions for each model
    y0s = {
        fitzhugh_nagumo: np.array([0, 0]),
        hodgkin_huxley: np.array([-70, 0, 0, 1]),
        hindmarsh_rose: np.array([0, -8, 2]),
        hindmarsh_rose_fast: np.array([0, 0]),
    }
    t_spans = {
        hindmarsh_rose_fast: np.array([-transients, 20]),
        hindmarsh_rose: np.array([-transients, 1000]),
        fitzhugh_nagumo: np.array([-transients, 150]),
    }
    tspan = t_spans[model] if model in t_spans else np.array(
        [-transients, 50])
    if model not in y0s:
        raise ValueError(
            "{0} is not a supported model type. Must be one of {1}".format(
                model, y0s.keys()
            )
        )
    # Set up the default integration time and initial conditions,
    # allowing for kwargs to override these
    solver_args = {**{"t_span": tspan, "y0": y0s[model]}, **solver_kwargs}

    # Set up and solve!
    def model_func(t, x): return model(x, **model_kwargs)
    solution = scipy.integrate.solve_ivp(model_func, **solver_args)

    vs = solution.y[0][solution.t >= 0]
    noise = np.random.normal(0, observation_noise, vs.shape)

    return solution.t[solution.t >= 0], vs + noise


DATASETS = {
    "HodgkinHuxley": hodgkin_huxley,
    "FitzhughNagumo": fitzhugh_nagumo,
    "HindmarshRose": hindmarsh_rose,
    "HRFast": hindmarsh_rose_fast,
}
