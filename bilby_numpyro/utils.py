import jax.numpy as jnp
import numpy as np
from numpyro import deterministic, sample
from numpyro.distributions import constraints
from numpyro.distributions import Normal, TransformedDistribution, Uniform, Unit
from numpyro.distributions.transforms import ParameterFreeTransform

__all__ = [
    "bilby_to_numpyro_prior",
    "bilby_to_numpyro_priors",
    "generic_bilby_likelihood_function",
    "construct_numpyro_model",
    "sample_prior",
]


class CosineTransform(ParameterFreeTransform):
    """
    Transform via the cosine function: y = cos(x)
    """

    domain = constraints.interval(0, np.pi)
    codomain = constraints.interval(-1, 1)

    def __call__(self, x):
        return jnp.arccos(x)

    def inv(self, y):
        return jnp.cos(y)

    def log_abs_det_jacobian(self, x, y, t):
        """
        y = arccos(x)

        dy/dx = -1/sqrt(1-x^2)
        log|dy/dx| = log(1/sqrt(1-x^2)) = -0.5*log(1-x^2)
        """
        return -0.5 * jnp.log(1 - x**2)


class SineTransform(ParameterFreeTransform):
    """
    Transform via the sine function: y = sin(x)
    """

    domain = constraints.interval(0, np.pi)
    codomain = constraints.interval(-1, 1)

    def __call__(self, x):
        return jnp.arcsin(x)

    def inv(self, y):
        return jnp.sin(y)

    def log_abs_det_jacobian(self, x, y, t):
        """
        y = arcsin(x)
        dy/dx = 1/sqrt(1-x^2)
        log|dy/dx| = log(1/sqrt(1-x^2))
        """
        return -0.5 * jnp.log(1 - x**2)


class Cosine(TransformedDistribution):
    """
    .. note:: We keep the same notation `rate` as in Pyro but
        it plays the role of scale parameter of InverseGamma in literatures
        (e.g. wikipedia: https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
    """

    arg_constraints = {}
    support = constraints.interval(-np.pi / 2, np.pi / 2)

    def __init__(
        self,
        *,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Uniform(low=-1, high=1)
        super().__init__(
            base_dist, SineTransform(), validate_args=validate_args
        )

    @property
    def mean(self) -> float:
        return 0.0

    @property
    def variance(self) -> float:
        return np.pi**2 / 4 - 2

    def entropy(self) -> float:
        return np.log(4) - 2


class Sine(TransformedDistribution):
    """
    .. note:: We keep the same notation `rate` as in Pyro but
        it plays the role of scale parameter of InverseGamma in literatures
        (e.g. wikipedia: https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
    """

    arg_constraints = {}
    support = constraints.interval(0, np.pi)

    def __init__(
        self,
        *,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Uniform(low=-1, high=1)
        super().__init__(
            base_dist, CosineTransform(), validate_args=validate_args
        )

    @property
    def mean(self) -> float:
        return np.pi / 2

    @property
    def variance(self) -> float:
        return np.pi**2 / 4 - 2

    def entropy(self) -> float:
        return np.log(4) - 2


def bilby_to_numpyro_prior(value):
    """
    Convert a :code:`bilby` prior to a :code:`numpyro` prior.

    This function attempts to match the prior class to a :code:`numpyro` prior class.
    If no match is found, it defaults to a uniform prior.

    Some features are not yet implemented, including constraints and conditional priors.

    Parameters
    ==========
    value: bilby.core.prior.Prior
        The prior to convert.

    Returns
    =======
    numpyro.distributions.Distribution
        The converted prior.
    """

    from numpyro.distributions import (
        Cauchy,
        Chi2,
        Delta,
        DoublyTruncatedPowerLaw,
        Exponential,
        Gamma,
        HalfNormal,
        Logistic,
        LogNormal,
        LogUniform,
        Normal,
        StudentT,
        TruncatedNormal,
        Uniform,
        VonMises,
    )

    mapping = dict(
        Cauchy=lambda x: Cauchy(x.alpha, x.beta),
        ChiSquared=lambda x: Chi2(x.nu),
        Cosine=lambda x: Cosine(),
        Constraint=lambda x: ValueError("Constraint prior not yet supported."),
        DeltaFunction=lambda x: Delta(x.peak),
        Exponential=lambda x: Exponential(x.mu),
        Gamma=lambda x: Gamma(x.k, x.theta),
        HalfNormal=lambda x: HalfNormal(x.sigma),
        HalfGaussian=lambda x: HalfNormal(x.sigma),
        Logisitic=lambda x: Logistic(x.mu, x.sigma),
        LogGaussian=lambda x: LogNormal(x.mu, x.sigma),
        LogNormal=lambda x: LogNormal(x.mu, x.sigma),
        LogUniform=lambda x: LogUniform(x.minimum, x.maximum),
        Normal=lambda x: Normal(x.mu, x.sigma),
        Gaussian=lambda x: Normal(x.mu, x.sigma),
        PowerLaw=lambda x: DoublyTruncatedPowerLaw(
            x.alpha, x.minimum, x.maximum
        ),
        Sine=lambda x: Sine(),
        StudentT=lambda x: StudentT(x.df, x.mu, x.sigma),
        TruncatedGaussian=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
        TruncatedNormal=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
        # Uniform=lambda x: Uniform(x.minimum, x.maximum) if not x.boundary == "periodic" else VonMises(loc=0, concentration=1e-7),
        Uniform=lambda x: Uniform(x.minimum, x.maximum),
    )
    if value.__class__.__name__ in mapping:
        return mapping[value.__class__.__name__](value)
    else:
        print(f"No matching prior class for {value}, defaulting to uniform")
        return Uniform(value.minimum, value.maximum)


def bilby_to_numpyro_priors(priors):
    """
    Convert a :code:`bilby.core.prior.PriorDict` to a dictionary of :code:`numpyro`
    priors.

    Parameters
    ==========
    priors: bilby.core.prior.PriorDict
        The priors to convert in :code:`bilby` format.

    Returns
    =======
    dict[str, numpyro.distributions.Distribution]
        The converted priors in :code:`numpyro` format.
    """
    jpriors = dict()
    for key, value in priors.items():
        jpriors[key] = bilby_to_numpyro_prior(value)
    return jpriors


def generic_bilby_likelihood_function(likelihood, parameters, *, use_ratio=True):
    """
    A wrapper to allow a :code:`Bilby` likelihood to be used with :code:`jax`.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to evaluate.
    parameters: dict
        The parameters to evaluate the likelihood at.
    use_ratio: bool, optional
        Whether to evaluate the likelihood ratio or the full likelihood.
        Default is :code:`True`.
    """
    if use_ratio:
        return likelihood.log_likelihood_ratio(parameters)
    else:
        return likelihood.log_likelihood(parameters)


def construct_numpyro_model(
    likelihood, priors, likelihood_func=generic_bilby_likelihood_function, **kwargs
):
    """
    Construct a :code:`numpyro` compatible model from :code:`bilby` likelihood and priors.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to evaluate.
    priors: bilby.core.prior.PriorDict
        The priors to use for inference.
    likelihood_func: callable, optional
        The function to use to evaluate the likelihood. Default is
        :func:`gwpopulation.experimental.jax.generic_bilby_likelihood_function`.
    kwargs: dict, optional
        Additional keyword arguments to pass to the likelihood function.

    Returns
    =======
    callable
        The :code:`numpyro` model.
    """
    from bilby.core.prior import DeltaFunction

    priors = priors.copy()
    priors.convert_floats_to_delta_functions()
    delta_fns = {
        k: v for k, v in priors.items() if isinstance(v, (int, float, DeltaFunction))
    }
    numpyro_priors = bilby_to_numpyro_priors(priors)

    from numpyro.infer.reparam import CircularReparam
    from numpyro import handlers

    reparam_config = dict()
    periodic = list()
    for key, value in priors.items():
        if key in delta_fns:
            continue
        if value.boundary == "periodic":
            periodic.append(key)
            # reparam_config[key] = CircularReparam()
            numpyro_priors[f"{key}_x"] = Normal(0, 1)
            numpyro_priors[f"{key}_y"] = Normal(0, 1)
            del numpyro_priors[key]
    # reparam_config = {
    #     k: CircularReparam() for k, v in priors.items()
    #     if k not in delta_fns and v.boundary =="periodic"
    # }
    # print(reparam_config, numpyro_priors)

    # @handlers.reparam(config=reparam_config)
    def model(*args, beta=1, **model_kwargs):
        parameters = sample_prior(numpyro_priors, delta_fns, periodic)
        ln_l = likelihood_func(likelihood, parameters, **kwargs) * beta
        sample("log_likelihood", Unit(ln_l), obs=ln_l)

    return model


def sample_prior(priors, delta_fns, periodic=None):
    """
    Draw a sample from the :code:`numpyro` prior distribution.

    Parameters
    ==========
    priors: dict[str, numpyro.distributions.Distribution]
        The priors to sample from.
    delta_fns: list[str]
        The list of keys that are delta functions.
    """
    parameters = dict()
    # base = Normal(0, 1)
    for k, v in priors.items():
        if k in delta_fns:
            parameters[k] = deterministic(k, v.mean)
            continue
        try:
            parameters[k] = sample(k, v)
            # temp = sample(f"{k}_scaled", Normal(0, 1))
            # parameters[k] = deterministic(k, v.icdf(base.cdf(temp)))
        except Exception:
            print("Prior sampling failed", k, v)
            raise
    if periodic is not None:
        for key in periodic:
            x = parameters[f"{key}_x"]
            y = parameters[f"{key}_y"]
            angle = jnp.arctan2(y, x) % (2 * jnp.pi)
            parameters[key] = deterministic(key, angle)
    return parameters
