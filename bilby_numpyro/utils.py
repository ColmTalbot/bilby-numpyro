import jax.numpy as jnp
from numpyro import deterministic, sample
from numpyro.distributions import Normal, Unit

__all__ = [
    "bilby_to_numpyro_prior",
    "bilby_to_numpyro_priors",
    "generic_bilby_likelihood_function",
    "construct_numpyro_model",
    "sample_prior",
]


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
    )

    mapping = dict(
        Cauchy=lambda x: Cauchy(x.alpha, x.beta),
        ChiSquared=lambda x: Chi2(x.nu),
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
        StudentT=lambda x: StudentT(x.df, x.mu, x.sigma),
        TruncatedGaussian=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
        TruncatedNormal=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
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


def generic_bilby_likelihood_function(likelihood, parameters, use_ratio=True):
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
    priors = bilby_to_numpyro_priors(priors)

    def model():
        parameters = sample_prior(priors, delta_fns)
        ln_l = generic_bilby_likelihood_function(
            likelihood, parameters, use_ratio=kwargs.get("use_ratio", True)
        )
        ln_l = jnp.nan_to_num(ln_l, nan=-jnp.inf)
        sample("log_likelihood", Unit(ln_l), obs=ln_l)

    return model


def sample_prior(priors, delta_fns):
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
    base = Normal(0, 1)
    for k, v in priors.items():
        if k in delta_fns:
            parameters[k] = deterministic(k, v.mean)
            continue
        try:
            temp = sample(f"{k}_scaled", Normal(0, 1))
            parameters[k] = deterministic(k, v.icdf(base.cdf(temp)))
        except Exception:
            print("Prior sampling failed", k, v)
            raise
    return parameters
