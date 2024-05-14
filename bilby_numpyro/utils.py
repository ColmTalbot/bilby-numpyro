import jax.numpy as jnp
from numpyro import deterministic, sample
from numpyro.distributions import Normal, Unit


def bilby_to_numpyro_prior(value):
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
    jpriors = dict()
    for key, value in priors.items():
        jpriors[key] = bilby_to_numpyro_prior(value)
    return jpriors


def generic_bilby_likelihood_function(likelihood, parameters, use_ratio=True):
    likelihood.parameters.update(parameters)
    if use_ratio:
        return likelihood.log_likelihood_ratio()
    else:
        return likelihood.log_likelihood()


def construct_numpyro_model(
    likelihood, priors, likelihood_func=generic_bilby_likelihood_function, **kwargs
):
    from bilby.core.prior import DeltaFunction

    priors = priors.copy()
    priors.convert_floats_to_delta_functions()
    delta_fns = {
        k: v for k, v in priors.items() if isinstance(v, (int, float, DeltaFunction))
    }
    priors = bilby_to_numpyro_priors(priors)

    def model():
        parameters = sample_prior(priors, delta_fns)
        ln_l = jnp.nan_to_num(
            likelihood_func(likelihood, parameters, **kwargs), nan=-jnp.inf
        )
        sample("log_likelihood", Unit(ln_l), obs=ln_l)

    return model


def sample_prior(priors, delta_fns):
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
