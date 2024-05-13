# bilby-numpyro

Sampler plugin for using `numpyro` samplers with `Bilby`

## Limitations

The likelihood class you use should be JIT compilable with `JAX`.
Generally, this will be the case if you use `jax.numpy` and `jax.scipy` instead of `numpy`/`scipy`.

Some `Bilby` priors have direct equivalents in `numpyro` and so we use those automatically.
You will see logged warnings if a prior is not available and a `Uniform` prior over the same range will be used instead.

## Usage

The sampler should always be given as `numpyro` regardless of which sampler (e.g., `HMC`, `NUTS`) is actually being used.

The sampler along with any sampler keyword arguments should be passed in the usual way to `bilby.core.sampler.run_sampler`.

This has not been tested with `numpyro` parallelisation.
