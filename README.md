# bilby-numpyro

Sampler plugin for using `numpyro` samplers with `Bilby`

`bilby-numpyro` is a sampler plugin for using `numpyro` samplers with `Bilby` likelihoods and priors.
Any likelihood that uses only `JAX` JIT-compilable functions/classes works automatically.

## Usage

The sampler should always be given as `numpyro` regardless of which sampler (e.g., `HMC`, `NUTS`) is actually being used.
The sampler is specified with `sampler_name` and any sampler keyword arguments should be passed in the usual way to
`bilby.core.sampler.run_sampler` for example

```python
run_sampler(
    ...,
    sampler="numpyro",
    sampler_name="NUTS",
    num_warmup=500,
    num_samples=1000,
)
```

This has not been tested with `numpyro` parallelisation.

## Limitations

The likelihood class you use should be JIT compilable with `JAX`.
Generally, this will be the case if you use `jax.numpy` and `jax.scipy` instead of `numpy`/`scipy`.

Some `Bilby` priors have direct equivalents in `numpyro` and so we use those automatically.
You will see logged warnings if a prior is not available and a `Uniform` prior over the same range will be used instead.

