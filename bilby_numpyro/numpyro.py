import inspect
import os
from importlib import import_module
import dill as pickle

import arviz
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.infer
from bilby.core.sampler.base_sampler import Sampler, signal_wrapper
from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir, logger
from numpyro.infer import MCMC, Predictive, log_likelihood


from .utils import construct_numpyro_model, generic_bilby_likelihood_function


class NumPyro(Sampler):
    """bilby wrapper for samplers implemented in `NumPyro` (https://pyro.ai/numpyro/)

    All keyword arguments needed for the sampler are passed via the keyword arguments
    passed to this class.
    The needed arguments are automatically determined from the sampler's signature
    other than the :code:`sampler_name` which should be importable from
    :code:`numpyro.infer`.

    Notes
    =====
    This sampler will fail if the likelihood function is not compatible with
    `jax` and `numpyro`.
    """

    sampling_seed_key = "seed"

    default_kwargs = dict(
        sampler_name="NUTS",
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        verbose=True,
        seed=1234,
        plot_trace=True,
        check_point=False,
        n_check_point=None,
        thinning=1,
        resume=False,
        postprocess_fn=None,
        chain_method='parallel',
        jit_model_args=False,
    )

    @signal_wrapper
    def run_sampler(self):
        """Runs Nestle sampler with given kwargs and returns the result

        Returns
        =======
        bilby.core.result.Result: Packaged information about the result

        """

        check_directory_exists_and_if_not_mkdir(self.outdir)

        rng_key = jax.random.PRNGKey(self.kwargs[self.sampling_seed_key])
        validate_key, sample_key, predictive_key = jax.random.split(rng_key, 3)

        numpyro_model = self.model
        validate_model(numpyro_model, validate_key)

        sampler_class = getattr(numpyro.infer, self.kwargs["sampler_name"])

        if (
            self.kwargs["check_point"]
            and self.kwargs["n_check_point"] is not None
        ):
            logger.info("Running numpyro with checkpointing enabled")
            self.kwargs["num_total_samples"] = self.kwargs["num_samples"]
            self.kwargs["num_samples"] = self.kwargs["n_check_point"]

            sampler = self.evaluate_with_kwargs(sampler_class, numpyro_model)
            mcmc = self.evaluate_with_kwargs(MCMC, sampler)
            mcmc, samples, log_likes = self._run_with_checkpointing(
                mcmc, sample_key
            )

            self.kwargs["num_samples"] = self.kwargs.pop("num_total_samples")
        else:
            logger.info("Running numpyro without checkpointing enabled")

            sampler = self.evaluate_with_kwargs(sampler_class, numpyro_model)
            mcmc = self.evaluate_with_kwargs(MCMC, sampler)
            mcmc.run(sample_key)
            samples = mcmc.get_samples()
            log_likes = log_likelihood(mcmc.sampler.model, mcmc.get_samples())

        predictive = Predictive(numpyro_model, samples)(predictive_key)
        predictive = {
            key: predictive[key]
            for key in set(predictive.keys()).difference(samples.keys())
        }
        azdata = arviz.from_dict(
            samples, posterior_predictive=predictive, log_likelihood=log_likes
        )

        keys = [
            key
            for key in azdata.posterior.keys()
            if not key.endswith("_scaled")
        ]

        if self.kwargs["plot_trace"]:
            self.plot_trace(azdata, keys)

        self.result.posterior = azdata.to_dataframe(groups="posterior")[keys]
        self.result.sampler_output = azdata.posterior
        self.result.log_likelihood_evaluations = azdata.log_likelihood[
            "log_likelihood"
        ].values.flatten()

        return self.result

    @property
    def model(self):
        likelihood_func = self.likelihood_function
        return construct_numpyro_model(
            likelihood=self.likelihood,
            priors=self.priors,
            likelihood_func=likelihood_func,
            **self.filter_kwargs(likelihood_func),
        )

    @property
    def likelihood_function(self):
        likelihood_func = self.kwargs.get(
            "likelihood_func", generic_bilby_likelihood_function
        )
        if isinstance(likelihood_func, str) and "." not in likelihood_func:
            raise ValueError(
                f"Likelihood function should be importable, got {likelihood_func}"
            )
        elif isinstance(likelihood_func, str):
            split_model = likelihood_func.split(".")
            module = ".".join(split_model[:-1])
            function = split_model[-1]
            likelihood_func = getattr(import_module(module), function)
        return likelihood_func

    def filter_kwargs(self, func):
        return {
            key: self.kwargs[key]
            for key in inspect.signature(func).parameters.keys()
            if key in self.kwargs
        }

    def evaluate_with_kwargs(self, func, *args, **kwargs):
        return func(*args, **kwargs, **self.filter_kwargs(func))

    def plot_trace(self, azdata, keys=None):
        _ = arviz.plot_trace(azdata, var_names=keys)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"{self.label}_trace.png"))
        plt.close()

    def _run_test(self):
        """
        Runs to test whether the sampler is properly running with the given
        kwargs without actually running to the end

        Returns
        =======
        bilby.core.result.Result: Dummy container for sampling results.

        """
        self.kwargs["num_warmup"] = 2
        self.kwargs["num_samples"] = 2
        return self.run_sampler()

    def write_current_state(self):
        """
        Nestle doesn't support checkpointing so no current state will be
        written on interrupt.
        """
        pass

    def _run_with_checkpointing(self, mcmc, sample_key):

        checkpoint_file = f"{self.outdir}/numpyro_checkpoint.pkl"

        # first run
        if not os.path.isfile(checkpoint_file):

            new_key, sample_key = jax.random.split(sample_key, 2)

            mcmc.run(sample_key)

            checkpoint = {}
            checkpoint["samples"] = mcmc.get_samples()
            checkpoint["state"] = mcmc.last_state
            checkpoint["num_samples"] = (
                checkpoint["samples"][list(checkpoint["samples"].keys())[0]]
            ).size
            checkpoint["log_likelihood"] = log_likelihood(
                mcmc.sampler.model, mcmc.get_samples()
            )
            checkpoint["new_key"] = new_key

            logger.info(
                "checkpointing at "
                + str(checkpoint["num_samples"])
                + " samples"
            )
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

        else:
            logger.info("loading from checkpoint")
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)

        while checkpoint["num_samples"] < self.kwargs["num_total_samples"]:

            new_key, sample_key = jax.random.split(checkpoint["new_key"], 2)

            # fix checkpoint state as post warmup state
            mcmc.post_warmup_state = checkpoint["state"]
            mcmc.run(sample_key)

            new_samples = mcmc.get_samples()
            new_state = mcmc.last_state
            new_log_likelihoods = log_likelihood(
                mcmc.sampler.model, mcmc.get_samples()
            )

            for key in checkpoint["samples"].keys():
                checkpoint["samples"][key] = jnp.append(
                    checkpoint["samples"][key], new_samples[key]
                )

            checkpoint["log_likelihood"]["log_likelihood"] = jnp.append(
                checkpoint["log_likelihood"]["log_likelihood"],
                new_log_likelihoods["log_likelihood"],
            )
            checkpoint["state"] = new_state
            checkpoint["num_samples"] = (
                checkpoint["samples"][list(checkpoint["samples"].keys())[0]]
            ).size
            checkpoint["new_key"] = new_key

            logger.info(
                "checkpointing at "
                + str(checkpoint["num_samples"])
                + " samples"
            )
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

        return mcmc, checkpoint["samples"], checkpoint["log_likelihood"]


def validate_model(numpyro_model, rng_key):
    from numpyro import handlers
    from numpyro.infer.util import find_valid_initial_params, substitute, trace

    with handlers.seed(rng_seed=0):
        model = numpyro_model
        model_trace = trace(numpyro_model).get_trace()
        (init_params, pe, grad), is_valid = find_valid_initial_params(
            rng_key,
            substitute(
                model,
                data={
                    k: site["value"]
                    for k, site in model_trace.items()
                    if site["type"] in ["plate"]
                },
            ),
        )
        if np.isnan(float(pe)):
            raise ValueError("Initial energy contain NaNs")
        elif any(np.isnan([float(value) for value in grad.values()])):
            raise ValueError("Initial gradient contain NaNs")
        elif not is_valid:
            raise ValueError(f"Invalid initial params {init_params}")
