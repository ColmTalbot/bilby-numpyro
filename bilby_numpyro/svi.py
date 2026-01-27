# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import tqdm

import jax
from jax import jit, lax, random
import jax.numpy as jnp

from numpyro.infer.svi import SVI as _SVI, SVIRunResult, _make_loss_fn

SVIState = namedtuple("SVIState", ["optim_state", "mutable_state", "rng_key", "args", "kwargs"])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **optim_state** - current optimizer's state.
 - **mutable_state** - extra state to store values of `"mutable"` sites
 - **rng_key** - random number generator seed used for the iteration.
 - **kwargs** - additional keyword arguments used for the iteration.
 """


class SVI(_SVI):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    **References**

    1. *SVI Part I: An Introduction to Stochastic Variational Inference in Pyro*,
       (http://pyro.ai/examples/svi_part_i.html)

    **Example:**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.distributions import constraints
        >>> from numpyro.infer import Predictive, SVI, Trace_ELBO

        >>> def model(data):
        ...     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
        ...     with numpyro.plate("N", data.shape[0] if data is not None else 10):
        ...         numpyro.sample("obs", dist.Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
        ...     beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
        ...                            constraint=constraints.positive)
        ...     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
        >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
        >>> svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        >>> svi_result = svi.run(random.PRNGKey(0), 2000, data)
        >>> params = svi_result.params
        >>> inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])
        >>> # use guide to make predictive
        >>> predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
        >>> samples = predictive(random.PRNGKey(1), data=None)
        >>> # get posterior samples
        >>> predictive = Predictive(guide, params=params, num_samples=1000)
        >>> posterior_samples = predictive(random.PRNGKey(1), data=None)
        >>> # use posterior samples to make predictive
        >>> predictive = Predictive(model, posterior_samples, params=params, num_samples=1000)
        >>> samples = predictive(random.PRNGKey(1), data=None)

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: An instance of :class:`~numpyro.optim._NumpyroOptim`, a
        ``jax.example_libraries.optimizers.Optimizer`` or an Optax
        ``GradientTransformation``. If you pass an Optax optimizer it will
        automatically be wrapped using :func:`numpyro.optim.optax_to_numpyro`.

            >>> from optax import adam, chain, clip
            >>> svi = SVI(model, guide, chain(clip(10.0), adam(1e-3)), loss=Trace_ELBO())

    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    def init(self, rng_key, *args, init_params=None, **kwargs):
        """
        Gets the initial SVI state.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param dict init_params: if not None, initialize :class:`numpyro.param` sites with values from
            this dictionary instead of using ``init_value`` in :class:`numpyro.param` primitives.
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: the initial :data:`SVIState`
        """
        state = super().init(rng_key, *args, init_params=init_params, **kwargs)

        return SVIState(state.optim_state, state.mutable_state, rng_key, args, kwargs)

    def make_update(self, forward_mode_differentiation=False, stable_update=False):
        if stable_update:
            update_func = self.optim.eval_and_stable_update
        else:
            update_func = self.optim.eval_and_update

        @jit
        def update(svi_state, _):
            rng_key, rng_key_step = random.split(svi_state.rng_key)
            loss_fn = _make_loss_fn(
                self.loss,
                rng_key_step,
                self.constrain_fn,
                self.model,
                self.guide,
                svi_state.args,
                svi_state.kwargs,
                self.static_kwargs,
                mutable_state=svi_state.mutable_state,
            )
            (loss_val, mutable_state), optim_state = update_func(
                loss_fn,
                svi_state.optim_state,
                forward_mode_differentiation=forward_mode_differentiation,
            )
            svi_state.kwargs["beta"] **= 0.9993
            return SVIState(optim_state, mutable_state, rng_key, svi_state.args, svi_state.kwargs), loss_val

        return update

    def run(
        self,
        rng_key,
        num_steps,
        *args,
        progress_bar=True,
        stable_update=False,
        forward_mode_differentiation=False,
        init_state=None,
        init_params=None,
        update_func=None,
        **kwargs,
    ):
        """
        (EXPERIMENTAL INTERFACE) Run SVI with `num_steps` iterations, then return
        the optimized parameters and the stacked losses at every step. If `num_steps`
        is large, setting `progress_bar=False` can make the run faster.

        .. note:: For a complex training process (e.g. the one requires early stopping,
            epoch training, varying args/kwargs,...), we recommend to use the more
            flexible methods :meth:`init`, :meth:`update`, :meth:`evaluate` to
            customize your training procedure.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param int num_steps: the number of optimization steps.
        :param args: arguments to the model / guide
        :param bool progress_bar: Whether to enable progress bar updates. Defaults to
            ``True``.
        :param bool stable_update: whether to use :meth:`stable_update` to update
            the state. Defaults to False.
        :param bool forward_mode_differentiation: whether to use forward-mode differentiation
            or reverse-mode differentiation. By default, we use reverse mode but the forward
            mode can be useful in some cases to improve the performance. In addition, some
            control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
            only supports forward-mode differentiation. See
            `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
            for more information.
        :param SVIState init_state: if not None, begin SVI from the
            final state of previous SVI run. Usage::

                svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
                svi_result = svi.run(random.PRNGKey(0), 2000, data)
                # upon inspection of svi_result the user decides that the model has not converged
                # continue from the end of the previous svi run rather than beginning again from iteration 0
                svi_result = svi.run(random.PRNGKey(1), 2000, data, init_state=svi_result.state)

        :param dict init_params: if not None, initialize :class:`numpyro.param` sites with values from
            this dictionary instead of using ``init_value`` in :class:`numpyro.param` primitives.
        :param kwargs: keyword arguments to the model / guide
        :return: a namedtuple with fields `params` and `losses` where `params`
            holds the optimized values at :class:`numpyro.param` sites,
            and `losses` is the collected loss during the process.
        :rtype: :data:`SVIRunResult`
        """

        if num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")

        if update_func is None:
            update_func = self.make_update(forward_mode_differentiation, stable_update)

        if init_state is None:
            svi_state = self.init(rng_key, *args, init_params=init_params, **kwargs)
        else:
            svi_state = init_state
        if progress_bar:
            losses = []
            with tqdm.trange(1, num_steps + 1) as t:
                batch = max(num_steps // 20, 1)
                for i in t:
                    svi_state, loss = update_func(svi_state, None)
                    losses.append(jax.device_get(loss))
                    if i % batch == 0:
                        if stable_update:
                            valid_losses = [x for x in losses[i - batch :] if x == x]
                            num_valid = len(valid_losses)
                            if num_valid == 0:
                                avg_loss = float("nan")
                            else:
                                avg_loss = sum(valid_losses) / num_valid
                        else:
                            avg_loss = sum(losses[i - batch :]) / batch
                        t.set_postfix_str(
                            f"init loss: {losses[0]:.4f}, avg. loss [{i - batch + 1}-{i}]: "
                            f"{avg_loss:.4f}, temperature: {svi_state.kwargs.get('beta', float('nan')):.4f}",
                            refresh=False,
                        )
            losses = jnp.stack(losses)
        else:
            svi_state, losses = lax.scan(update_func, svi_state, None, length=num_steps)

        # XXX: we also return the last svi_state for further inspection of both
        # optimizer's state and mutable state.
        return SVIRunResult(self.get_params(svi_state), svi_state, losses)
