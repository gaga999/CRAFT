"""Local forward/gradient checks; no training, rollouts, or run directories."""

import unittest
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from craft.agents.craft import CRAFTAgent
from craft.config import AgentConfig
from craft.losses.contrastive import contrastive_loss
from craft.models.tokens import MixerBlock
from craft.utils.checkpoint import pure_state
from craft.utils.random import make_random_engines


class MixerTests(unittest.TestCase):
    def test_convolution_matches_dense_outputs_and_gradients(self):
        def dense_reference(block, x):
            x = block.local(x)
            y = jnp.swapaxes(block.token_norm(x), -1, -2)
            first, second = block.token_mlp.layers
            y = nnx.gelu(y @ first.kernel[0] + first.bias[...])
            y = y @ second.kernel[0] + second.bias[...]
            return x + jnp.swapaxes(y, -1, -2)

        for tokens, width in ((5, 16), (6, 16), (7, 16), (6, 128)):
            with self.subTest(tokens=tokens, width=width):
                engine, _ = make_random_engines(0)
                block = MixerBlock(width, tokens, True, rngs=engine.jax)
                x = jax.random.normal(engine.task_key(0), (2, tokens, width))
                np.testing.assert_allclose(
                    block(x), dense_reference(block, x), atol=1e-6
                )

                def objective(model, inputs):
                    return jnp.square(model(inputs)).mean()

                def reference_objective(model, inputs):
                    return jnp.square(dense_reference(model, inputs)).mean()

                actual = nnx.grad(objective, argnums=(0, 1))(block, x)
                expected = nnx.grad(reference_objective, argnums=(0, 1))(block, x)
                self.assertEqual(
                    jax.tree.structure(actual), jax.tree.structure(expected)
                )
                for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                primitives = str(jax.make_jaxpr(block.token_mlp)(x))
                self.assertEqual(primitives.count("conv_general_dilated["), 2)
                self.assertNotIn("transpose[", primitives)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = asdict(AgentConfig(width=8, depth=2, latent_dim=4, vae_latent_dim=2))
        engine, cls.evaluation = make_random_engines(1)
        cls.initial = CRAFTAgent.create(3, 2, config, rngs=engine.jax, batch_size=3)
        cls.batch = {
            "observations": jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 9,
            "value_goals": jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 7 - 1,
            "action_chunks": jnp.sin(jnp.arange(30, dtype=jnp.float32)).reshape(
                3, 5, 2
            ),
        }

    def setUp(self):
        self.agent = nnx.clone(self.initial)

    def test_paired_score_matches_siglip_diagonal_and_loss(self):
        critic = self.agent.critic
        critic.log_scale[...] = jnp.log(jnp.array(3.0))
        critic.bias[...] = jnp.array(-2.0)
        s, a, g = (
            self.batch[k] for k in ("observations", "action_chunks", "value_goals")
        )
        phi, psi = critic(s, a, g)
        expected = 3 * (phi @ psi.T) - 2
        np.testing.assert_allclose(critic.score(s, a, g), jnp.diag(expected), atol=1e-6)
        loss, metrics = contrastive_loss(
            phi, psi, critic.log_scale[...], critic.bias[...]
        )
        labels = 2 * jnp.eye(3) - 1
        np.testing.assert_allclose(
            loss, -jax.nn.log_sigmoid(labels * expected).sum(-1).mean()
        )
        np.testing.assert_allclose(
            metrics["critic/positive"], jnp.diag(expected).mean()
        )

    def test_guidance_scale_bias_and_gradient_ownership(self):
        agent = self.agent
        agent.kl_weight = 0.0
        s, a, g = (
            self.batch[k] for k in ("observations", "action_chunks", "value_goals")
        )
        q = agent.posterior(agent.critic.features(s, a), agent.critic.encode(g))
        key = self.evaluation.task_key(0)

        def loss(prior):
            return agent.prior_objective(prior, self.batch, q, key)[0]

        agent.critic.log_scale[...] = jnp.log(jnp.array(2.0))
        first_loss, first_grad = nnx.value_and_grad(loss)(agent.prior)
        self.assertGreater(
            float(sum(jnp.square(x).sum() for x in jax.tree.leaves(first_grad))), 0
        )
        agent.critic.log_scale[...] = jnp.log(jnp.array(7.0))
        scaled_loss, scaled_grad = nnx.value_and_grad(loss)(agent.prior)
        agent.critic.bias[...] += 4
        before = pure_state(agent)
        shifted_loss, shifted_grad = nnx.value_and_grad(loss)(agent.prior)
        np.testing.assert_allclose(
            shifted_loss, scaled_loss - 4 * agent.guide_weight, atol=1e-6
        )
        self.assertTrue(np.isfinite(first_loss))
        for original, scaled, shifted in zip(
            jax.tree.leaves(first_grad),
            jax.tree.leaves(scaled_grad),
            jax.tree.leaves(shifted_grad),
        ):
            np.testing.assert_allclose(scaled, original * 3.5, rtol=3e-5, atol=1e-7)
            np.testing.assert_allclose(scaled, shifted, atol=1e-7)
            self.assertTrue(np.isfinite(scaled).all())
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(pure_state(agent))):
            np.testing.assert_array_equal(a, b)

    def test_log_gate_is_positive_and_dropout_blocks_condition(self):
        decoder = self.agent.decoder
        np.testing.assert_allclose(decoder.gate, 0.1, rtol=1e-6)
        hs = jnp.ones((3, 8))
        hg = hs * 2
        z = jnp.ones((3, 5, 2))
        off, on = jnp.zeros((3, 1, 1)), jnp.ones((3, 1, 1))
        for log_gate in (-5.0, 0.0, 2.0):
            decoder.log_gate[...] = jnp.array(log_gate)
            self.assertGreater(float(decoder.gate), 0)
            np.testing.assert_array_equal(
                decoder(z, hs, hg, off), decoder(z, hs + 3, hg - 4, off)
            )
        grad = nnx.grad(lambda d: jnp.square(d(z, hs, hg, on)).sum())(decoder)
        self.assertTrue(np.isfinite(grad.log_gate[...]))
        self.assertNotEqual(float(grad.log_gate[...]), 0)
