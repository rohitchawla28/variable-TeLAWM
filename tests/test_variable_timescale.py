"""CPU smoke tests with synthetic trajectories; no OGBench rollouts or training runs."""

import importlib
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict, from_modules

from agents.modules.actor_critic import Actor, QNetwork, VNetwork
from agents.modules.lam import LatentActionModel
from agents.telawm import TeLAWM
from common.buffer import Buffer
from common.utils.parser import cfg_to_dataclass


def config(conditioned=True):
    cfg = cfg_to_dataclass(OmegaConf.load(Path(__file__).resolve().parents[1] / 'config.yaml'))
    cfg.agent, cfg.obs_shape, cfg.action_dim = 'telawm', (2,), 2
    cfg.compile = cfg.cudagraphs = False
    cfg.geom_sample_value = cfg.geom_sample_policy = False
    cfg.pi_subgoal = False
    cfg.reward = 'unsupervised'
    cfg.batch_size, cfg.data_episode_length = 1024, 8
    cfg.arch_mlp['h_dim'] = 16
    cfg.telawm.update(k_conditioned=conditioned, train_k_values=[2, 4, 7], lam_timescale=2,
                      h_dim=16, latent_dim=2, n_sg_samples=8, pretrained_vf=True, pretrained_policy=True)
    return cfg


def cpu_agent(cfg, directory):
    """Use the real agent initialization and frozen IQL loading, placing its networks on CPU."""
    cfg.vf_chkpt = cfg.policy_chkpt = str(Path(directory) / 'iql.pt')
    torch.save(dict(vf=VNetwork(cfg).state_dict(), pi=Actor(cfg).state_dict(),
                    qfs=from_modules(QNetwork(cfg), QNetwork(cfg), as_module=True).state_dict()), cfg.vf_chkpt)
    with ExitStack() as stack:
        for cls in (Actor, QNetwork, VNetwork, LatentActionModel):
            stack.enter_context(patch(f'agents.telawm.{cls.__name__}',
                side_effect=lambda cfg, device=None, cls=cls: cls(cfg, 'meta' if device == 'meta' else 'cpu')))
        agent = TeLAWM(cfg)
    agent.device = torch.device('cpu')
    return agent


class VariableTimescaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_pairs_stay_in_episode_and_have_exact_k(self):
        for conditioned in (False, True):
            for lazy in (False, True):
                with self.subTest(conditioned=conditioned, lazy=lazy):
                    cfg = config(conditioned)
                    cfg.buffer_lazy_gather = lazy
                    # Each observation records its episode ID and time index.
                    obs = torch.stack([torch.arange(4).repeat_interleave(9), torch.arange(9).repeat(4)], dim=-1).float()
                    data = TensorDict(dict(obs=obs, action=torch.zeros_like(obs)), batch_size=[len(obs)])
                    buffer = Buffer(cfg, None, len(obs))
                    buffer._device = torch.device('cpu')
                    with patch('torch.cuda.mem_get_info', return_value=(0, 0)):
                        buffer.load(data)
                    batch = buffer.sample()
                    start = buffer.normalizer.unnormalize(batch['obs'])
                    end = buffer.normalizer.unnormalize(batch['lam_next_obs'])
                    k = batch['lam_k'].flatten()
                    torch.testing.assert_close(start[:, 0], end[:, 0])
                    torch.testing.assert_close(end[:, 1] - start[:, 1], k.float())
                    self.assertTrue((end[:, 1] <= 8).all())
                    expected = cfg.telawm['train_k_values'] if conditioned else [2]
                    self.assertEqual(sorted(k.unique().tolist()), expected)
                    if conditioned:
                        counts = torch.stack([(k == value).sum() for value in expected])
                        self.assertTrue((counts > cfg.batch_size / len(expected) * 0.75).all())
                        self.assertTrue((start[k == 2, 1] >= 5).any())

    def test_k_reaches_all_networks_and_loss_gradients(self):
        cfg = config()
        lam = LatentActionModel(cfg)
        obs, target = torch.randn(6, 2), torch.randn(6, 2)
        k = torch.tensor([2, 4, 7, 2, 4, 7]).view(-1, 1)
        inputs = {}
        handles = [module.register_forward_pre_hook(
            lambda module, args, name=name: inputs.update({name: args[0].detach().clone()}))
            for name, module in [('posterior', lam.encoder), ('decoder', lam.decoder), ('prior', lam.prior_policy)]]
        losses = lam.losses(obs, target, k=k)
        for handle in handles:
            handle.remove()
        for value in inputs.values():
            torch.testing.assert_close(value[:, -1], k.flatten().float() / 7)
        for metric in ('recon', 'kl'):
            torch.testing.assert_close(losses[f'lam_{metric}'],
                torch.stack([losses[f'lam_{metric}_k{value}'] for value in [2, 4, 7]]).mean())
        losses['lam_loss'].backward()
        for module in (lam.encoder, lam.decoder, lam.prior_policy):
            self.assertTrue(torch.isfinite(module[0].weight.grad).all())
            self.assertGreater(module[0].weight.grad[:, -1].abs().sum().item(), 0)
        # Missing horizons in a small batch must not produce NaN diagnostics.
        self.assertTrue(all(torch.isfinite(v) for v in lam.losses(obs, target, k=torch.full_like(k, 2)).values()))
        scalar_losses = lam.losses(obs, target, k=2)
        self.assertEqual(scalar_losses['lam_count_k2'].item(), len(obs))

    def test_checkpoint_planning_and_frozen_iql(self):
        with TemporaryDirectory() as directory:
            agent = cpu_agent(config(), directory)
            frozen = {name: {key: value.detach().clone() for key, value in getattr(agent, name).named_parameters()}
                      for name in ('pi', 'vf', 'qnet_params')}
            batch = TensorDict(dict(obs=torch.randn(6, 2), lam_next_obs=torch.randn(6, 2),
                                    lam_k=torch.tensor([2, 4, 7, 2, 4, 7]).view(-1, 1)), batch_size=[6])
            agent.update(batch, 0)  # One synthetic optimizer smoke step, not an experiment.
            for name, state in frozen.items():
                for key, value in state.items():
                    torch.testing.assert_close(dict(getattr(agent, name).named_parameters())[key], value, rtol=0, atol=0)
            checkpoint = Path(directory) / 'latest.pt'
            agent.save(checkpoint)
            cfg = config()
            cfg.telawm['train_k_values'] = [1, 100]  # loading must restore the original scale and supported K
            restored = cpu_agent(cfg, directory)
            restored.load(checkpoint)
            self.assertEqual(restored.lam.k_scale, 7)
            self.assertEqual(restored.lam.train_k_values, [2, 4, 7])
            obs, goal = torch.zeros(1, 2), torch.ones(1, 2)
            for repeat in (False, True):
                restored.latent_action_repeat = repeat
                for k in [2, 4, 7]:
                    restored.lam_timescale = k
                    with patch.object(restored.lam_detach, 'sample_policy', wraps=restored.lam_detach.sample_policy) as prior, \
                         patch.object(restored.lam_detach, 'forward_dynamics', wraps=restored.lam_detach.forward_dynamics) as decoder:
                        subgoal = restored._plan_with_lam(obs, obs, goal)
                    self.assertTrue(torch.isfinite(subgoal).all())
                    self.assertEqual(decoder.call_count, 2)
                    self.assertEqual(prior.call_count, 1 if repeat else 2)
                    self.assertTrue(all(call.kwargs['k'] == k for call in prior.call_args_list + decoder.call_args_list))
            torch.testing.assert_close(restored.lam(obs, goal, True, 2)['next_obs_pred'],
                                       agent.lam(obs, goal, True, k=2)['next_obs_pred'])

    def test_original_checkpoint_without_conditioning_metadata(self):
        with TemporaryDirectory() as directory:
            agent = cpu_agent(config(False), directory)
            checkpoint = Path(directory) / 'original.pt'
            agent.save(checkpoint)
            state = torch.load(checkpoint)
            del state['lam_conditioning']
            torch.save(state, checkpoint)
            restored = cpu_agent(config(False), directory)
            restored.load(checkpoint)
            obs = torch.randn(3, 2)
            expected = agent.lam(obs, obs, True)['next_obs_pred']
            torch.testing.assert_close(restored.lam(obs, obs, True)['next_obs_pred'], expected)
            torch.testing.assert_close(restored.lam(obs, obs, True, k=100)['next_obs_pred'], expected)
            self.assertEqual(restored.lam.encoder[0].in_features, 4)


class EvaluationSweepTests(unittest.TestCase):
    def test_one_seed_per_k_and_video_outputs(self):
        # Exercise the entry point without MuJoCo or video encoding.
        with patch.dict('sys.modules', {'common.ogbench_wrapper': SimpleNamespace(make_env=None)}):
            evaluation = importlib.import_module('evaluate')
        with TemporaryDirectory() as directory:
            cfg = config()
            cfg.seed = 17
            cfg.eval_log_dir = directory
            cfg.eval_checkpoint = Path(directory) / 'latest.pt'
            cfg.eval_checkpoint.touch()
            cfg.eval_k_sweep = True
            cfg.num_eval_episodes, cfg.num_video_episodes = 3, 1
            cfg.save_video = True
            agent = SimpleNamespace(lam=SimpleNamespace(k_conditioned=True, train_k_values=[2, 4, 7]), load=Mock())
            calls = []

            def rollout(eval_cfg, env, current_agent):
                calls.append((current_agent.lam_timescale, eval_cfg.seed, eval_cfg.num_eval_episodes, eval_cfg.num_video_episodes))
                self.assertIs(current_agent, agent)
                return dict(overall_success=0.5, overall_episode_reward=-10.0), ['video frames']

            with patch.object(evaluation, 'parse_cfg', return_value=cfg), \
                 patch.object(evaluation.torch.cuda, 'is_available', return_value=True), \
                 patch.object(evaluation, 'make_env'), \
                 patch.object(evaluation, 'make_agent', return_value=agent), \
                 patch.object(evaluation, 'set_seed') as seed, \
                 patch.object(evaluation, 'eval_agent', side_effect=rollout), \
                 patch.object(evaluation, 'frames_list_to_array', return_value=np.zeros((1, 3, 2, 2))), \
                 patch.object(evaluation, 'ImageSequenceClip') as clip:
                evaluation.evaluate.__wrapped__(cfg)
            agent.load.assert_called_once_with(cfg.eval_checkpoint)
            self.assertEqual(calls, [(2, 17, 3, 1), (4, 17, 3, 1), (7, 17, 3, 1)])
            self.assertEqual(seed.call_args_list, [call(17)] * 4)  # initialization, then each K
            self.assertEqual(clip.return_value.write_videofile.call_args_list,
                [call(str(Path(directory) / f'k{k}' / 'eval_video.mp4')) for k in [2, 4, 7]])
            for k in [2, 4, 7]:
                metrics = (Path(directory) / f'k{k}' / 'metrics.txt').read_text()
                self.assertIn(f'lam_timescale: {k}', metrics)
                self.assertIn('seed: 17', metrics)
                self.assertIn('overall_success: 0.5', metrics)


if __name__ == '__main__':
    unittest.main()
