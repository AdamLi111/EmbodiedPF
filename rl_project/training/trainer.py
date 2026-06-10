"""Main training loop: collect rollouts, update PPO, evaluate, checkpoint."""

import os
import time
import torch
import numpy as np


class Trainer:
    def __init__(self, env, agent, reward_computer, config, logger):
        self.env = env
        self.agent = agent
        self.rc = reward_computer
        self.config = config
        self.logger = logger

        ppo = config.get("ppo", {})
        self.total_timesteps = ppo.get("total_timesteps", 500_000)
        self.rollout_steps = ppo.get("rollout_steps", 2048)
        self.gamma = ppo.get("gamma", 0.99)
        self.gae_lambda = ppo.get("gae_lambda", 0.95)

        tr = config.get("training", {})
        self.log_interval = tr.get("log_interval", 10)
        self.eval_interval = tr.get("eval_interval", 50)
        self.eval_episodes = tr.get("eval_episodes", 100)
        self.save_interval = tr.get("save_interval", 100)
        self.checkpoint_dir = tr.get("checkpoint_dir", "checkpoints/")

    def train(self):
        from rl_project.agents.rollout_buffer import RolloutBuffer

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        obs, info = self.env.reset()
        state = torch.as_tensor(obs, dtype=torch.float32)

        global_step = 0
        iteration = 0
        ep_rewards = []
        ep_reward = 0.0

        train_start = time.time()
        time_env = 0.0      # time in env.step + env.reset
        time_encode = 0.0   # time building observations (encoder)
        time_ppo = 0.0      # time in PPO update
        time_eval = 0.0     # time in evaluation

        while global_step < self.total_timesteps:
            buf = RolloutBuffer()
            iteration += 1

            # ---- collect rollout ----
            for _ in range(self.rollout_steps):
                action, log_prob, value = self.agent.select_action(state)

                t0 = time.time()
                obs, _env_reward, terminated, truncated, info = self.env.step(action)
                time_env += time.time() - t0

                reward = self.rc.compute(info)
                done = terminated or truncated
                buf.add(state, action, reward, done, log_prob, value)

                ep_reward += reward
                global_step += 1

                if done:
                    ep_rewards.append(ep_reward)
                    self.logger.log_episode(
                        {"reward": ep_reward, **info}, global_step
                    )
                    ep_reward = 0.0
                    t0 = time.time()
                    obs, info = self.env.reset()
                    time_env += time.time() - t0

                state = torch.as_tensor(obs, dtype=torch.float32)

                if global_step >= self.total_timesteps:
                    break

            # ---- compute advantages ----
            with torch.no_grad():
                last_value = self.agent.value_network(state).item()
            buf.compute_returns_and_advantages(
                last_value, self.gamma, self.gae_lambda
            )

            # ---- PPO update ----
            t0 = time.time()
            update_info = self.agent.update(buf)
            time_ppo += time.time() - t0
            self.logger.log_update(update_info, global_step)

            # ---- logging ----
            if iteration % self.log_interval == 0:
                elapsed = time.time() - train_start
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                remaining = ((self.total_timesteps - global_step) / steps_per_sec
                             if steps_per_sec > 0 else 0)
                recent = ep_rewards[-20:] if ep_rewards else [0]
                mean_r = np.mean(recent)
                print(
                    f"[iter {iteration}  step {global_step}/{self.total_timesteps}]  "
                    f"mean_reward={mean_r:.3f}  "
                    f"policy_loss={update_info['policy_loss']:.4f}  "
                    f"value_loss={update_info['value_loss']:.4f}  "
                    f"entropy={update_info['entropy']:.4f}  "
                    f"| {steps_per_sec:.1f} steps/s  "
                    f"elapsed={self._fmt_time(elapsed)}  "
                    f"ETA={self._fmt_time(remaining)}"
                )

            # ---- evaluation ----
            if iteration % self.eval_interval == 0:
                t0 = time.time()
                self._run_eval(global_step)
                time_eval += time.time() - t0
                # Re-sync env state after eval used the same env
                obs, info = self.env.reset()
                state = torch.as_tensor(obs, dtype=torch.float32)
                ep_reward = 0.0

            # ---- checkpoint ----
            if iteration % self.save_interval == 0:
                self._save_checkpoint(iteration, global_step)

        # Final save
        self._save_checkpoint(iteration, global_step, final=True)

        total_time = time.time() - train_start
        print(f"\nTraining complete. {global_step} total steps, "
              f"{len(ep_rewards)} episodes, {self._fmt_time(total_time)} total.")
        print(f"  Time breakdown:")
        print(f"    Environment (step+reset): {self._fmt_time(time_env)} "
              f"({100*time_env/total_time:.0f}%)")
        print(f"    PPO updates:              {self._fmt_time(time_ppo)} "
              f"({100*time_ppo/total_time:.0f}%)")
        print(f"    Evaluation:               {self._fmt_time(time_eval)} "
              f"({100*time_eval/total_time:.0f}%)")
        other = total_time - time_env - time_ppo - time_eval
        print(f"    Other (logging, buffer):  {self._fmt_time(other)} "
              f"({100*other/total_time:.0f}%)")

    def _run_eval(self, global_step):
        from rl_project.evaluation.evaluator import Evaluator

        evaluator = Evaluator(self.env, self.agent, self.config)
        tracker = evaluator.evaluate(self.eval_episodes)
        summary = tracker.compute_summary()
        print(f"  [eval]  success={summary['success_rate']:.2%}  "
              f"avg_turns={summary['avg_turns']:.1f}  "
              f"safety_violations={summary['total_safety_violations']}")
        self.logger.log_scalar("eval/success_rate", summary["success_rate"],
                               global_step)
        self.logger.log_scalar("eval/avg_turns", summary["avg_turns"],
                               global_step)

    @staticmethod
    def _fmt_time(seconds):
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def _save_checkpoint(self, iteration, global_step, final=False):
        tag = "final" if final else f"iter_{iteration}"
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "iteration": iteration,
            "global_step": global_step,
            "policy_state_dict": self.agent.policy.state_dict(),
            "value_state_dict": self.agent.value_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")
