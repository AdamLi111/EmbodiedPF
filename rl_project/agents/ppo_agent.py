"""PPO agent — drives policy updates from rollout data."""

import torch
import torch.nn as nn


class PPOAgent:
    def __init__(self, policy, value_network, config: dict):
        self.policy = policy
        self.value_network = value_network
        ppo_cfg = config.get("ppo", {})
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.value_coef = ppo_cfg.get("value_coef", 0.5)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.update_epochs = ppo_cfg.get("update_epochs", 4)
        self.num_minibatches = ppo_cfg.get("num_minibatches", 4)

        lr = ppo_cfg.get("lr", 3e-4)
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_network.parameters()),
            lr=lr,
        )

    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        action, log_prob = self.policy.get_action(state)
        value = self.value_network(state).item()
        return action, log_prob, value

    def update(self, rollout_buffer) -> dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            for (states, actions, old_log_probs,
                 returns, advantages) in rollout_buffer.get_batches(
                     self.num_minibatches):

                # Normalise advantages
                adv = advantages
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                new_log_probs, entropy = self.policy.evaluate_action(
                    states, actions
                )
                values = self.value_network(states)

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (values - returns).pow(2).mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_network.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "num_updates": num_updates,
        }
