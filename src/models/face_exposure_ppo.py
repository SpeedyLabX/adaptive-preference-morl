"""General-dimension on-policy categorical PPO with PopArt vector critic (no hindsight relabeling).

Reuses the frozen ActorCritic, scalar_advantage, clipped_policy_loss, and gae_advantages primitives
from src/models/onpolicy.py and src/models/lunar_onpolicy.py without mutating frozen source hashes.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.lunar_onpolicy import gae_advantages
from src.models.onpolicy import PPO, ActorCritic, clipped_policy_loss, scalar_advantage


class SimplexPPO(PPO):
    """Categorical PPO with faithful raw-utility scalarization, PopArt critic, and action-dim-aware entropy."""

    def __init__(self, model: ActorCritic, plan: dict[str, Any]):
        super().__init__(model, plan)
        self.action_dim = int(model.actor.head.out_features)
        self.gae_lambda = float(plan.get("gae_lambda", 0.95))

    def update(self, batch: dict[str, torch.Tensor], step: int) -> dict[str, float]:
        p, model = self.plan, self.model
        advantages = gae_advantages(
            batch["rewards"],
            batch["values"],
            batch["next_values"],
            batch["terminated"],
            batch["truncated"],
            float(p["gamma"]),
            self.gae_lambda,
        )
        targets = advantages + batch["values"]
        flat = {key: value.flatten(0, 1) for key, value in batch.items()}
        targets = targets.flatten(0, 1)

        # Raw-utility-preserving scalarization (no PopArt objective-wise sigma division in actor advantage).
        advantage = scalar_advantage(targets, flat["values"], flat["preferences"])
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)

        model.update_popart(targets, float(p["popart_rate"]), float(p["popart_std_floor"]))
        normalized_targets = (targets - model.mean) / model.std

        max_entropy = math.log(self.action_dim)
        entropy_start = float(p.get("entropy_start", max_entropy))
        entropy_end = float(p["entropy_end"])
        target_entropy = entropy_start + min(step / float(p["training_interactions"]), 1.0) * (
            entropy_end - entropy_start
        )

        diagnostics = []
        for _ in range(int(p["epochs"])):
            order = torch.randperm(len(targets), device=targets.device)
            for index in order.tensor_split(int(p["minibatches"])):
                s, w = flat["states"][index], flat["preferences"][index]
                for _ in range(int(p["critic_steps_per_actor"])):
                    loss_c = (model.critic(s, w) - normalized_targets[index]).square().mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(
                        model.critic.parameters(), float(p["max_grad_norm"]), error_if_nonfinite=True
                    )
                    self.critic_optimizer.step()
                    self.critic_updates += 1

                distribution = model.distribution(s, w)
                logp = distribution.log_prob(flat["actions"][index])
                loss_pi = clipped_policy_loss(logp, flat["logp"][index], advantage[index], float(p["clip"]))
                entropy = distribution.entropy().mean()
                error = target_entropy - float(entropy.detach())
                coefficient = max(0.0, self.multiplier + float(p["entropy_damping"]) * error)
                loss = loss_pi - coefficient * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), float(p["max_grad_norm"]), error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.multiplier = max(0.0, self.multiplier + float(p["entropy_multiplier_lr"]) * error)
                self.actor_updates += 1
                diagnostics.append(torch.stack([loss_pi.detach(), loss_c.detach(), entropy.detach()]))

        means = torch.stack(diagnostics).mean(0).cpu().tolist()
        if not all(math.isfinite(x) for x in [*means, self.multiplier]):
            raise FloatingPointError("Nonfinite SimplexPPO diagnostics")
        return dict(zip(["policy_loss", "value_loss", "entropy"], means, strict=True)) | {
            "entropy_target": target_entropy,
            "entropy_multiplier": self.multiplier,
        }
