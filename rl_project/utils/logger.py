"""TensorBoard + console logger."""

import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "runs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode_info: dict, step: int):
        self.log_scalar("episode/reward", episode_info.get("reward", 0), step)
        self.log_scalar("episode/turns", episode_info.get("turn_number", 0), step)
        if episode_info.get("task_success"):
            self.log_scalar("episode/success", 1.0, step)
        else:
            self.log_scalar("episode/success", 0.0, step)
        if episode_info.get("safety_violation"):
            self.log_scalar("episode/safety_violation", 1.0, step)

    def log_update(self, update_info: dict, step: int):
        for key, val in update_info.items():
            if isinstance(val, (int, float)):
                self.log_scalar(f"train/{key}", val, step)

    def close(self):
        self.writer.close()
