"""Reward shaping from environment info dict."""


class RewardComputer:
    def __init__(self, config: dict):
        r = config.get("reward", {})
        self.task_success = r.get("task_success", 1.0)
        self.task_failure = r.get("task_failure", -1.0)
        self.safety_violation = r.get("safety_violation", -5.0)
        self.safety_correct_friction = r.get("safety_correct_friction", 0.1)
        self.per_turn_penalty = r.get("per_turn_penalty", -0.05)
        self.successful_disambiguation = r.get("successful_disambiguation", 0.2)

    def compute(self, env_info: dict) -> float:
        reward = self.per_turn_penalty

        if env_info.get("task_success"):
            reward += self.task_success
        elif env_info.get("task_failure"):
            reward += self.task_failure

        if env_info.get("safety_violation"):
            reward += self.safety_violation

        if env_info.get("friction_prevented_unsafe_action"):
            reward += self.safety_correct_friction

        if env_info.get("ambiguity_resolved") and env_info.get("applied_friction"):
            reward += self.successful_disambiguation

        return reward
