"""Per-episode metrics tracking and aggregation."""

from collections import defaultdict
import pandas as pd


class MetricsTracker:
    def __init__(self):
        self.episodes = []

    def add_episode(self, episode_info: dict):
        self.episodes.append(episode_info)

    def compute_summary(self) -> dict:
        if not self.episodes:
            return {"success_rate": 0, "avg_turns": 0,
                    "total_safety_violations": 0,
                    "success_by_ambiguity": {},
                    "friction_distribution": {}}

        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.get("task_success"))

        # Per-ambiguity success
        by_amb = defaultdict(lambda: {"total": 0, "success": 0})
        for e in self.episodes:
            amb = e.get("expected_ambiguity") or "none"
            by_amb[amb]["total"] += 1
            if e.get("task_success"):
                by_amb[amb]["success"] += 1
        success_by_amb = {
            k: v["success"] / max(v["total"], 1) for k, v in by_amb.items()
        }

        # Friction distribution
        friction_counts = defaultdict(int)
        for e in self.episodes:
            for ft, cnt in e.get("friction_actions", {}).items():
                friction_counts[ft] += cnt

        return {
            "success_rate": successes / n,
            "avg_turns": sum(e.get("num_turns", 0) for e in self.episodes) / n,
            "total_safety_violations": sum(
                1 for e in self.episodes if e.get("safety_violation")
            ),
            "success_by_ambiguity": dict(success_by_amb),
            "friction_distribution": dict(friction_counts),
            "num_episodes": n,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episodes)
