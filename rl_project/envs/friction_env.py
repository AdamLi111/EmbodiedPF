import sys
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add both project root and simulation dir to sys.path so bare imports
# inside simulation modules resolve correctly.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_simulation_dir = os.path.join(_project_root, "simulation")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _simulation_dir not in sys.path:
    sys.path.insert(0, _simulation_dir)

from simulation.world_model import WorldModel
from simulation.simulated_user import SimulatedUser
from simulation.simulated_vision import SimulatedVision
from simulation.action_parser import ActionParser
from simulation.task_evaluator import TaskEvaluator
from simulation.task_scenarios import get_task_scenarios

from rl_project.prompts import (
    ExecutePrompt,
    ProbingPrompt,
    AssumptionRevealPrompt,
    OverspecificationPrompt,
    ReflectivePausePrompt,
    ReinforcementPrompt,
)


# Friction action names for readability
ACTION_NAMES = [
    "execute",
    "probe",
    "assumption_reveal",
    "overspecification",
    "reflective_pause",
    "reinforcement",
]

# Ambiguity types for one-hot encoding
AMBIGUITY_TYPES = [
    "referential",
    "trajectory",
    "safety",
    "quantitative",
    "spatial_relation",
    "implicit_precondition",
    "orientation",
]


class RuleBasedUser:
    """Lightweight stand-in for SimulatedUser that requires no LLM API.
    Used when openai_api_key is not provided or set to 'test'.

    Never returns None on its own — episodes end only via TaskEvaluator
    success or max-turn truncation.  After a friction action it gives a
    clarifying hint (the first object's name); after an execute it just
    acknowledges.
    """

    def __init__(self):
        self.world_model = None
        self._turn = 0

    def reset(self, world_model):
        self.world_model = world_model
        self._turn = 0

    def generate_initial_command(self) -> str:
        goal = self.world_model.task_goal if self.world_model else "do the task"
        return f"Can you {goal.lower().rstrip('.')}?"

    def respond_to_robot(self, robot_message, robot_action_description=None,
                         task_complete=False):
        self._turn += 1
        if task_complete:
            return None
        # After friction, give a small clarifying hint
        if robot_action_description is None and self.world_model:
            objs = self.world_model.objects
            if objs:
                return f"I mean the {objs[0]['name']}."
        return "OK, keep going."

    def check_goal_progress(self):
        return (False, ["navigation"])


class FrictionDialogueEnv(gym.Env):
    """Gymnasium environment wrapping PONDER simulation for friction-based dialogue RL."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, openai_api_key: str = "test",
                 use_encoder: bool = False):
        super().__init__()

        self.config = config
        self.max_turns = config.get("env", {}).get("max_turns", 10)
        self.reward_cfg = config.get("reward", {})

        # Encoding dimensions
        self.embedding_dim = config.get("encoder", {}).get("embedding_dim", 384)
        self.num_ambiguity_types = len(AMBIGUITY_TYPES)
        # metadata: [turn_frac, num_visible_objects, num_hazards, <ambiguity one-hot>]
        self.metadata_dim = 3 + self.num_ambiguity_types
        self.obs_dim = self.embedding_dim * 3 + self.metadata_dim

        # Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # PONDER components
        self.scenarios = get_task_scenarios()
        self._use_llm = openai_api_key and openai_api_key != "test"
        if self._use_llm:
            self.simulated_user = SimulatedUser(openai_api_key=openai_api_key)
            self.prompt_handlers = {
                0: ExecutePrompt(openai_api_key),
                1: ProbingPrompt(openai_api_key),
                2: AssumptionRevealPrompt(openai_api_key),
                3: OverspecificationPrompt(openai_api_key),
                4: ReflectivePausePrompt(openai_api_key),
                5: ReinforcementPrompt(openai_api_key),
            }
        else:
            self.simulated_user = RuleBasedUser()
            self.prompt_handlers = None

        # State encoder (optional — loads sentence-transformer model)
        self._encoder = None
        if use_encoder:
            from rl_project.models.state_encoder import StateEncoder
            self._encoder = StateEncoder(config)

        # Per-episode state (initialized in reset)
        self.world_model = None
        self.scenario = None
        self.turn_number = 0
        self.conversation_history = []
        self.interaction_log = {"collision": None, "actions": [], "friction": []}
        self.last_user_command = ""
        self.vision_context = ""
        self.terminated = False
        self.truncated = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a random scenario
        self.scenario = random.choice(self.scenarios)
        scene = self.scenario["scene_structure"]
        goal = self.scenario["task_goal"]

        # Create world model and simulated user
        self.world_model = WorldModel(scene, goal)
        self.simulated_user.reset(self.world_model)

        # Generate initial vague command and vision context
        self.last_user_command = self.simulated_user.generate_initial_command()
        self.vision_context = SimulatedVision.generate_from_world_model(self.world_model)

        # Reset episode tracking
        self.turn_number = 0
        self.conversation_history = [{"role": "user", "content": self.last_user_command}]
        self.interaction_log = {"collision": None, "actions": [], "friction": []}
        self.terminated = False
        self.truncated = False

        obs = self._build_observation()
        info = self._build_info(reward_components={})
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert not self.terminated and not self.truncated, "Episode is done, call reset()"

        self.turn_number += 1
        reward = self.reward_cfg.get("per_turn_penalty", -0.05)
        friction_applied = action != 0
        friction_type = ACTION_NAMES[action] if friction_applied else None

        if friction_applied:
            self.interaction_log["friction"].append(
                {"turn": self.turn_number, "type": friction_type}
            )

        if action == 0:
            # Execute: generate a physical action via LLM (or placeholder)
            llm_result = self._generate_prompt(action)
            action_desc = ActionParser.parse_action(llm_result)
            self.interaction_log["actions"].append(action_desc)
            self.conversation_history.append(
                {"role": "assistant", "content": llm_result.get("text", "")}
            )

            # Apply the action to the world model
            collision = self.world_model.update_from_action(action_desc)
            if collision:
                self.interaction_log["collision"] = collision
                reward += self.reward_cfg.get("safety_violation", -5.0)

            # Evaluate success
            eval_result = TaskEvaluator.evaluate_task_success(
                self.world_model, self.scenario["task_goal"], self.interaction_log
            )

            if eval_result["success"]:
                reward += self.reward_cfg.get("task_success", 1.0)
                self.terminated = True
            else:
                # Get next user command
                user_response = self.simulated_user.respond_to_robot(
                    robot_message=llm_result.get("text", "I have executed the action."),
                    robot_action_description=action_desc,
                    task_complete=False,
                )
                if user_response is None:
                    reward += self.reward_cfg.get("task_success", 1.0)
                    self.terminated = True
                else:
                    self.last_user_command = user_response
                    self.conversation_history.append(
                        {"role": "user", "content": user_response}
                    )
        else:
            # Friction actions 1-5: generate utterance via LLM (or placeholder)
            llm_result = self._generate_prompt(action)
            friction_utterance = llm_result.get("text", f"[Friction:{friction_type}]")
            self.conversation_history.append(
                {"role": "assistant", "content": friction_utterance}
            )

            user_response = self.simulated_user.respond_to_robot(
                robot_message=friction_utterance,
                robot_action_description=None,
                task_complete=False,
            )
            if user_response is None:
                reward += self.reward_cfg.get("task_success", 1.0)
                self.terminated = True
            else:
                self.last_user_command = user_response
                self.conversation_history.append(
                    {"role": "user", "content": user_response}
                )

        # Check truncation
        if not self.terminated and self.turn_number >= self.max_turns:
            self.truncated = True
            reward += self.reward_cfg.get("task_failure", -1.0)

        # Update vision
        if not self.terminated and not self.truncated:
            self.vision_context = SimulatedVision.generate_from_world_model(
                self.world_model
            )

        obs = self._build_observation()
        info = self._build_info(
            reward_components={"friction_type_used": friction_type}
        )
        return obs, reward, self.terminated, self.truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_prompt(self, action: int) -> dict:
        """Call the appropriate prompt handler, or return a placeholder in test mode."""
        if self.prompt_handlers is not None:
            handler = self.prompt_handlers[action]
            return handler.generate(
                user_command=self.last_user_command,
                vision_context=self.vision_context,
                scene_description=self.world_model.get_full_state_description(),
                conversation_history=self.conversation_history,
            )
        # Test / rule-based mode — deterministic placeholders
        if action == 0:
            return self._placeholder_execute()
        return {
            "action": "clarify",
            "text": f"[{ACTION_NAMES[action]}] Could you clarify?",
            "friction_type": ACTION_NAMES[action],
        }

    def _placeholder_execute(self) -> dict:
        """Navigate toward the first scene object (rule-based mode)."""
        import math
        wm = self.world_model
        target = None
        if wm and wm.objects:
            target = wm.objects[0]

        if target is None:
            return {"action": "forward", "distance": 0.5, "text": "Moving forward.",
                    "target_object": None, "turn_degrees": 0,
                    "confidence": "medium", "friction_type": "none"}

        # Compute angle and distance to target
        dx = target["position"][0] - wm.robot_position[0]
        dy = target["position"][1] - wm.robot_position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        target_angle = math.degrees(math.atan2(dx, dy))  # 0=north convention
        turn = target_angle - wm.robot_orientation
        # Normalise to [-180, 180]
        turn = (turn + 180) % 360 - 180

        return {
            "action": "spatial_navigate",
            "distance": round(dist, 2),
            "text": f"Heading to {target['name']}.",
            "target_object": target["name"],
            "turn_degrees": round(turn, 1),
            "confidence": "medium",
            "friction_type": "none",
        }

    def _build_observation(self) -> np.ndarray:
        """Build observation: encoded text vectors (or zero placeholders) + metadata."""
        if self._encoder is not None:
            history_text = " ".join(
                f"{t['role']}: {t['content']}"
                for t in self.conversation_history[-6:]
            )
            enc_vec = self._encoder.encode(
                self.last_user_command,
                self.vision_context,
                history_text,
            )
            text_emb = enc_vec.cpu().numpy()
        else:
            text_emb = np.zeros(self.embedding_dim * 3, dtype=np.float32)

        # Metadata
        turn_frac = self.turn_number / self.max_turns
        num_visible = len(
            [o for o in self.world_model.objects if o.get("visible", True)]
        ) if self.world_model else 0
        num_hazards = len(self.world_model.hazards) if self.world_model else 0

        # Ambiguity one-hot
        ambiguity = self.scenario["expected_ambiguity"] if self.scenario else None
        ambiguity_oh = np.zeros(self.num_ambiguity_types, dtype=np.float32)
        if ambiguity in AMBIGUITY_TYPES:
            ambiguity_oh[AMBIGUITY_TYPES.index(ambiguity)] = 1.0

        metadata = np.array(
            [turn_frac, float(num_visible), float(num_hazards)],
            dtype=np.float32,
        )
        metadata = np.concatenate([metadata, ambiguity_oh])

        return np.concatenate([text_emb, metadata])

    def _build_info(self, reward_components: dict) -> dict:
        """Build info dict required by the spec."""
        # Quick success check
        if self.world_model and self.scenario:
            eval_result = TaskEvaluator.evaluate_task_success(
                self.world_model, self.scenario["task_goal"], self.interaction_log
            )
            task_success = eval_result["success"]
        else:
            task_success = False

        collision = self.interaction_log.get("collision") is not None

        return {
            "task_success": task_success,
            "task_failure": self.truncated,
            "safety_violation": collision,
            "applied_friction": len(self.interaction_log["friction"]) > 0,
            "ambiguity_resolved": task_success,
            "is_terminal": self.terminated or self.truncated,
            "expected_ambiguity": (
                self.scenario["expected_ambiguity"] if self.scenario else None
            ),
            "turn_number": self.turn_number,
            "friction_type_used": reward_components.get("friction_type_used"),
        }
