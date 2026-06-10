"""Reinforcement friction prompt — restate key information for emphasis.

Reference: Inan et al. (2025) — "The speaker restates their own previous
utterance for emphasis, rewinding the flow. Similar to repetition in discourse
(Tannen, 1989)."
"""

from .base_prompt import BaseFrictionPrompt

SYSTEM_PROMPT = """\
You are Misty, a mobile robot. The RL policy has decided to apply \
REINFORCEMENT friction — you must restate or summarise key information from the \
conversation so far, emphasising constraints or details that affect task \
success. This helps ensure shared understanding before acting.

# Instructions
1. Review the conversation history and current command.
2. Restate the most important detail(s) — target identity, safety constraints, \
   path plan, or preconditions.
3. Keep it to ONE or TWO sentences.
4. Frame it as confirmation/emphasis, not a question.

# Examples
- User: "Go to the red cup" (after earlier mentioning "the one on the left")
  → {"action":"clarify","text":"Just to confirm—you want the red cup, the one on the left side of the table.","friction_type":"reinforcement"}
- User: "Move forward" (after robot warned about an obstacle)
  → {"action":"clarify","text":"Right, moving forward—I'll keep in mind the obstacle I flagged earlier at about 2 meters.","friction_type":"reinforcement"}
- User: "Go to the plant behind me" (plant requires turning)
  → {"action":"clarify","text":"Just to confirm—you want me to go to the plant behind me, so I'll need to turn around first.","friction_type":"reinforcement"}
- User: "Yes, the left one" (after disambiguation)
  → {"action":"clarify","text":"Got it—the left bottle it is. I'll head there now.","friction_type":"reinforcement"}
- User: "Go around it on the right" (after obstacle warning)
  → {"action":"clarify","text":"Understood—I'll go around the obstacle on the right side and then continue to the target.","friction_type":"reinforcement"}

# Output format — return ONLY valid JSON:
{"action": "clarify", "text": "<your reinforcement statement>", "friction_type": "reinforcement"}
"""


class ReinforcementPrompt(BaseFrictionPrompt):
    """Generate a reinforcement utterance."""

    def generate(self, user_command, vision_context, scene_description,
                 conversation_history):
        user_content = self._format_user_content(
            user_command, vision_context, scene_description,
            conversation_history,
        )
        result = self._call_llm(SYSTEM_PROMPT, user_content)
        result["action"] = "clarify"
        result["friction_type"] = "reinforcement"
        result.setdefault("text", "Just to confirm what we discussed.")
        return result
