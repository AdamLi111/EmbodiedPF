"""Base class for all friction / execution prompt handlers."""

import json
from openai import OpenAI


class BaseFrictionPrompt:
    """Abstract base for prompt scripts that call an LLM to generate an
    utterance (friction) or a physical action (execute)."""

    def __init__(self, openai_api_key: str, model: str = "gpt-5-nano-2025-08-07"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    # ------------------------------------------------------------------
    # Public interface — subclasses must override
    # ------------------------------------------------------------------

    def generate(self, user_command: str, vision_context: str,
                 scene_description: str, conversation_history: list) -> dict:
        """Return a dict describing the robot's next action or utterance.

        For execution prompts the dict follows the action schema:
            {"action": str, "text": str, "friction_type": "none",
             "distance": float, "target_object": str|None,
             "turn_degrees": float, "confidence": str}

        For friction prompts the dict follows:
            {"action": "clarify", "text": str, "friction_type": str}
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_content: str) -> dict:
        """Call the OpenAI chat API, extract JSON from the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            # Extract the first JSON object from the response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(raw[json_start:json_end])

            # Fallback — return raw text wrapped in a dict
            return {"action": "unknown", "text": raw, "friction_type": "none"}
        except Exception as e:
            return {
                "action": "unknown",
                "text": f"LLM error: {e}",
                "friction_type": "none",
            }

    # ------------------------------------------------------------------
    # Helpers for building user content
    # ------------------------------------------------------------------

    @staticmethod
    def _format_user_content(user_command: str, vision_context: str,
                             scene_description: str,
                             conversation_history: list) -> str:
        """Build a single user-message string that every subclass can use."""
        history_str = ""
        if conversation_history:
            recent = conversation_history[-10:]
            history_str = "\n".join(
                f"  {turn['role']}: {turn['content']}" for turn in recent
            )

        return (
            f"User command: \"{user_command}\"\n\n"
            f"Vision context (what the robot currently sees):\n{vision_context}\n\n"
            f"Scene description: {scene_description}\n\n"
            f"Recent conversation history:\n{history_str}"
        )
