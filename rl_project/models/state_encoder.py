"""Sentence-transformer encoder that converts text observations into fixed-size
vectors for the RL policy.  Weights are frozen during training."""

import torch
from sentence_transformers import SentenceTransformer


class StateEncoder:
    """Encode (command, scene, history) texts into a single concatenated vector."""

    def __init__(self, config: dict):
        enc_cfg = config.get("encoder", {})
        model_name = enc_cfg.get("model_name", "all-MiniLM-L6-v2")
        self.embedding_dim = enc_cfg.get("embedding_dim", 384)
        self.freeze = enc_cfg.get("freeze", True)

        self.model = SentenceTransformer(model_name)
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode(self, utterance: str, scene_description: str,
               conversation_history: str) -> torch.Tensor:
        """Encode three text inputs and concatenate into shape (embedding_dim*3,)."""
        vecs = self.model.encode(
            [utterance, scene_description, conversation_history],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # vecs: (3, embedding_dim) → flatten to (embedding_dim*3,)
        return vecs.reshape(-1)
