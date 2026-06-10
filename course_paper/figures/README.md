# Figures needed

Replace the following placeholders with your real artifacts (PDF preferred):

- `motivation.pdf` — the "with vs without friction" comparison figure (reuse Fig. 1 from PONDER).
- `pipeline.pdf` — end-to-end block diagram: MiniLM → 1162-dim observation → policy → action → LLM → world model → reward.
- `friction_distribution.pdf` — bar chart of friction-type usage per policy. Can be generated from `runs_full_500/friction_dist.json`.

The learning-curve figure in `sections/05_experiments.tex` is drawn with TikZ/pgfplots directly — replace the coordinates with TensorBoard exports from `runs_full_500/`.
