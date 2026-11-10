# Pilotes solo ONNX

`driver1` est le PPO réactif ; `driver2` est le PPO avec quatre observations.
Chaque dossier contient un agent autonome, son modèle opset 17 et de vrais poids externes.
Les modèles et les prétraitements sont identifiés dans `export.json` et dans
[`docs/results/e-18/selection/agents.json`](../docs/results/e-18/selection/agents.json).

La sélection maximise les finitions nominales en validation, puis retient la plus petite
seed en cas d’égalité. Elle utilise les modèles de fin de budget. Les résultats de toutes
les seeds restent dans le rapport ; la comparaison de mémoire n’utilise pas seulement
les deux modèles retenus.

Les fichiers générés `agent.py` ont été formatés avec Ruff après export, puis leur hash
a été mis à jour dans `export.json`. Ce formatage ne change pas le prétraitement.

Ces deux pilotes constituent la livraison solo actuelle. Le pilote SAC et le champion
multi-agent relèvent des étapes E-24 et E-26/E-31.

```bash
uv run python vendor/simulation/verify_submission.py --submission submission/driver1
uv run python vendor/simulation/verify_submission.py --submission submission/driver2
```
