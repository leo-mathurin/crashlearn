# Pilotes solo ONNX

`driver1` est le PPO réactif ; `driver2` est le PPO avec quatre observations.
Chaque dossier contient un agent autonome, son modèle opset 17 et de vrais poids externes.
Les modèles et les prétraitements sont identifiés dans `export.json`. Résultats et
provenance : [driver1](../docs/results/driver1/README.md),
[driver2 et la campagne E-18](../docs/results/e-18/README.md).

Les fichiers générés `agent.py` ont été formatés avec Ruff après export, puis leur hash
a été mis à jour dans `export.json`. Ce formatage ne change pas le prétraitement.

Ces deux pilotes constituent la livraison solo actuelle. Le pilote SAC et le champion
multi-agent relèvent des étapes E-24 et E-26/E-31.

```bash
uv run python vendor/simulation/verify_submission.py --submission submission/driver1
uv run python vendor/simulation/verify_submission.py --submission submission/driver2
```
