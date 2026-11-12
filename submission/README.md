# Pilotes solo ONNX

`driver1` est le PPO réactif, réentraîné sous météo changeante (E-23) ; `driver2` est le
PPO avec quatre observations. Chaque dossier contient un agent autonome, son modèle
opset 17 et de vrais poids externes. Les modèles et les prétraitements sont identifiés
dans `export.json`. Résultats et provenance : [driver1](../docs/results/e-23/README.md)
(modèle précédent : [E-19](../docs/results/driver1/README.md)),
[driver2 et la campagne E-18](../docs/results/e-18/README.md).

Les fichiers générés `agent.py` ont été formatés avec Ruff après export, puis leur hash
a été mis à jour dans `export.json`. Ce formatage ne change pas le prétraitement.

Ces deux pilotes constituent la livraison solo actuelle. Le pilote SAC et le champion
multi-agent relèvent des étapes E-24 et E-26/E-31.

**Poids adversaires à zéro.** L'entraînement et toutes les validations se font sans
adversaire : les colonnes 105 à 125 des features restent nulles. Leurs poids d'entrée ont
été mis à zéro pour la course commune des pilotes solo (E-28) : avec des adversaires,
chaque pilote conduit exactement comme seul en piste. Seul, il conduit comme avant
([preuves](../docs/results/opponent-weights/README.md)). Le comportement réel avec
d'autres voitures n'est pas encore mesuré.

```bash
uv run python vendor/simulation/verify_submission.py --submission submission/driver1
uv run python vendor/simulation/verify_submission.py --submission submission/driver2
```
