# Pilotes solo : poids adversaires à zéro

`driver1` et `driver2` s'entraînent seuls en piste. Les colonnes 105 à 125 de chaque
frame (adversaires) restent donc nulles pendant l'apprentissage, et leurs poids d'entrée
gardaient leur initialisation aléatoire. Avec des voitures adverses, ces poids décalaient
les actions sans avoir rien appris : jusqu'à **0,249** (driver1) et **0,246** (driver2) en
action normalisée, sur 4096 entrées aléatoires.

`scripts/zero_opponent_weights.py` met ces poids à zéro dans la première couche de
l'acteur et du critique, pour chaque frame empilée (21 colonnes pour driver1, 84 pour
driver2). Aucune interaction n'est ajoutée. Le manifest dérivé garde la source et, pour
driver1, la dérivation précédente (progression masquée, plafond 4,75 m/s).

## Preuves

| Contrôle | driver1 | driver2 |
|---|---|---|
| Nouveau modèle avec adversaires = ancien modèle seul, 4096 entrées ONNX | écart 0 | écart 0 |
| Courses seul en piste, image CPU du tournoi | 40/40 identiques à la [confirmation 400](../driver1/confirmation-400.json), seeds 200 à 209 | 20/20 identiques à l'ancien agent, seeds 200 à 204 |
| Accord ONNX/PyTorch à l'export | 3,87 × 10⁻⁷ | 6,56 × 10⁻⁷ |

« Identiques » veut dire même statut, même progression, même nombre de décisions, mêmes
temps au tour, mêmes contacts et même vitesse maximale. Seul en piste, le pilote conduit
donc exactement comme avant : les résultats publiés de driver1 (400/400) et de la
campagne E-18 restent valables. Avec des adversaires, il conduit comme s'il était seul.
Son comportement réel avec d'autres voitures n'est pas encore mesuré (E-28).

Fichiers : [parité driver1](onnx-parity-driver1.json),
[parité driver2](onnx-parity-driver2.json), [courses driver1](driver1-new-200-209.json),
[courses driver2 avant](driver2-old-200-204.json) et [après](driver2-new-200-204.json),
[manifests](manifests/). Archive de la station :
`/home/leo/runs/crashlearn/opponent-weights-20261112` (modèles dérivés, exports, anciens
dossiers de soumission).

## Reproduction

```bash
uv run python scripts/zero_opponent_weights.py \
  --run-dir /home/leo/runs/crashlearn/e19-100-20261110/cap4.75-progress-blind \
  --output-dir runs/driver1-solo-only
uv run python scripts/export_agent.py --run-dir runs/driver1-solo-only \
  --output-dir runs/driver1-solo-only-agent
uv run python scripts/compare_opponent_inputs.py \
  --new runs/driver1-solo-only-agent --old <ancien dossier submission/driver1>
```

Pour driver2, la source est `/home/leo/runs/crashlearn/e18-20261109/ppo-h4-seed1`.
L'`agent.py` livré reste celui de la version précédente : le générateur est inchangé et
le contrôle de la CI compare son code à `agent_source`.
