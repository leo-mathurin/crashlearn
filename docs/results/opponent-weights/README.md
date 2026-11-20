# Pilotes solo : poids adversaires à zéro

`driver1` et `driver2` s'entraînent seuls en piste : les colonnes 105 à 125 de chaque
frame (adversaires) restent nulles, et leurs poids d'entrée gardaient leur
initialisation aléatoire. Avec des voitures adverses, ces poids non appris décalaient
les actions normalisées jusqu'à **0,249** (driver1) et **0,246** (driver2), sur 4096
entrées aléatoires.

`scripts/zero_opponent_weights.py` met ces poids à zéro dans la première couche de
l'acteur et du critique, pour chaque frame empilée : 21 colonnes pour driver1 (h1), 84
pour driver2 (h4). Aucune interaction n'est ajoutée.

| Pilote | Modèle livré | Source |
|---|---|---|
| driver1 | `c343c68b19985d250453642a8c49e4b6609de99cd81767160be0f0cbcd8064e1` | `4956e2a9…`, [livrable driver1](../driver1/README.md) (progression masquée, 4,75 m/s) |
| driver2 | `fcfbfb6e9dac47ca79e24ed5fbd6bbbf426f8960ed7aafd143c3f13668b1d89e` | `3e495a4a…`, [E-18](../e-18/README.md) `ppo-h4` seed 1 |

Les [manifests](manifests/) dérivés gardent la source et ses dérivations.

## Preuves

| Contrôle | driver1 | driver2 |
|---|---|---|
| Nouveau modèle avec adversaires contre ancien modèle seul, 4096 entrées ONNX | écart 0 | écart 0 |
| Courses seul en piste, nominal, image CPU du tournoi | 40/40 identiques à `confirmation-400.json` de driver1, seeds 200 à 209 | 20/20 identiques à l'ancien agent, seeds 200 à 204 |
| Écart ONNX/PyTorch à l'export, 256 entrées | 3,87 × 10⁻⁷ | 6,56 × 10⁻⁷ |

« Identiques » : même statut, même progression, même nombre de décisions, mêmes temps
au tour, mêmes contacts et même vitesse maximale. Seul en piste, chaque pilote conduit
donc comme avant, et les résultats publiés de driver1 (400/400) et de E-18 restent
valables. Avec des adversaires, il conduit comme s'il était seul ; son comportement réel
avec d'autres voitures n'est pas encore mesuré (E-28).

Fichiers bruts (artefact W&B `results-opponent-weights`, voir
[Résultats bruts](../../experiments.md#résultats-bruts)) : `onnx-parity-driver1.json`,
`onnx-parity-driver2.json`, `driver1-new-200-209.json`, `driver2-old-200-204.json` et
`driver2-new-200-204.json`. Archive de la station :
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
L'`agent.py` livré ne change pas : le générateur est le même, et la CI compare son code
à `agent_source`.
