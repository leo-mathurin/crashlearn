# E-49 : driver1 finit toutes les courses météo de la confirmation

`driver1` borne maintenant sa vitesse à **4,75 × √μ m/s**, μ étant `friction_current` lu
dans `info` (le tournoi le donne). À μ = 1 la borne vaut l'ancien plafond ; à μ = 0,7 elle
vaut 3,97 m/s. Le modèle ne change pas (`d473a5ba64d4…`, celui d'E-23) : seul `agent.py`
change. Le protocole a été figé dans `docs/memlog.md` avant le premier essai.

## Confirmation sur courses neuves

| 4 circuits de validation, agent ONNX | Avant (E-23) | Après |
|---|---:|---:|
| `changing_weather`, seeds 350 à 449 | 389/400 | **400/400** |
| `low_friction` (0,7), seeds 350 à 399 | 162/200 | **200/200** |
| `heavy_rain` (0,7 et bruit 0,1), seeds 350 à 399 | 165/200 | **200/200** |
| Nominal, seeds 200 à 299, image du tournoi | 400/400 | **400/400** |

- Les 84 échecs d'avant étaient à Montreal (57) et à YasMarina (27).
- **Nominal inchangé** : les 400 courses sont identiques à celles d'E-23 (statut,
  progression, décisions, temps au tour, contacts), puisque la borne ne joue pas à μ = 1.
- **Coût** : sur les 716 courses météo finies par les deux versions, le temps médian passe
  de 247,6 à 261,5 s (+5,6 %).
- Les fichiers livrés sont identiques avant et après les courses (`candidate-freeze.json`).

## Sélection

Les 280 courses de confirmation d'E-23 servent de cas de sélection. Le protocole retient le
plus petit exposant k qui les finit toutes.

| k | Finies | Temps médian des finisseurs |
|---|---:|---:|
| 0 (sans borne) | 260/280 | 247,4 s |
| **0,5** | **280/280** | 256,0 s |
| 0,75 | 280/280 | 276,8 s |
| 1 | 260/280 | 280,5 s |

À k = 1 la voiture devient trop lente à Sochi sous 0,7 (20 DNF de stagnation). Sans borne,
l'agent ONNX finit 260 cas, contre 262 pour le modèle PyTorch dans E-23 : sous météo, de
petits écarts numériques font diverger les trajectoires, et Montreal à 0,7 est à la limite.
La sélection et la confirmation utilisent donc l'agent ONNX livré
(`evaluate_drivers.py --submission`).

## Limites

- Résultat sur les 4 circuits de validation, seul en piste. Les circuits test restent
  scellés ; le comportement avec des adversaires n'est pas mesuré (E-28).
- La borne suppose que le tournoi donne `friction_current`, comme le sujet l'annonce.
  Absente ou invalide, elle vaut 1 : l'agent roule alors comme le driver1 d'E-23.

## Provenance

Les `.json` cités (`selection/`, `confirmation/`, `candidate-freeze.json`) sont dans
l'artefact W&B `results-e-49` ; `SHA256SUMS` donne sa version et l'empreinte de chaque
fichier ([récupération](../../experiments.md#résultats-bruts)). Archive de la station :
`/home/leo/runs/crashlearn/e49-20261112` (agents exportés, logs, scripts `selection.sh` et
`confirm.sh`).

```bash
uv run python scripts/export_agent.py --friction-exponent 0.5 \
  --run-dir /home/leo/runs/crashlearn/e23-20261112/A-seed2-derived --output-dir runs/driver1
uv run python scripts/evaluate_drivers.py --submission runs/driver1 \
  --condition changing_weather --seed 350 --output runs/check.json
```
