# driver1 : 400/400 courses finies en validation nominale

Le pilote réactif livré dans `submission/driver1` termine **400/400 courses
nouvelles** sur Montreal, Sochi, Spielberg et YasMarina, après 132/132 cas de
sélection. Trois tours, départ officiel, conditions nominales, 8000 décisions au
plus. Les DNF et TIMEOUT comptent tous.

![Finitions nominales par circuit](comparaison.png)

## Du réactif E-18 au livrable

Chaque étape a été décrite avant d'être mesurée. Les seeds de course ne sont pas
des entraînements indépendants : le taux décrit ces quatre cartes en nominal.

| Étape | Validation nominale | Ce qui change |
|---|---:|---|
| Réactif E-18, seed 0 | 7/12 | Point de départ, 5 007 360 interactions |
| Adaptation, plafond 5 m/s | 10/12 | +2 027 520 interactions sur les 15 cartes train, optimiseur neuf |
| Départs train diversifiés | 10/12 | Waypoint aléatoire, ±0,2 m latéral, ±0,1 rad de cap |
| Connexions de progression retirées | 12/12 | Colonnes 102 à 104 à zéro dans la première couche ; aucune interaction |
| Même modèle, seeds 100 à 129 | 118/120 | Confirmation ; 2 DNF à Montreal (seeds 104, 118) |
| Plafond d'inférence 4,75 m/s | 132/132 | Mêmes 13 tenseurs appris ; borne de vitesse 0,1667 → 0,125 |
| Agent ONNX figé, seeds 200 à 299 | **400/400** | Confirmation, 100/100 par carte |

- **Plafond seul** : sans réentraînement, aucun plafond de 10 à 4 m/s ne finit
  les quatre circuits. Ralentir seul ne suffit pas.
- **Ablation** : `scripts/ablate_progress.py` met à zéro les connexions depuis le
  sinus et le cosinus de progression et les tours accomplis, dans l'acteur et le
  critique. Un contrôle sur 256 entrées compare au modèle original avec ces
  entrées masquées. Le modèle source reste intact.
- **Calibration** : les deux DNF de Montreal sont reproduits avec l'agent ONNX
  réel. Le plus haut plafond global qui les corrige, 4,75 m/s, est retenu sur les
  132 anciens cas, sans réglage par circuit ni par seed
  ([diagnostic](speed-diagnostic.json), [preuve](calibration-proof.json)).
- **Confirmation** : l'agent est [figé par ses hashes](candidate-freeze.json)
  avant la première course nouvelle, puis lancé par `scripts/confirm_submission.py`
  dans l'image CPU du tournoi, sans PyTorch ni Stable-Baselines3. Les hashes sont
  identiques après les 400 courses ([résultats bruts](confirmation-400.json)).

## Coût et limites

- **Chrono** : sur les 118 courses finies par les modèles à 5 et 4,75 m/s, le
  temps médian passe de 236,17 s à 247,57 s ([calcul](time-comparison.json)).
  Sur les 7 courses finies par le réactif E-18, le modèle à 5 m/s passait déjà
  de 120,65 s à 203,00 s.
- **Robustesse** : la [campagne de 105 cas](full-campaign-105.json) garde 45/45
  sur train et 12/12 en nominal, bruit LiDAR et marge stricte, mais **6/12** sous
  faible adhérence et forte pluie. Suite dans E-23.
- Les circuits test restent scellés.

## Provenance et reproduction

| Modèle | Manifest |
|---|---|
| Entraînement source, plafond 5 m/s, départs diversifiés | [cap5-random-h1-seed0-2m](manifests/cap5-random-h1-seed0-2m.json) |
| Ablation, 5 m/s, 118/120 ([courses](confirmation-5mps-120.json)) | [cap5-progress-blind](manifests/cap5-progress-blind.json) |
| Livrable, 4,75 m/s, `4956e2a9…` | [cap4.75-progress-blind](manifests/cap4.75-progress-blind.json) |

La [preuve de livraison](submission-proof.json) relie le modèle aux courses :
vérificateur officiel valide, écart ONNX/PyTorch de 4,77 × 10⁻⁷ sur 256 entrées.
Archives de la station : `/home/leo/runs/crashlearn/e19-20261110` (entraînements,
checkpoints, logs) et `/home/leo/runs/crashlearn/e19-100-20261110` (calibration).

```bash
uv run python scripts/ablate_progress.py --max-speed 4.75 \
  --run-dir /home/leo/runs/crashlearn/e19-20261110/cap5-random-h1-seed0 \
  --output-dir runs/recheck-driver1
uv run python scripts/export_agent.py --run-dir runs/recheck-driver1 \
  --output-dir runs/recheck-driver1-agent
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v "$PWD/runs:/runs" -w /tmp \
  -e PYTHONPATH=/workspace/src -e NUMBA_CACHE_DIR=/runs/numba-cache-cpu \
  crashlearn-sim:20260918 python /workspace/scripts/confirm_submission.py \
  --submission /workspace/submission/driver1 \
  --output /runs/recheck-confirmation.json --seed-start 200 --seed-count 100
```
