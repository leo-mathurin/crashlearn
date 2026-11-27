# E-23 : driver1 face à une météo qui change en pleine course

Le protocole est figé dans `docs/memlog.md` (2026-11-12) avant le premier résultat. Les
conditions sont décrites dans [docs/weather.md](../../weather.md). Toutes les courses
comptent, DNF et TIMEOUT compris.

## Résultat

Le nouveau `driver1` finit **262/280** courses météo neuves, contre **218/280** pour
l'ancien sur les mêmes courses, et garde **400/400** en nominal dans l'image du tournoi.
Il remplace donc l'ancien, comme le prévoit le protocole.

| Confirmation, 4 circuits de validation | Ancien driver1 | Nouveau driver1 |
|---|---:|---:|
| `changing_weather`, seeds 300 à 349 | 162/200 | **195/200** |
| `low_friction` (0,7), seeds 300 à 309 | 28/40 | **34/40** |
| `heavy_rain` (0,7 et bruit 0,1), seeds 300 à 309 | 28/40 | **33/40** |
| Nominal, seeds 200 à 299, agent ONNX figé | 400/400 | **400/400** |

- **Par circuit** : le gain vient de Montreal, 15/70 courses météo finies avant et 55/70
  après. Sochi et Spielberg finissent tout dans les deux cas ; YasMarina passe de 63/70 à
  67/70.
- **DNF restants** : 15 à Montreal, 3 à YasMarina. À Montreal, le nouveau modèle finit
  4/10 en `low_friction` comme en `heavy_rain`.
- **Chrono** : temps total médian de 230,0 s contre 230,5 s sur les 400 courses
  nominales. Le plafond de vitesse reste 4,75 m/s.

## Campagne de 117 courses

Chaque modèle court la campagne : train nominal (45 courses) et validation en six
conditions (12 courses chacune), seeds 0 à 2. « Météo » additionne `low_friction`,
`heavy_rain` et `changing_weather` (36 courses).

| Modèle | Train /45 | Val. nominal /12 | Météo /36 | Validation /72 |
|---|---:|---:|---:|---:|
| Source (`cap5-random-h1-seed0`) | 45 | 10 | 22 | 51 |
| Ancien driver1 (dérivé de la source) | 45 | 12 | 27 | 63 |
| A seed 0, brut / dérivé | 45 / 45 | 9 / 12 | 22 / 31 | 49 / 66 |
| A seed 1, brut / dérivé | 43 / 45 | 6 / 12 | 18 / 26 | 36 / 62 |
| **A seed 2**, brut / **dérivé** | 43 / 42 | 12 / 12 | 21 / **31** | 57 / **67** |
| B seed 0 / 1 / 2, brut | 31 / 45 / 45 | 6 / 9 / 8 | 17 / 26 / 23 | 34 / 53 / 48 |
| C seed 0 / 1 / 2, brut | 45 / 44 / 45 | 10 / 6 / 12 | 23 / 18 / 33 | 52 / 36 / 69 |

- **Variantes** : A, météo changeante avec 25 % d'épisodes nominaux ; B, A avec
  `friction_current` en entrée ; C, même adaptation sans météo. 2 027 520 interactions
  chacune, depuis la source de l'ancien driver1, plafond 5 m/s et départs diversifiés.
- **Dérivé** : chaîne de l'ancien driver1, sans interaction ajoutée : progression masquée,
  plafond 4,75 m/s, poids adversaires à zéro.
- **Sélection** : A seed 2 dérivé finit le plus de courses de validation (67/72). Il perd
  3 courses train nominales (42/45) ; la confirmation de 400 courses nominales neuves est
  la garde prévue pour ce risque, et elle passe.

## Limites

- **La part de la météo dans le gain n'est pas isolée.** Brut, C (sans météo) finit 74
  courses météo sur 108, contre 61 pour A. Les seeds varient beaucoup (C : 18 à 33), et C
  seed 2 brut finit même 69/72 en validation. Le protocole ne retenait que les A dérivés
  comme candidats ; C dérivé n'a pas été évalué. Le nouveau driver1 est meilleur que
  l'ancien sur des courses neuves, mais ce gain peut venir des interactions ajoutées
  plutôt que de la météo.
- **Friction en entrée non retenue.** B finit 66 courses météo sur 108 contre 61 pour A,
  sous le seuil de +10 fixé avant les résultats, et perd en nominal de validation (23/36
  contre 27/36). Les features restent à 126 colonnes.
- Les circuits test restent scellés. Les seeds de course ne sont pas des entraînements
  indépendants.

## Provenance

Le livrable est le modèle `d473a5ba64d4…` ([A-seed2-derived](manifests/A-seed2-derived.json)),
dérivé de [A-seed2-derived-blind](manifests/A-seed2-derived-blind.json), lui-même dérivé de
[A-seed2](manifests/A-seed2.json). Les [manifests](manifests/) couvrent tous les runs.

- **Commits** : les runs de la campagne enregistrent `a28cf7c`, fixé par `GIT_COMMIT` au
  lancement. Le premier B seed 0 a échoué au démarrage (colonne de friction créée sur le
  CPU, modèle sur CUDA) ; il reste dans
  [B-seed0-failed-device](manifests/B-seed0-failed-device.json). La correction `f55c68b`
  ne touche que cette colonne. B seeds 1 et 2 ont tourné avec elle sous le même
  `GIT_COMMIT` ; B seed 0, relancé ensuite, enregistre `f55c68b`.
- **Résultats bruts** : un fichier par modèle, dans `campaign/` (par exemple
  `A-seed2-derived.json`) et `confirmation/` (`A-seed2-derived-weather-280.json`,
  `A-seed2-nominal-400.json`, `driver1-weather-280.json`). Ils sont dans l'artefact W&B
  `results-e-23` : `SHA256SUMS` donne sa version et l'empreinte de chaque fichier
  ([récupération](../../experiments.md#résultats-bruts)).
- **Archive de la station** : `/home/leo/runs/crashlearn/e23-20261112` (runs, checkpoints,
  logs, résultats bruts, ancien `driver1`).

```bash
uv run python scripts/train_ppo.py --history-depth 1 --timesteps 2000000 --device cuda \
  --n-envs 15 --fixed-map-workers --max-speed 5 --random-start --weather --nominal-share 0.25 \
  --initialize-from /home/leo/runs/crashlearn/e19-20261110/cap5-random-h1-seed0/model.zip \
  --seed 2 --output-dir runs/A-seed2
uv run python scripts/ablate_progress.py --max-speed 4.75 --run-dir runs/A-seed2 \
  --output-dir runs/A-seed2-blind
uv run python scripts/zero_opponent_weights.py --run-dir runs/A-seed2-blind \
  --output-dir runs/A-seed2-derived
uv run python scripts/export_agent.py --run-dir runs/A-seed2-derived --output-dir runs/agent
```
