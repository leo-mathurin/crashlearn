# Provenance du simulateur

| Champ | Valeur |
|---|---|
| Archive | `simulation_20260918.zip` |
| Origine | Ressources du projet Epitech « Crash & Learn » (fournie par l'équipe pédagogique) |
| Taille | 4 503 845 octets |
| sha256 | `42323790f2f45986bcc60d533bea377d0b3c509963aa72ab38c55b58cdf53826` |
| Dates des entrées | 2026-09-18 |
| Date d'import | 2026-09-27 (ticket E-46) |
| Version amont | `f110_gym` 0.2.1 (fork F1TENTH, d'après `setup.py`) ; commit amont inconnu |
| Emplacement | `vendor/simulation/` |

## Méthode d'import

```bash
git rm -rq vendor/simulation
unzip -q simulation_20260918.zip -x '__MACOSX/*' -d vendor/
sha256sum simulation_20260918.zip
```

Seules les métadonnées macOS (`__MACOSX/`) ont été écartées.

**Aucune modification.** Tous les autres fichiers sont identiques à l'archive (`diff -rq` sans écart). C'est la version qui fait foi pour le tournoi : la consigne est d'exploiter ses comportements, pas de les corriger. Ne pas modifier ce dossier ; les réglages d'entraînement (friction, bruit, etc.) passent par notre code, hors `vendor/`.

## Historique

| Archive | sha256 | Import | Remarque |
|---|---|---|---|
| `simulation.zip` (2026-09-08) | `7110e6580d267d6bdfe6d6530b08b93e9c01679100af16bfa65896330ef8ba2e` | E-8, 2026-09-16 | Auditée dans E-10 (`docs/audit_simulateur.md`). Les correctifs locaux d'E-12 n'ont jamais été mergés. |
| `simulation_20260918.zip` | `42323790f2f45986bcc60d533bea377d0b3c509963aa72ab38c55b58cdf53826` | E-46, 2026-09-27 | Version courante. |

## Changements de la version 2026-09-18

Règles de course (`env_simulation.py`) :

- **Stagnation** : DNF si la progression cumulée gagne moins de 1 m sur 10 s (200 décisions). Avant : 4 s sans nouveau maximum de la phase du tour.
- **Sortie de piste** : DNF immédiat quand le centre de la voiture dépasse la demi-largeur locale plus `_OFFTRACK_MARGIN_M`. Le code vaut **1,00 m**, `INSTRUCTIONS.md` annonce 0,50 m ; c'est le code qui s'exécute.
- **Mur** : l'iTTC du moteur annule la vitesse, puis le wrapper rend une vitesse de recul de 0,3 × la vitesse d'impact, plafonnée à 0,45 m/s.
- **Contact entre voitures** : knockback appliqué au sous-pas du contact, une fois par paire et par décision, plafonné à 0,5 m. Chaque voiture prend au moins 20 % du déplacement. Le déplacement ne vérifie pas la grille d'occupation : une voiture peut être poussée dans un mur (voulu d'après le code).
- **Classement** : les DNF sont classés à la distance parcourue (`_max_progress`). `lap_complete` rapporte aussi le tour gagnant.
- **Graine** : `reset(num_cars, seed=None)` réensemence le bruit LiDAR et le moteur.
- **Friction et bruit LiDAR** : toujours figés (friction 1,0, bruit ±0,1 %), avec des `TODO` qui invitent à les faire varier pendant la course.

Contrat et outillage :

- Schéma adversaires unique `x_rel, y_rel, yaw_rel, speed, progress, lap_count, status`, y compris dans `get_space_info` et les observations factices d'`agent_loader` (qui ont maintenant `agent_id`).
- `agent_loader` : un dossier avec des sous-dossiers est lu en mode Grand Prix et doit contenir au moins `num_cars` équipes.
- Nouveau `verify_submission.py` : imports autorisés `onnxruntime`, `numpy`, `os` ; opset ONNX ≤ 17 ; mode `--grandprix` (dossiers `driver1` à `driver4`).
- Cartes : largeurs de piste mesurées dans les CSV de centerline (`maps/measure_widths.py`, 1,44 à 2,60 m de couloir) ; doublon `Mexico City/` supprimé ; `example` listé par `get_available_maps()`.
- `sim_recorder.py` : `--submissions` devient `--submission`, ajout de `--map` et `--seed`, 2000 pas par défaut. `viz_replay.py --record` exporte en MP4.
- `Dockerfile` : versions exactes (numpy 2.4.6, numba 0.65.1, scipy 1.17.1, torch 2.13.0+cpu, onnx 1.22.0, onnxscript 0.7.1, onnxruntime 1.29.0), image CPU, aucun framework d'entraînement fourni. `pyproject.toml` reprend ces versions pour le moteur et la chaîne ONNX.

## Remarques

- Ce dossier est exclu de Ruff et de pytest (voir `pyproject.toml`).
