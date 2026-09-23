# Audit contractuel du simulateur — résultats d'exécution (E-10)

Complète par l'exécution le document Linear [Simulateur fourni, contrats et écarts](https://linear.app/e-hamilton/document/simulateur-fourni-contrats-et-ecarts-de44f3cb7d5d), qui reposait sur une lecture du code. Les observations ci-dessous ont toutes été reproduites ; les tests qui les figent sont dans `tests/`.

Le simulateur est vendorisé dans `vendor/simulation/` (voir `vendor/PROVENANCE.md`) ; les noms de circuits viennent de `crashlearn.tracks` (voir `docs/tracks.md`). Les vérifications de contrat les plus basiques (schéma `get_obs`, absence de `done`, clipping des actions) ont été écrites indépendamment dans `tests/test_simulator_contract.py` ; les tests ci-dessous les complètent sans les dupliquer.

## Environnement

| Élément | Valeur |
| -- | -- |
| Archive | `simulation.zip` (4 567 078 octets, fichiers datés du 08/09/2026), vendorisée dans `vendor/simulation/` |
| Machine | Linux 7.0.0-31, 8 cœurs, 15 Go RAM, CPU uniquement |
| Python | 3.11.16 via uv 0.12.15 |
| Dépendances | numpy 2.4.6, numba 0.67.0, scipy 1.17.1, Pillow, pyyaml 6.0.3, pytest 9.1.1, ruff 0.16.7 |
| Docker | non utilisé pour cet audit (image fournie non construite) |

Le Dockerfile fourni ignore volontairement `setup.py` (qui épingle gym 0.19 et numpy ≤ 1.22) : nos dépendances suivent les pins lâches du Dockerfile. Aucune incompatibilité rencontrée avec numpy 2.x / numba 0.67.

## Commandes

```bash
uv sync                                               # environnement + lockfile
uv run pytest                                         # exclut les tests marqués slow (CI)
uv run pytest -m slow                                 # suite lente (parcours des 23 circuits, tours complets)
cd vendor/simulation && uv run python test_contract.py     # tests fournis : 20 passed, 1 failed (T08)
cd vendor/simulation && uv run python test_integration.py  # tests fournis : 5 passed
```

## Tests fournis

| Suite | Résultat | Détail |
| -- | -- | -- |
| `test_contract.py` (17 tests, 21 assertions nommées) | 20 PASS / 1 FAIL | T08 « friction starts at 1.0, then drops » échoue : `friction should have dropped, got 1.0`. Prévu par l'audit statique (D1). |
| `test_integration.py` (5 tests) | 5 PASS | T05 collision véhicule, T10 DNF sans corruption, T-wall, T15 progression, T23 friction initiale. |

Les tests fournis ne couvrent que 1 à 3 voitures ; `tests/test_simulator_reset_and_obs.py` étend à 4.

## Contrat confirmé par exécution

- `reset(n)` pour n ∈ {1,2,3,4} : `info` contient toujours les 4 slots ; les slots absents ont `agent_status=0`, `opponents_mask=False`, collisions/lap_complete/stagnation `False`, `lap_times=[]`, rang `_MAX_SLOTS`.
- `get_obs` : clés `agent_id, lidar (100 float32), velocity, steering, progress, lap_count, rank, opponents{0..3}` ; adversaires avec `x_rel, y_rel, yaw_rel, speed, progress, lap_count, status`. Slot propre et slots absents/inactifs neutralisés à zéro. Repère ego confirmé sur la grille 2×2 (voisin à `|y_rel| = 0.6`, rangée arrière à `x_rel = −0.8`).
- `get_step_info` : 13 clés (`step_count, time_elapsed, ranks, collisions{wall,vehicle}, progress_delta, lap_complete, stagnation, friction_current, opponents_mask, agent_status, lap_times, race_over, max_progress`). Pas de `done`.
- Actions clipées silencieusement à vitesse [−2, 10] et braquage ±0.4189 ; ignorées pour une voiture non active.
- Déterminisme : même séquence d'actions → mêmes scans (moteur seedé 42, rng LiDAR seedé 0), **mais seulement tant qu'on reste dans une même construction de `_Sim`**. `RaceCar.scan_simulator` est un singleton *de classe* (documenté dans la docstring d'`env_simulation` comme « LiDAR class-variable trap ») : après suffisamment de cycles `reset()`/`close()` dans un même interpréteur, un léger flottement (< 0,05 sur l'amplitude du scan) apparaît d'une construction à l'autre. Sans impact sous `SubprocVecEnv` (isolation par process), mais à surveiller pour tout harnais d'évaluation qui enchaîne de nombreux épisodes dans un seul process long. Découvert en portant `test_step_is_deterministic_for_same_seed` sur la suite complète (invisible en l'exécutant seul).
- LiDAR : bruit multiplicatif uniforme **±0,1 %** (max mesuré 0,097 %), tiré une fois par `simulation_step` et réutilisé par `get_obs` ; aucun dropout ; ne voit pas les voitures (scan identique que l'adversaire soit à 0,6 m ou à 50 m).
- Anti-stagnation : voiture immobile → DNF exactement au pas **81** (fenêtre 80 pas = 4 s), téléportée en (10000+5·slot, 10000), `race_over=True` si plus aucune active. Une voiture à 3 m/s n'est jamais DNF sur la fenêtre.
- Tours et arrivée (pilote pure pursuit à 3 m/s, circuit d'exemple, 156 m) : `lap_complete` aux pas 1045 et 2084, arrivée au pas 3124 (`status=2`, `race_over`, 3 `lap_times` ≈ 52 s). Le `lap_complete` du dernier tour est masqué (statut ≠ 1). Voiture arrivée téléportée hors piste, rang 1 devant les actives.
- Classement : arrivées par ordre d'arrivée, actives par `_cum` décroissant, DNF en dernier, slots vides à 4.
- Mur : au premier contact, vitesse annulée, cap conservé (le moteur forké ne clobbère plus le yaw, `base_classes.py:246-252`), statut inchangé. Marche arrière efficace si appliquée dans les ~30 pas suivant le contact.
- Circuits : 23 circuits distincts chargés avec 4 voitures, 20 pas à 2 m/s sans contact mur, centerline de 260 m (Oschersleben) à 554 m (Spa), `set_map` ≈ 1 s (rebuild + JIT). Aucun `_config.yaml` : pose de départ dérivée du premier point de centerline.

## Défauts reproduits

Chaque défaut a été figé par un test `xfail(strict=True)` dans `tests/test_simulator_defects.py` : le jour où il est corrigé, le test passe en XPASS strict et force à retirer le marqueur. **E-12 a corrigé D1, D4, D5 et D6** (détail et commits dans `vendor/PROVENANCE.md`) ; D2 et D3 restent en xfail, volontairement (voir « Corrections attendues »).

| ID | Défaut | Mesure | Test |
| -- | -- | -- | -- |
| D1 | `_update_friction` est un no-op : `friction_current` et le `mu` moteur restent à 1.0 | 600 pas (30 s) → une seule valeur `{1.0}` ; T08 fourni échoue | `test_friction_changes_over_time`, `test_engine_mu_follows_friction_current` |
| D2 | `get_space_info` annonce des adversaires `rel_x/rel_y/rel_dist/rel_yaw/velocity/active`, `get_obs` renvoie `x_rel/y_rel/yaw_rel/speed/progress/lap_count/status` | aucune clé commune | `test_space_info_opponent_keys_match_real_obs` |
| D3 | `agent_loader.create_dummy_obs/info` ne suivent pas le vrai schéma : pas d'`agent_id`, clés adversaires du faux schéma, `info` sans `time_elapsed/ranks/progress_delta/race_over`, listes au lieu de dicts | le dry-run du loader ne détecte pas un agent qui lirait `obs["agent_id"]` ou `info["ranks"]` | `test_loader_dummy_obs_matches_real_obs_schema`, `test_loader_dummy_info_matches_real_info_schema` |
| D4 | `get_available_maps()` liste `"Mexico City"` mais ses fichiers sont préfixés `MexicoCity_` : `set_map("Mexico City")` → `FileNotFoundError` | 24 noms listés, 23 chargeables | `test_every_available_map_can_be_loaded` |
| D5 | Un `set_map` raté laisse `_current_map` sur la carte cassée et `_sim=None` : tout `reset()` suivant replante jusqu'à un `set_map` valide | reproduit avec une carte listée sans fichiers, indépendamment de D4 | `test_failed_set_map_does_not_poison_singleton` |
| D6 | **Mur traversable.** Le test iTTC (`laser_models.py:206-213`) ne fire que si `0 ≤ (scan − côté)/(v·cos) < 0,015 s`. Au redémarrage depuis v=0 la fenêtre fait ~1 mm : la voiture avance de quelques mm par pas (fluage), le flag `collisions.wall` clignote. Une fois le corps dans le mur, `scan − côté < 0` → plus aucune détection : la voiture traverse, ressort et roule hors carte **en restant ACTIVE** (la projection sur la centerline continue de progresser, donc pas de DNF). La marche arrière ne libère plus une voiture qui a pénétré. | poussée à 10 m/s : cellule occupée au pas 50, 0,92 m au pas 120, 133 m hors carte au pas 400, statut 1 ; à 3 m/s : 40 m au pas 400. Marche arrière après 120 pas : v reste 0 sur 60 pas. | `test_wall_is_impassable_when_pushing_for_10_seconds`, `test_car_pinned_against_a_wall_is_dnf_by_stagnation`, `test_reverse_frees_car_after_wall_push` |

D6 est absent de l'audit statique et prioritaire pour E-12 : un agent RL peut apprendre à couper à travers les murs, et une voiture sortie de piste n'est jamais classée DNF. Piste minimale : dans le wrapper, comparer la pose post-sous-pas à l'occupancy grid (ou imposer une distance LiDAR minimale ≥ demi-largeur) et restaurer la pose indépendamment de l'iTTC ; déclarer DNF une voiture hors carte.

## Observations statiques confirmées (sans test dédié ou test « documentaire »)

| ID | Observation |
| -- | -- |
| O1 | Bornes friction `(0.99, 1.0)` et intervalle `(20, 20)` s : même corrigée, la variation annoncée est négligeable (test `test_friction_bounds_are_almost_flat`). |
| O2 | Docstring de `get_obs` annonce ±3 % de bruit LiDAR, constante `_LIDAR_NOISE = 0.001` (test `test_get_obs_docstring_overstates_lidar_noise`). |
| O3 | `info["lap_complete"][i]` est un `bool` Python pour une voiture active mais un `np.bool_` pour une voiture arrivée/DNF (`bool(...) and self._status[i] == 1`). Utiliser `bool()` côté wrapper (pertinent pour E-13/E-14). |
| O4 | Le moteur réassigne `agent.state` à chaque sous-pas (`base_classes.py:378`) : toute référence conservée entre deux `simulation_step()` est périmée. Piège pour les tests et le recorder. |
| O5 | `sim_recorder.py` : `--laps` par défaut 2 alors que l'aide dit 1 ; `--steps` par défaut 2000 alors que l'INSTRUCTIONS.md dit 500 ; il force `_finish` lui-même à `--laps` tours, court-circuitant les 3 tours du moteur. En self-play, une seule instance d'`Agent` est partagée par toutes les voitures (`sim_recorder.py:137-142`). |
| O6 | `agent_loader` ne vérifie que `agent.py` et `model.onnx` ; aucune whitelist d'imports (le contrat de soumission de `tests/test_submission_contract.py` en ajoute une côté équipe) ; `model.onnx.data` non exigé. |
| O7 | Le docstring de `env_simulation` décrit un rollback anti-clobber du yaw devenu inutile : le moteur forké ne remet plus `state[4]` à 0. Le rollback reste inoffensif mais n'empêche pas D6. |
| O8 | La ligne droite de départ du circuit d'exemple est courte : à fond, mur au pas ~10, v max ≈ 4,8 m/s. Ne pas s'en servir pour calibrer la vitesse. |
| O9 | Montreal et Shanghai n'ont pas de `_raceline.csv` ; Monaco cité en exemple n'existe pas. |

## Corrections attendues (vers E-12) et état

1. D6 — rendre les murs infranchissables et DNF hors carte (bloquant pour l'entraînement). **Fait (E-12)** : contact testé sur la grille d'occupation à chaque sous-pas ; une voiture ne peut plus quitter l'espace libre, et une voiture bloquée contre un mur est DNF par la règle de stagnation.
2. **Fait (E-12)** D1 — implémenter `_update_friction` (mettre à jour `mu` via `update_params`) et rendre plages/intervalle configurables pour la randomisation d'entraînement.
3. **Fait (E-12)** D4/D5 — supprimer le dossier `Mexico City` (copie mal nommée) et rendre `set_map` atomique (n'écrire `_current_map` qu'après succès).
4. **Non traité dans E-12, volontairement** D2/D3 — aligner `get_space_info` et les dummies du loader sur le vrai schéma, ou fournir notre propre adaptateur (E-14, `crashlearn.features.adapt_observation`) et ne jamais faire confiance au dry-run du loader.
5. O5 — ne pas dépendre de `sim_recorder.py` pour l'évaluation ; écrire notre propre boucle avec une instance d'`Agent` par voiture (E-13).
