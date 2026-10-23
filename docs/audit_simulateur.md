# Audit contractuel du simulateur — résultats d'exécution (E-10)

Complète par l'exécution le document Linear [Simulateur fourni, contrats et écarts](https://linear.app/e-hamilton/document/simulateur-fourni-contrats-et-ecarts-de44f3cb7d5d). Toutes les observations ci-dessous ont été reproduites ; les tests qui les figent sont dans `tests/`.

**Règle depuis le 27/09/2026 :** le simulateur est vendorisé **sans modification** (`vendor/PROVENANCE.md`). C'est la version du tournoi, et **ses défauts peuvent être exploités**. Les tests ne décrivent donc plus des défauts « à corriger ». Ils figent le comportement officiel : si une nouvelle archive change une règle, la suite casse, au lieu de changer en silence ce que nos agents apprennent.

## Environnement

| Élément | Valeur |
| -- | -- |
| Archive | `simulation_20260918.zip` (sha256 `42323790…`, fichiers datés du 18/09/2026) |
| Machine | Linux 7.0.0-34, 8 cœurs, CPU uniquement |
| Python | 3.11.16 via uv 0.12.15 |
| Dépendances | versions exactes du `Dockerfile` officiel : numpy 2.4.6, numba 0.65.1, scipy 1.17.1, onnxruntime 1.29.0 (voir `pyproject.toml`) |
| Docker | non utilisé pour cet audit |

## Commandes

```bash
uv run pytest                                               # CI (exclut slow)
uv run pytest -m slow                                       # tours complets, 23 circuits, exploits
cd vendor/simulation && uv run python test_contract.py      # tests fournis : 23 passed, 1 failed (T08)
cd vendor/simulation && uv run python test_integration.py   # tests fournis : 6 passed
cd vendor/simulation/gym && uv run python -m pytest f110_gym/unittest/dynamics_test.py  # 5 passed
```

T08 (« friction starts at 1.0, then drops ») échoue toujours : la friction reste figée, et `_update_friction` est laissé en `TODO` par l'équipe pédagogique.

## Règles de la version 2026-09-18

| Règle | Mesure | Test |
| -- | -- | -- |
| Stagnation | Voiture immobile : DNF exactement au pas **201** (fenêtre de 200 pas = 10 s, gain minimal 1 m). | `test_idle_car_is_dnf_after_window_plus_one_steps` |
| Sortie de piste | DNF immédiat si le centre sort du couloir (demi-largeur locale + `_OFFTRACK_MARGIN_M` = **1,00 m** ; `INSTRUCTIONS.md` annonce 0,50 m). | `test_car_outside_the_track_corridor_is_dnf_immediately`, `test_offtrack_margin_differs_from_the_instructions` |
| Mur | Au contact : vitesse remplacée par un recul de 0,3 × la vitesse d'impact (≤ 0,45 m/s), cap conservé, statut inchangé. | `test_wall_contact_bounces_back_keeps_yaw_and_does_not_terminate` |
| Contact entre voitures | Knockback au sous-pas, 0,5 m au plus, au moins 20 % pour chaque voiture, sans contrôle de la grille d'occupation. | `test_knockback_constants` |
| Tours | `lap_complete` rapporte aussi le tour gagnant ; arrivée au pas 3124 à 3 m/s sur le circuit d'exemple. | `test_pure_pursuit_completes_three_laps_and_finishes` |
| Classement | Arrivées d'abord, puis actives à la progression cumulée, puis DNF à la distance parcourue. | `test_ranks_follow_cumulative_progress_for_active_cars` |
| Friction | Figée à 1,0 (bornes 0,99–1,0, toutes les 20 s, jamais appliquées). | `test_friction_is_frozen_at_one` |
| LiDAR | Bruit multiplicatif ±0,1 %, pas de dropout, voit les murs mais pas les voitures. | `test_lidar_*` |
| Schémas | `get_space_info`, `create_dummy_obs/info` et `get_obs/get_step_info` concordent (anciens D2/D3). | `test_space_info_*`, `test_loader_dummy_*` |
| Cartes | 23 circuits + `example` ; `set_map` atomique (anciens D4/D5). | `test_simulator_lists_the_23_circuits_and_example`, `test_failed_set_map_does_not_poison_singleton` |

## Exploits mesurés

### Traverser un mur fin

Le test iTTC du moteur ne se déclenche plus quand la vitesse est quasi nulle ni une fois la carrosserie dans le mur. Pied au plancher contre un mur, la voiture rebondit, puis s'enfonce d'environ 5 mm par pas ; une fois dedans, son rayon frontal lit 15 m. La progression se mesure par projection sur la centerline. Si le mur sépare deux portions de piste assez proches pour que le centre ne quitte jamais le couloir toléré, la voiture est créditée de tout le tronçon sauté.

- **Mur épais** (circuit d'exemple) : moins de 1,5 m parcourus, puis DNF de stagnation au pas 201.
- **Zandvoort**, du point de départ vers le point 174 de la centerline (mur de 2,66 m, 78 m de piste plus loin), à fond : **traversée en 249 pas, +78 m crédités, voiture toujours active** (`test_a_thin_wall_can_be_cut_through_for_free_progress`, lent).
- **Sochi**, du point 972 vers le point 120 (4,14 m, 126 m plus loin) : +126 m crédités, puis DNF. Le couloir est trop étroit pour repartir. Un DNF est classé à la distance parcourue, qui garde ce gain.
- **Montreal**, mur de 1,95 m : +14 m crédités à 10 m/s, puis DNF de stagnation.
- Sans succès à 2 et 10 m/s : 12 autres candidats (Hockenheim, Nuerburgring, SaoPaulo, Shanghai, YasMarina…), finis en DNF de stagnation.

Circuits dont deux portions sont assez proches pour que le couloir toléré se recoupe (marge de 1,00 m) : Hockenheim, Montreal, Nuerburgring, SaoPaulo, Shanghai, Sochi, YasMarina, Zandvoort. Si la marge passe à 0,50 m comme l'annonce `INSTRUCTIONS.md`, la plupart disparaissent.

### Se faire pousser par l'arrière

Sur Spa, voiture de devant à 6 m/s, voiture de derrière partant 1 m derrière à 10 m/s, pendant 5 s : la voiture de devant gagne **+4,5 m** par rapport à la même course seule (+2,2 m à 3 m/s contre 6 m/s). Aucune des deux n'est DNF, et celle qui pousse dépasse ensuite (`test_being_rear_ended_pushes_the_front_car_forward`, lent). Le knockback déplace les voitures sans vérifier la grille d'occupation. Une voiture poussée latéralement peut donc être envoyée dans un mur, voire à travers un mur fin (non mesuré).

### Pistes non explorées

- Pousser un adversaire dans un mur, ou le faire passer en DNF (stagnation, sortie de piste).
- Se servir du rebond sur un mur pour repartir en marche arrière sans perte de temps.

## Première archive (2026-09-08) : défauts relevés et devenir

Audit exécuté le 16/09/2026 sur `simulation.zip`. Correctifs locaux proposés dans E-12 (PR #7), **fermée sans merge** le 27/09/2026.

| ID | Défaut dans la première archive | Dans la version 2026-09-18 |
| -- | -- | -- |
| D1 | `_update_friction` sans effet, `mu` figé à 1,0 | Inchangé (`TODO` en amont). La friction variable se fera côté entraînement (E-23). |
| D2 | `get_space_info` annonce un autre schéma d'adversaires | Corrigé en amont. |
| D3 | Observations et `info` factices du loader hors schéma | Corrigé en amont (`agent_id`, clés et types réels). |
| D4 | `Mexico City` listé mais inchargeable | Dossier supprimé en amont ; `Mexico City` reste un alias de saisie (`docs/tracks.md`). |
| D5 | `set_map` raté qui empoisonne le singleton | Corrigé en amont (`set_map` atomique). |
| D6 | Murs traversables par fluage ; voiture hors carte jamais DNF | Fluage toujours là, mais DNF de sortie de piste et de stagnation : exploitable seulement sur les murs fins (voir plus haut). |

Les faux DNF autour de la ligne et la divergence en marche arrière braquée (relevés par le pilote de contrôle E-11) sont aussi corrigés en amont. La graine se règle par `reset(num_cars, seed=)`.

Observations de la première archive toujours vraies :

- `RaceCar.scan_simulator` est un singleton de classe. Après de nombreux cycles `reset()`/`close()` dans un même interpréteur, le scan flotte légèrement (< 0,05). Sans impact sous `SubprocVecEnv`.
- Le moteur réassigne `agent.state` à chaque sous-pas (`base_classes.py`) : toute référence gardée entre deux `simulation_step()` est périmée.
- `sim_recorder.py` impose sa propre limite `--laps` (2 par défaut). En self-play, il partage une même instance d'`Agent` entre voitures : notre évaluation garde sa propre boucle, avec une instance par voiture (E-13).
- La ligne droite de départ du circuit d'exemple est courte (mur au pas ~10 à fond) : ne pas s'en servir pour calibrer la vitesse.
- Montreal et Shanghai n'ont pas de `_raceline.csv` ; Monaco, cité en exemple, n'existe pas.
