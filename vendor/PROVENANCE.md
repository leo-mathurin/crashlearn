# Provenance du simulateur

| Champ | Valeur |
|---|---|
| Archive | `simulation.zip` |
| Origine | Ressources du projet Epitech « Crash & Learn » (fournie par l'équipe pédagogique) |
| Taille | 4 567 078 octets |
| sha256 | `7110e6580d267d6bdfe6d6530b08b93e9c01679100af16bfa65896330ef8ba2e` |
| Dates des entrées | 2026-09-08 |
| Date d'import | 2026-09-16 (ticket E-8) |
| Version amont | `f110_gym` 0.2.1 (fork F1TENTH, d'après `setup.py`) ; commit amont inconnu |
| Emplacement | `vendor/simulation/` |

## Méthode d'import

```bash
unzip -q simulation.zip -x '__MACOSX/*' -d vendor/
sha256sum simulation.zip
```

Seules les métadonnées macOS (`__MACOSX/`) ont été écartées.

Tous les autres fichiers étaient identiques à l'archive jusqu'au ticket E-12.

## Modifications (E-12)

Chaque correctif est un commit séparé, lié au défaut reproduit par l'audit E-10 (`docs/audit_simulateur.md`) et au test qui le vérifie dans `tests/test_simulator_defects.py`. Le contrat d'inférence (`get_obs`, `apply_action`, `get_step_info` : clés, types, bornes) n'est pas modifié.

| Défaut | Fichier | Correctif | Tests |
|---|---|---|---|
| D6 murs traversables | `env_simulation.py` | Après chaque sous-pas, l'empreinte de la voiture (trois disques de rayon largeur/2 sur l'axe long) est testée sur la distance transform de la grille d'occupation ; un contact déclenche le même retour à la pose légale que l'iTTC. Un point hors de l'image compte comme occupé. Un knockback qui pousserait une voiture dans un mur est annulé pour cette voiture. | `test_wall_is_impassable_when_pushing_for_10_seconds`, `test_car_pinned_against_a_wall_is_dnf_by_stagnation`, `test_reverse_frees_car_after_wall_push` |
| D1 friction non appliquée | `env_simulation.py` | `_update_friction` fait tourner le compte à rebours, tire la friction uniformément dans `[_FRICTION_LO, _FRICTION_HI]` avec le RNG du simulateur et l'applique au moteur par `update_params`. Les valeurs livrées (0,99–1,0, toutes les 20 s) restent les défauts ; `set_friction_profile(lo, hi, interval_sec)` règle la randomisation **d'entraînement**, hors contrat d'inférence. | `test_friction_changes_over_time`, `test_engine_mu_follows_friction_current`, `test_training_friction_profile_*` ; T08 fourni passe |
| D5 `set_map` non atomique | `env_simulation.py` | Les fichiers de carte (`_map.yaml` et image) sont vérifiés avant de basculer ; en cas d'échec, les globales de carte sont restaurées et le simulateur est reconstruit sur la carte précédente, puis l'exception est relevée (la course en cours redémarre). | `test_failed_set_map_does_not_poison_singleton`, `test_failed_set_map_before_any_reset_keeps_the_previous_map` |
| D4 doublon `Mexico City` | `maps/Mexico City/` | Dossier supprimé : copie exacte de `MexicoCity/` (sha256 identiques) dont les fichiers restaient préfixés `MexicoCity_`, donc inchargeable. `get_available_maps()` liste désormais 23 circuits ; `Mexico City` reste un alias de saisie dans `src/crashlearn/tracks.yaml`. | `test_every_available_map_can_be_loaded`, `test_simulator_lists_the_23_circuits`, `test_mexico_city_is_an_alias_without_a_directory` |
| Bugs 1 et 2 de `docs/scripted_driver.md` (E-11) : DNF après un passage de ligne ou un recul sur la ligne | `env_simulation.py` | La fenêtre anti-stagnation suit le maximum de la progression **cumulée** (`_cum`), continue d'un tour à l'autre, au lieu du maximum de la phase du tour, qui repasse près de 1,0 juste après la ligne. La phase est aussi ramenée dans [0, 1) : `(0.0 - 1e-19) % 1.0` vaut 1,0 en flottant. `info["max_progress"]` garde sa définition (maximum de la phase dans le tour). | `test_reversing_over_the_start_line_is_not_dnf`, `test_crossing_the_line_does_not_dnf` |

## Remarques

- E-12 a corrigé D1, D4, D5 et D6 (tableau ci-dessus). D2 et D3 (schémas annoncés par `get_space_info` et par les observations factices d'`agent_loader`) ne sont **pas** corrigés ici : ils relèvent de notre adaptateur d'observations (E-14/E-16), et les modifier changerait ce que voit un agent au tournoi. La détection de carte du visualiseur et les métadonnées de replay restent à traiter avec les replays.
- Cohérence des paramètres vérifiée (tests `test_physics_and_decision_rates_are_consistent`, `test_engine_mu_is_the_friction_not_the_nominal_parameter`) : pas de 0,01 s × 5 sous-pas = 50 ms (20 Hz), `ttc_thresh` appliqué à chaque construction. **Divergence conservée** : le `mu` nominal de `_PARAMS` (1,0489) n'atteint jamais la physique, car `reset()` impose `mu = friction_current` (1,0). On garde ce comportement livré ; `friction_current` est donc un `mu` absolu.
- Réglages d'entraînement et règles locales : les constantes livrées (friction 0,99–1,0 toutes les 20 s, bruit LiDAR ±0,1 %) restent les défauts et décrivent les règles locales. La randomisation d'entraînement passe par `set_friction_profile` ; elle ne change ni les clés ni les bornes de `get_obs`/`get_step_info`.
- `Dockerfile` et `docker-compose.yml` sont conservés tels quels, mais ne servent **pas** de référence : ils n'ont pas de configuration GPU, et SB3 y est annoncé sans être installé. L'image reproductible fait l'objet d'un ticket dédié.
- Ce dossier est exclu de Ruff et de pytest (voir `pyproject.toml`).
