# E-55 — Test d'installation sur machine vierge

## 1. Informations du test

| Information | Valeur |
|---|---|
| Testeur | Tiziano Denaro |
| Date | 27-11-2026|
| Système d'exploitation | Windows 11|
| Python | 3.11 |
| Version uv |uv 0.13.8|
| Branche testée | main |
| Commit testé | 7e5deac8|

## 2. Tableau de validation

Statuts :
- ⬜ Non testé
- 🟢 Réussi
- 🟠 Réussi avec difficulté
- 🔴 Échoué

| ID  | Tâche                                                     | Commande / Action                                                                                    | Statut | Blocage |
|-----|-----------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------|---------|
| T01 | Clonage du dépôt                                          | `git clone`                                                                                          | 🟢     |         |
| T02 | Installation des dépendances                              | `uv sync`                                                                                            | 🟢     |         |
| T03 | Smoke test du simulateur                                  | `uv run python scripts/smoke_sim.py --map Austin --steps 200`                                        | 🟢     |         |
| T04 | Vérification Ruff                                         | `uv run ruff check .`                                                                                | 🟢     |         |
| T05 | Vérification du formatage                                 | `uv run ruff format --check .`                                                                       | 🟢     |         |
| T06 | Exécution des tests                                       | `uv run pytest`                                                                                      | 🟢     |         |
| T07 | Tests longs                                               | `uv run pytest -m slow`                                                                              | 🟢     |         |
| T08 | Installation des hooks Git                                | `uv run pre-commit install`                                                                          | 🟢     |         |
| T09 | Vérification des hooks                                    | `uv run pre-commit run --all-files`                                                                  | 🟢     |         |
| T10 | Vérification commitlint                                   | `uv run pre-commit run commitlint --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG` | 🟢     |         |
| T11 | Installation des dépendances d'entraînement               | `uv sync --group train`                                                                              | 🟢     |         |
| T12 | Installation du groupe inference                          | `uv sync --only-group inference`                                                                     | 🟢     |         |
| T13 | Vérification des circuits                                 | `uv run python scripts/track_inventory.py`                                                           | 🟠     |         |
| T14 | Régénération des données des circuits                     | `uv run python scripts/track_inventory.py --write`                                                   | 🟢     |         |
| T15 | Vérification du chargement des circuits                   | `uv run python scripts/check_track_loading.py`                                                       | 🟢    |         |
| T16 | Contrôler le fonctionnement de la simulation de référence | `uv run python scripts/control_race.py`                                                              | 🟢      |         |
| T17 | Contrôler la simulation avec des paramètres personnalisés | `uv run python scripts/control_race.py --map Austin --cars 4 --seed 1 --output runs/austin.json`     | 🟢      |         | |                                                           |                                                                                                    |        |         |

## 4. Bilan

| Indicateur | Résultat |
|---|---|
| Tâches réussies | 16|
| Tâches réussies avec difficulté | 1|
| Tâches échouées | 0|
| Tâches non testées | 0|
| Blocages rencontrés | 1|
| Blocages corrigés | 1|

### Conclusion

- [x] Le README permet de cloner le projet.
- [x] L'installation fonctionne sur une machine vierge.
- [x] Les commandes de vérification fonctionnent.
- [x] Les hooks Git fonctionnent.
- [x] Les commandes de gestion des circuits fonctionnent.

**Résultat du test :** Validé 

**Observations finales :** Le readme est clair et complet, l'installation est simple et rapide. Le test a été réalisé sur une machine vierge et toutes les étapes ont été validées avec succès.

