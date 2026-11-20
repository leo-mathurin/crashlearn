# E-55 — Test d'installation sur machine vierge

## 1. Informations du test

| Information | Valeur |
|---|---|
| Testeur | Tiziano Denaro |
| Date | |
| Système d'exploitation | |
| Python | 3.11 |
| Version uv | |
| Branche testée | |
| Commit testé | |

## 2. Tableau de validation

Statuts :
- ⬜ Non testé
- 🟢 Réussi
- 🟠 Réussi avec difficulté
- 🔴 Échoué

| ID | Tâche | Commande / Action | Statut | Blocage |
|---|---|---|---|---|
| T01 | Clonage du dépôt | `git clone` | 🟢 | |
| T02 | Installation des dépendances | `uv sync` | 🟢 | |
| T03 | Smoke test du simulateur | `uv run python scripts/smoke_sim.py --map Austin --steps 200` | 🟢 | |
| T04 | Vérification Ruff | `uv run ruff check .` | 🟢 | |
| T05 | Vérification du formatage | `uv run ruff format --check .` | 🟢 | |
| T06 | Exécution des tests | `uv run pytest` | 🟢 | |
| T07 | Tests longs | `uv run pytest -m slow` | 🟢 | |
| T08 | Installation des hooks Git | `uv run pre-commit install` | 🟢 | |
| T09 | Vérification des hooks | `uv run pre-commit run --all-files` | 🟢 | |
| T10 | Vérification commitlint | `uv run pre-commit run commitlint --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG` | 🔴 | |
| T11 | Installation des dépendances d'entraînement | `uv sync --group train` | 🟢 | |
| T12 | Installation du groupe inference | `uv sync --only-group inference` | 🟢 | |
| T13 | Vérification des circuits | `uv run python scripts/track_inventory.py` | 🟠 | |
| T14 | Régénération des données des circuits | `uv run python scripts/track_inventory.py --write` | 🟢 | |
| T15 | Vérification du chargement des circuits | `uv run python scripts/check_track_loading.py` | 🟢 | |

## 3. Journal des blocages

| ID | Tâche concernée | Problème rencontré | Correction apportée |
|---|---|---|---|
| B01 | | | |
| B02 | | | |
| B03 | | | |

### Détail d'un blocage

#### B01 — [Titre]

**Commande exécutée :**
```bash
```

**Erreur obtenue :**
```text
```

**Cause :**

**Solution :**

**Modification nécessaire dans le README :**

---

## 4. Bilan

| Indicateur | Résultat |
|---|---|
| Tâches réussies | |
| Tâches réussies avec difficulté | |
| Tâches échouées | |
| Tâches non testées | |
| Blocages rencontrés | |
| Blocages corrigés | |

### Conclusion

- [ ] Le README permet de cloner le projet.
- [ ] L'installation fonctionne sur une machine vierge.
- [ ] Les commandes de vérification fonctionnent.
- [ ] Les hooks Git fonctionnent.
- [ ] Les commandes de gestion des circuits fonctionnent.

**Résultat du test :** Validé / Partiellement validé / Non validé

**Observations finales :**

