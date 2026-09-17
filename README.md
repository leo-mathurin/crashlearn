# Crash & Learn

Projet de conduite autonome par reinforcement learning : Léo Mathurin, Lucas Noirie et Thibaud Combaz Deville.

Début : **14 septembre 2026 à 09:00 (Europe/Paris)**. Rendu : **7 février 2027 à 23:42 (Europe/Paris)**.

Objectif : viser le podium national avec trois pilotes solo et un champion multi-agent. Dashboard web local de replays en bonus prioritaire.

## Documentation et suivi

Le [projet Linear Crash & Learn](https://linear.app/e-hamilton/project/crash-and-learn-e85ceaa5b493) centralise tickets, décisions et documents. Le PDF original du sujet est dans ses ressources.

- [Cahier des charges et organisation](https://linear.app/e-hamilton/document/cahier-des-charges-et-organisation-099e0ee24aa9)
- [Architecture RL, MDP et décisions](https://linear.app/e-hamilton/document/architecture-rl-mdp-et-decisions-38244e935fee)
- [Simulateur fourni, contrats et écarts](https://linear.app/e-hamilton/document/simulateur-fourni-contrats-et-ecarts-de44f3cb7d5d)
- [Expériences, robustesse et évaluation](https://linear.app/e-hamilton/document/experiences-robustesse-et-evaluation-6cd41fcc563f)
- [Qualité du code et livraison](https://linear.app/e-hamilton/document/qualite-du-code-et-livraison-1da5f06a62ac)
- [Infrastructure, stockage et reproductibilité](https://linear.app/e-hamilton/document/infrastructure-stockage-et-reproductibilite-9553c0c81bbc)
- [Dashboard de course et vocabulaire](https://linear.app/e-hamilton/document/dashboard-de-course-et-vocabulaire-0987cb2ff84a)

## Développement

Dépôt principal privé : [leo-mathurin/crashlearn](https://github.com/leo-mathurin/crashlearn) (`origin`). GitHub Actions est activé sur ce dépôt personnel.

Le remote `epitech` pointe vers [le dépôt de remise Epitech](https://github.com/EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8). Les pushes courants vont vers `origin` ; le dépôt Epitech est réservé à la remise finale.

Branches persistantes prévues : `develop` pour l'intégration, `main` pour les jalons validés. Branches courtes, PR vers `develop`, revue croisée puis promotion vers `main` avec tag et métriques. Backlog partagé, tâches prises librement et aucun domaine réservé. Conventional Commits.

Socle retenu : Python 3.11, uv, Gymnasium, NumPy, Stable-Baselines3/PyTorch, PPO et SAC, ONNX/ONNX Runtime. Entraînements sur la station Ubuntu de Léo ; W&B avec suivi local de secours. Les versions exactes sont épinglées dans `uv.lock`.

Le cadrage est maintenu dans Linear, y compris les ADR et le glossaire. Ce dépôt contient le code, les tests et la documentation technique.

Exemple de nom de branche : `feat/e-8-bootstrap`. Commits en anglais au format Conventional Commits (`feat:`, `fix:`, `build:`, `docs:`, `test:`, `chore:`).

## Démarrage local

Prérequis : Python 3.11 et [uv](https://docs.astral.sh/uv/). Pas besoin de GPU ni de Docker pour démarrer.

```bash
git clone git@github.com:leo-mathurin/crashlearn.git && cd crashlearn
uv sync                                                    # runtime simulateur + outils dev
uv run python scripts/smoke_sim.py --map Austin --steps 200  # run de contrôle headless
```

La commande `uv sync` n'installe pas les groupes lourds. Pour les ajouter :

```bash
uv sync --group train             # torch, stable-baselines3, wandb
uv sync --only-group inference    # environnement minimal numpy + onnxruntime, sans le projet
```

Qualité : ce sont exactement les commandes exécutées par la CI (`.github/workflows/ci.yml`).

```bash
uv run ruff check .
uv run ruff format --check .      # `uv run ruff format .` pour corriger
uv run pytest                     # exclut les tests marqués slow et gpu
uv run pytest -m slow             # tests longs, hors CI
```

Hooks Git, à installer une fois par clone (`uv` doit être dans le `PATH`) :

```bash
uv run pre-commit install         # hooks pre-commit (ruff check --fix, ruff format) et commit-msg
uv run pre-commit run --all-files
uv run python scripts/lint_commit_msg.py --message "feat: add x"   # tester un message à la main
```

Les hooks sont déclarés en `repo: local` et appellent `uv run --locked ruff …` : ils utilisent les versions de `uv.lock`, les mêmes qu'en CI. Il n'existe pas de second environnement géré par pre-commit, donc aucune dérive de version n'est possible. En contrepartie, `uv` est nécessaire pour committer, et un lock périmé bloque le commit comme il bloque la CI. `tests/test_hook_versions.py` échoue si un hook distant réintroduit un outil déjà présent dans `uv.lock` avec une autre version.

Périmètre des fichiers : il est défini **uniquement** dans `[tool.ruff]` de `pyproject.toml`, `vendor/` étant exclu. Les hooks lancent exactement les commandes de la CI sur `.`, sans filtre propre ; un fichier Python non suivi et non ignoré est donc lui aussi vérifié.

Markdown : Ruff 0.16 inclut les `*.md` et ne reformate **que** les blocs de code `python`, `py` et `pycon`. La prose et les blocs `bash`, `toml` ou sans langage ne sont pas touchés, et un bloc Python invalide est laissé tel quel. Ce comportement est volontaire : il garde les exemples de la doc cohérents. Pour préserver la mise en forme d'un bloc, utiliser `# fmt: off` dans le bloc, ou un autre langage.

### Messages de commit

Format [Conventional Commits](https://www.conventionalcommits.org/), vérifié par `scripts/lint_commit_msg.py`. Ce script utilise commitlint et ajoute les règles que commitlint (PyPI) ne propose pas.
- Types en minuscules : `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`, `chore`, `revert`, `bump`.
- Titre de **72 caractères au maximum**, pour rester lisible dans `git log --oneline` et sur GitHub.
- Lignes du corps de **100 caractères au maximum**. Les trailers (`Refs: E-9`, `Co-Authored-By: …`) et les lignes contenant une URL en sont exemptés.
- Les messages `Merge …` et `Revert "…"` générés par Git sont ignorés.
- Les commits `fixup!`, `squash!` et `amend!` sont **acceptés en local**, pour `git commit --fixup <sha>` puis `git rebase -i --autosquash`, mais **refusés en CI** : un fixup non résorbé fait échouer la PR.

| Message | Résultat |
|---|---|
| `feat: add track inventory` | valide |
| `fix(tracks)!: seal the test split` | valide |
| corps + `Refs: E-9` + `Co-Authored-By: <nom long> <adresse>` | valide |
| `Update readme` | refusé : format |
| `Feat: add x` | refusé : type en majuscule |
| `feat:add x` | refusé : espace manquante |
| titre de 73 caractères | refusé : longueur |
| ligne de corps de 101 caractères, hors trailer et URL | refusé : longueur |
| `fixup! feat: add x` | valide en local, refusé en CI |

Ces exemples sont rejoués par `tests/test_commit_messages.py`.

Les options de pytest (`--strict-markers`, marqueurs, sélection par défaut) sont dans `pyproject.toml`. Les tests du wrapper, des features, de l'export et des soumissions sont skippés tant que le module ou `submission/<pilote>/` correspondant n'existe pas. Ils s'activent seuls ensuite, et `-rs` affiche la raison de chaque skip.

## Intégration continue

GitHub Actions lance deux jobs, **Ruff** et **Pytest**, sur chaque PR et chaque push vers `develop` et `main`. Le workflow **Commitlint** s'exécute sur chaque PR et à chaque modification de son titre.
- Il vérifie le **titre de la PR suivi de ` (#N)`**, c'est-à-dire le titre du commit que GitHub crée lors d'une fusion en squash. Le titre de la PR doit donc tenir en 65 caractères environ.
- Il vérifie aussi **chaque commit** de la PR, sans exception pour les fixup. Tant que le dépôt autorise aussi les fusions par merge commit et par rebase, ces commits peuvent arriver tels quels sur `develop`.
- Réglage recommandé : fusion en squash uniquement, avec le titre de la PR (`PR_TITLE`) comme titre de commit. Ce réglage n'est pas encore appliqué. Une fois en place, seul le titre de la PR ferait foi, et la vérification des commits pourrait se limiter au refus des fixup.
- L'environnement est installé avec `uv sync --locked` : la CI échoue si `uv.lock` n'est plus aligné sur `pyproject.toml`. Après toute modification des dépendances, lancer `uv lock` et committer le lock.
- La CI n'installe ni torch, ni CUDA, ni les binaires Git LFS.

Protection prévue sur `develop` et `main` :
- PR obligatoire, avec l'approbation d'un autre membre ;
- checks **Ruff**, **Pytest** et **Commitlint** requis, branche à jour avant fusion ;
- force-push et suppression interdits.

Le run de contrôle accepte `--map`, `--steps`, `--speed` et `--steer-gain`. Il affiche quelques champs de `get_step_info()`, puis renvoie `0` si tout s'est bien passé, `1` en cas d'erreur de simulation et `2` si la carte est inconnue. La liste des cartes s'affiche quand on passe un nom invalide.

## Organisation

```
src/crashlearn/     package applicatif (code de l'équipe)
scripts/            points d'entrée CLI (run de contrôle, …)
tests/              tests pytest (contrat simulateur, soumission, wrapper, features, export)
.github/workflows/  CI GitHub Actions (ci.yml, commitlint.yml)
.pre-commit-config.yaml  hooks Git locaux
vendor/simulation/  simulateur fourni, copie unique et non modifiée
vendor/PROVENANCE.md  origine et sha256 de l'archive
```

Le dossier `vendor/` est exclu de Ruff et de pytest. Toute correction du simulateur doit y faire l'objet d'un commit dédié, pour rester visible dans le diff.
