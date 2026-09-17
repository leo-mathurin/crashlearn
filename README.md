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

Qualité :

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Le run de contrôle accepte `--map`, `--steps`, `--speed` et `--steer-gain`. Il affiche quelques champs de `get_step_info()`, puis renvoie `0` si tout s'est bien passé, `1` en cas d'erreur de simulation et `2` si la carte est inconnue. La liste des cartes s'affiche quand on passe un nom invalide.

## Organisation

```
src/crashlearn/     package applicatif (code de l'équipe)
scripts/            points d'entrée CLI (run de contrôle, …)
tests/              tests pytest
vendor/simulation/  simulateur fourni, copie unique et non modifiée
vendor/PROVENANCE.md  origine et sha256 de l'archive
```

Le dossier `vendor/` est exclu de Ruff et de pytest. Toute correction du simulateur doit y faire l'objet d'un commit dédié, pour rester visible dans le diff.
