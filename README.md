# Crash & Learn

Projet de conduite autonome par reinforcement learning — Léo Mathurin, Lucas Noirie et Thibaud Combaz Deville.

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

Dépôt : [EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8](https://github.com/EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8) (`origin`).

Branches persistantes prévues : `develop` pour l'intégration, `main` pour les jalons validés. Branches courtes, PR vers `develop`, revue croisée puis promotion vers `main` avec tag et métriques. Backlog partagé, tâches prises librement et aucun domaine réservé. Conventional Commits.

Socle retenu : Python 3.11, uv, Gymnasium, NumPy, Stable-Baselines3/PyTorch, PPO et SAC, ONNX/ONNX Runtime. Entraînements sur la station Ubuntu de Léo ; W&B avec suivi local de secours. Les versions et commandes seront ajoutées lors de l'initialisation.

Le code n'est pas encore initialisé. Le cadrage est maintenu dans Linear ; ce dépôt accueillera le code, les tests et la documentation technique générée. Les ADR et le glossaire du cadrage sont conservés dans les documents Linear.
