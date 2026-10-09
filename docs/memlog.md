# Memlog

Journal chronologique du projet : décisions, changements et mesures qui en ont décidé d'autres. Il sert à expliquer le projet en entrevue et à écrire le rapport final (E-42). Le détail reste dans Linear, les PR et `docs/`.

Règles :

- **Ajouter en bas, ne jamais réécrire.** Si une décision change, on ajoute une nouvelle entrée qui renvoie à l'ancienne, et on explique pourquoi.
- Une entrée par décision, changement de reward, d'observation ou de fin d'épisode, résultat d'entraînement marquant ou échange avec la pédagogie.
- Format : `## AAAA-MM-JJ · [Tag] Titre`, puis quelques lignes : quoi, **pourquoi**, lien (issue, PR, doc).
- Tags : `Organisation`, `Simulateur`, `Reward`, `MDP`, `Entraînement`, `Évaluation`, `Pédagogie`, `Outillage`.

---

## 2026-09-14 · [Organisation] Lancement du projet

Équipe E-Hamilton : Léo Mathurin, Lucas Noirie et Thibaud Combaz Deville. Rendu le 07/02/2027 à 23:42. Objectif : trois pilotes solo et un champion multi-agent, avec un dashboard de replays en bonus. Cadrage et backlog dans le projet Linear « Crash & Learn ».

## 2026-09-16 · [MDP] Stack et MDP de départ

Python 3.11 + uv, Gymnasium, Stable-Baselines3 (PPO en baseline, SAC en comparaison), export ONNX. Observation solo : 100 rayons LiDAR, vitesse, braquage, progression et tour. Action normalisée, remappée vers [-2, 10] m/s et ±0,4189 rad. On garde la marche arrière pour se dégager d'un mur. Pourquoi : c'est la chaîne imposée par la soumission (ONNX et NumPy seulement dans `agent.py`), et SB3 couvre PPO et SAC. Doc Linear « Architecture RL, MDP et décisions ».

## 2026-09-16 · [Reward] Reward initiale proposée (v0, non validée)

`1000 × progress_delta`, +100 par tour, +300 à l'arrivée, -100 en cas d'abandon, -5 par contact avec un mur, -10 par contact avec une voiture. Pénalités légères pour la stagnation, le recul inutile et les à-coups d'action. Pourquoi : suivre la progression plutôt que la vitesse brute ou la simple survie. Coefficients à vérifier avant de les fixer (E-15).

## 2026-09-16 · [Simulateur] Import de la première archive, sans modification

`simulation.zip` (08/09) est vendorée dans `vendor/simulation/` (E-8). Pourquoi : une seule copie officielle, pour s'entraîner sur le simulateur du tournoi.

## 2026-09-16 · [Simulateur] Audit de la première archive : six défauts relevés

Friction figée, schémas d'observation incohérents, `Mexico City` inchargeable, `set_map` non atomique, murs traversables par fluage… (E-10, `docs/audit_simulateur.md`). Décision du moment : corriger le minimum localement (E-12).

## 2026-09-18 · [Outillage] CI et conventions

Ruff, Pytest et Commitlint en CI sur les PR vers `develop`. Conventional Commits, merge en squash, relecture croisée (E-6, E-17).

## 2026-09-23 · [Évaluation] Partitions de circuits figées

23 circuits : 15 en train, 4 en validation, 4 en test scellé, ouvert une seule fois après le gel du 17/01/2027 (E-9, `docs/tracks.md`). Pourquoi : mesurer la généralisation à des circuits jamais vus, sans régler nos choix sur le test.

## 2026-09-27 · [Pédagogie] Exploiter les bugs du simulateur, ne pas les corriger

Consigne de l'équipe pédagogique : les bugs du simulateur peuvent être exploités (traverser un mur, se servir des contacts ; il n'y a pas de casse). Conséquences :

- E-12 est annulée et sa PR #7 fermée sans merge : corriger le moteur nous aurait éloignés du simulateur du tournoi (ADR 0002 révisé).
- ADR 0004 : tout comportement que le simulateur officiel récompense est légitime.
- Le format des courses finales est inconnu, et deux de nos voitures ont peu de chances d'être dans la même course : aucune tactique ne repose sur un coéquipier.

## 2026-09-27 · [Simulateur] Nouvelle archive officielle du 18/09

`simulation_20260918.zip` est vendorée sans modification (E-46, PR #8, `vendor/PROVENANCE.md`). Nouvelles règles : DNF si la progression est inférieure à 1 m en 10 s, DNF immédiat en cas de sortie de piste (marge de 1,00 m dans le code, 0,50 m annoncée), rebond en marche arrière sur un mur, largeurs de piste mesurées. Versions du moteur et d'ONNX épinglées sur le Dockerfile officiel.

## 2026-09-27 · [Reward] Pénalités de collision remises en cause

Les pénalités de -5 (mur) et -10 (voiture) de la reward v0 sont suspendues : on ne les garde que si E-47 montre qu'un contact coûte plus qu'il ne rapporte. Pourquoi : ADR 0004. Un contact peut être rentable (mur fin traversé, poussée par l'arrière), et le pénaliser par principe ferait apprendre à l'agent à éviter ce qui paie.

## 2026-09-27 · [Simulateur] Premiers exploits mesurés

Sur Zandvoort, la voiture traverse un mur de 2,66 m en 249 pas : +78 m crédités, et elle reste active. Sur Sochi : +126 m, puis DNF. Poussée par l'arrière sur Spa : +4,5 m en 5 s (E-10, PR #6, `docs/audit_simulateur.md`). Suite du travail dans E-47.

## 2026-09-28 · [MDP] Wrapper Gymnasium et fins d'épisode

`crashlearn.env.CrashLearnEnv` (E-14, PR #9). `terminated` quand la voiture finit ou fait un DNF, tel que décidé par le simulateur ; `truncated` à 8000 pas (400 s). Un contact ne termine jamais l'épisode : sinon l'agent apprendrait à éviter des contacts qui peuvent être rentables, et le simulateur ne les sanctionne pas. Friction et bruit LiDAR réglables de notre côté, sans toucher `vendor/`.

## 2026-09-28 · [Reward] Reward par défaut du wrapper : progression seule (provisoire)

`1000 × progress_delta` seulement, en attendant la calibration d'E-15. Pourquoi : c'est le cœur de la v0, sans les pénalités de collision suspendues ; les bonus de tour et d'arrivée restent à mesurer.

## 2026-09-28 · [Organisation] Création de ce memlog

À l'entrevue, la pédagogie attend qu'on sache expliquer le projet, en particulier les reward functions, leur historique et les raisons de chaque changement (E-48).

## 2026-10-06 · [Pédagogie] Marge de sortie de piste : se fier au code

`env_simulation.py` fixe `_OFFTRACK_MARGIN_M = 1.00` m, alors que `INSTRUCTIONS.md` annonce une tolérance de 0,50 m. Réponse de la pédagogie : le problème est remonté, la doc n'a sans doute pas été mise à jour, et le simulateur peut encore être corrigé d'ici le tournoi. Décisions :

- On se fie au code, mais la marge n'est jamais recopiée en dur : on la lit dans le simulateur.
- `test_offtrack_margin_differs_from_the_instructions` sert d'alarme : il cassera si une nouvelle archive change la marge.
- Les exploits qui reposent sur le recouvrement des couloirs (8 circuits, `docs/audit_simulateur.md`) restent des bonus fragiles, à revérifier à chaque archive. Ils ne doivent pas guider la reward principale.

## 2026-10-06 · [Organisation] Remise sans Git LFS

Accès au dépôt Epitech vérifié (E-7, `docs/remise.md`) : la remise sera un fast-forward de notre `main`. Les modèles ONNX sont versionnés directement, sans Git LFS. Pourquoi : quelques Mo par modèle, loin des limites de GitHub, alors qu'une récupération du rendu qui ne suit pas LFS ne trouverait que des pointeurs à la place des modèles.
