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

## 2026-10-09 · [Infrastructure] Validation de la station et périmètre de la documentation

E-5 / PR #12 : les reprises PPO depuis le checkpoint final et le dernier checkpoint périodique sont validées sur CUDA. La lecture T3 s’exécute sous l’utilisateur non privilégié et les travaux survivant dans une session `closing` restent détectés. Le seuil de charge 0,5 est conservé après une observation courte de 0,16 à 0,34 sans entraînement ni build ; ses limites et la procédure de recalibrage sont décrites dans `docs/station-gpu.md`. L’extinction réelle après une heure n’a pas été testée pendant cette validation.

La documentation versionnée conserve le matériel, les commandes reproductibles et l’exploitation utile au projet. Les notes de migration et de configuration personnelles sont retirées du dépôt de livraison.

## 2026-10-14 · [Reward] Reward v1 : progression en mètres, contacts journalisés sans poids

Remplace la v0 du 16/09 (E-15, `docs/mdp.md`, `src/crashlearn/reward.py`). +1 par mètre de progression signée, +20 par tour, +50 à l'arrivée, -200 en cas d'abandon. Légères pénalités pour la lenteur (-0,1 par pas sous 1 m/s), la marche arrière au-delà de 1 s (-0,1 par pas) et les à-coups d'action (-0,02 × ‖Δa‖²). Contacts mur et voiture à 0 mais journalisés. Pourquoi :

- **Mètres plutôt que `1000 × progress_delta`** : les circuits d'entraînement vont de 293 à 487 m. La v0 payait le même mètre 1,7 fois plus sur IMS que sur Sepang, et le même raccourci valait plus sur un petit circuit.
- **Primes de tour et d'arrivée réduites** (+100 et +300 en v0) : la progression paie déjà le tour. Les primes ne font que marquer l'événement.
- **DNF porté à -200** (-100 en v0) : un raccourci mortel comme celui de Sochi (+126 m puis DNF) doit rester perdant.

Mesuré avec `scripts/reward_scale.py` sur les 15 circuits d'entraînement : un pilote propre à 6 m/s gagne environ +1 330 par course, dont 92 % en progression. Rester à l'arrêt, reculer ou conduire au hasard coûte entre -220 et -255. Coefficients de départ, à ajuster sur les entraînements (E-18, E-19, E-24).

## 2026-10-14 · [MDP] Features uniques : 126 entrées, adversaires masqués aussi en solo

E-16 (`docs/features.md`, `src/crashlearn/features.py`). Une seule source NumPy pour les features, utilisée par l'environnement d'entraînement et recopiée dans les `agent.py`. Aux 105 entrées existantes (LiDAR, vitesse, braquage, phase, tours) s'ajoutent 3 emplacements adversaires de 7 valeurs : masque, position relative, lacet, vitesse et écart de course. Seuls les adversaires en course sont lus, triés du plus proche au plus loin. Une donnée manquante prend une valeur neutre. Pourquoi :

- **Même entrée en solo et pour le champion** : la politique solo se transfère sans changer d'architecture (ADR 0001).
- **Tri par distance** : la politique ne doit pas dépendre du numéro de voiture.
- **Lecture tolérante** : une exception dans l'agent vaut un DNF au tournoi.

L'ancien schéma d'adversaires du loader est abandonné, puisque l'archive du 18/09 n'en a plus qu'un.

## 2026-10-23 · [MDP] Mémoire temporelle par voiture, remise à zéro au moindre écart de pas

E-13 (`docs/history.md`, `src/crashlearn/history.py`). Les pilotes avec historique empilent leurs quatre dernières observations dans une mémoire indexée par `agent_id`, jamais sur l'instance d'`Agent`. Au début d'un épisode, la première observation est recopiée quatre fois. Pourquoi :

- **Par voiture** : rien n'oblige le tournoi à créer une instance par voiture, et `sim_recorder` prévoit d'en recopier une sur toutes les voitures. Avec le loader du 18/09, ce chemin n'est pas atteint, mais le harness du tournoi est inconnu.
- **Nouvel épisode dès que le pas n'est pas le suivant** : ça couvre le recul du compteur, deux épisodes courts qui se suivent au même pas et une mémoire périmée après un trou.
- **Dry-run reconnu** : l'observation factice du loader arrive au pas 0, juste avant le pas 1 de la course. Sans la reconnaître, elle entrerait dans l'historique de la voiture 0.
- **Copie de la première observation plutôt que des zéros** : c'est aussi ce qu'il faudra faire à l'entraînement (pas de `VecFrameStack`), pour que le réseau voie les mêmes débuts d'épisode qu'au tournoi.

Le pilote réactif reste sans état et sert de repli.

## 2026-10-30 · [Évaluation] Témoin déterministe : les 15 cartes d'entraînement sont bouclées

Pilote scripté sans apprentissage, qui vise la direction la plus dégagée du LiDAR (E-11, PR #5, `docs/scripted_driver.md`). Pourquoi : quand une métrique semble aberrante, rejouer la même course avec lui dit si le problème vient du simulateur, de la mesure ou de la politique RL. Sur l'archive du 18/09, aucun DNF ni contact mur : 3 tours seul sur 5 cartes (seeds 0 à 2), 3 tours à 4 voitures sur Austin et Budapest, 1 tour seul sur les 10 autres cartes.

Il a confirmé les défauts de la première archive : les DNF d'IMS, BrandsHatch et MexicoCity venaient du faux DNF après la ligne, et ils ont disparu à configuration de pilote inchangée. Temps de référence sur 3 tours, seed 0 : IMS 132,35 s, BrandsHatch 163,50 s, MexicoCity 176,80 s, Budapest 189,45 s, Austin 224,30 s. Tout agent RL doit au moins faire mieux que ces temps.

## 2026-10-30 · [Évaluation] Protocole E-18 : mesurer la mémoire contre un PPO réactif identique

Chaîne commune créée pour driver2 (E-18, `docs/results/e-18/README.md`) : `HistoryWrapper`, `train_ppo.py`, `evaluate_drivers.py`, `compare_drivers.py`. Le PPO à 4 observations est comparé à un PPO à 1 observation, même code, mêmes hyperparamètres, mêmes 5 seeds et même budget. Le pilote scripté reste un témoin. Pourquoi :

- **PPO réactif en baseline** : comparer un PPO à un script mesurerait l'apprentissage, pas la mémoire.
- **Critère écrit avant les résultats** : un intervalle bootstrap à 95 % par seed d'entraînement doit exclure 0, sinon la conclusion est « non concluant ».
- **Chronos sur les courses finies par les deux pilotes** : sinon, échouer sur les circuits durs améliorerait le chrono.
- **Échecs versionnés** dans `docs/results/e-18/`, puisque `runs/` est ignoré par git.

Conditions de robustesse : friction 0,5 et bruit LiDAR 0,1, bornes « forte pluie » du simulateur, sur les 4 circuits de validation. Budget à fixer après un banc de débit sur la station. `log_std_init = -1` pour les deux profondeurs : avec le défaut, l'écart-type de 1 sature l'action de bruit et la reward d'entraînement baisse sur 200 000 pas. Choix sur une seule seed, sans course finie : une validation sur la station passe avant la campagne.

## 2026-11-01 · [Reward] Reward v1 inchangée après revue : contacts, marche arrière et raccourci de Sochi

Revue de Thibaud sur la PR #14. Aucun coefficient ne change, pour ne pas invalider la campagne E-18 qui attend une reward stable. Trois corrections de l'entrée du 14/10 :

- **Contacts** : avec un poids nul, `wall_contact` et `vehicle_contact` valent toujours 0 dans `reward_terms`, donc ils n'étaient pas journalisés. Les contacts se comptent dans `info["wall_contact"]` et `info["vehicle_contact"]`, ce que fait maintenant `scripts/reward_scale.py`. La poursuite à 6 m/s frotte le mur 21 pas sur 15 circuits (Monza et MexicoCity).
- **Marche arrière** : la seconde de grâce échappe au terme `reverse`, pas à `slow`. Gardé tel quel : reculer d'un mur coûte un peu, ce qui ne gêne pas le dégagement.
- **Sochi** : le -74 ne tenait pas compte de l'actualisation. Le DNF tombe 200 pas après le raccourci, donc à γ = 0,99 il ne pèse plus qu'environ -27. Mesuré par Thibaud, le raccourci reste perdant face à la conduite propre, mais de très peu (+13,5 contre +15,3). À revoir si γ ou le DNF changent.

## 2026-11-01 · [Évaluation] Témoin : une voiture arrivée continue de rouler sur les courses courtes

Avec `--laps 1` ou `2`, le simulateur laisse une voiture arrivée `ACTIVE` sur la piste. `control_race.py` lui envoyait `(0, 0)` : elle s'arrêtait après la ligne et ralentissait les suivantes, même sans contact. Elle continue maintenant d'être conduite, et ses mesures restent figées à son arrivée. Pourquoi : un témoin ne doit pas fausser les temps des autres voitures. Référence Budapest à 1 tour et 2 voitures : voiture 0 64,65 → 64,50 s, voiture 1 inchangée (63,45 s). Les courses de 3 tours ne changent pas : le simulateur y fige la voiture arrivée.

## 2026-11-01 · [Évaluation] Protocole E-18 aligné sur le document d'expériences

Ajouts au protocole du 30/10 (E-18, `docs/results/e-18/README.md`), avant tout résultat. Pourquoi : le document Linear « Expériences, robustesse et évaluation » les demande, et le protocole devait être figé avant la campagne.

- **Condition `strict_margin`** : 4 circuits de validation avec la marge de sortie de piste de 0,50 m annoncée par `INSTRUCTIONS.md`, au lieu du 1,00 m du code. La pédagogie peut encore corriger le simulateur, et un pilote ne doit pas dépendre de cette tolérance. 105 courses par modèle au lieu de 93.
- **Métriques** : taux d'arrivée moyenné par carte, contacts par 100 m, immobilité, marche arrière, variation des commandes, sauts de progression et écart entre progression créditée et distance roulée (raccourcis).
- **Bruit LiDAR gardé à 0,1** (±10 %), au-delà des ±5 % de la plage « Champion » du document : c'est la pire condition annoncée pour le tournoi, alors que la plage du document sert au curriculum (E-23).


## 2026-11-09 · [Outillage] Runs reprenables et agents ONNX autonomes

E-20 / E-25 : chaque tentative garde ses logs CSV et son statut dans le manifest, y compris après une erreur ou SIGTERM. La reprise refuse un changement de seed, de prétraitement, de reward ou de distribution d'entraînement. W&B reste facultatif et hors ligne par défaut. L'export incorpore les features et l'historique canoniques dans un agent autonome, opset 17, poids externes, accord PyTorch/ONNX contrôlé à 10⁻⁵. Pourquoi : dix runs sur une station partagée doivent survivre aux interruptions, et le tournoi n'autorise que NumPy et ONNX Runtime.

## 2026-11-09 · [Entraînement] Cache de géométrie et un circuit par worker

Un reset Austin coûtait 0,341 s, presque tout dans le chargement LiDAR. Un cache de deux cartes par processus le ramène à environ 0,0003 s, avec des trajectoires identiques bit à bit et sans toucher `vendor/`. 15 workers portent chacun un circuit train (`--fixed-map-workers --n-envs 15`). Pourquoi : les interactions sont réparties uniformément par carte, et un changement de carte ne bloque plus tout le `SubprocVecEnv`.

## 2026-11-09 · [Entraînement] Budget de campagne : 5 000 000 interactions sur GPU

Banc d'un million d'interactions : 3 699 interactions/s en CPU, 6 088 en CUDA. Le banc CUDA finit déjà BrandsHatch, Budapest et IMS. La RTX 5070 est retenue. Budget fixé avant le premier run : 5 000 000 interactions demandées, 5 007 360 après arrondi SB3, pour les dix runs. Les bancs sont exclus de la comparaison.

## 2026-11-10 · [Évaluation] Campagne E-18 : apport de la mémoire non concluant

Dix PPO de fin de budget, cinq seeds et deux profondeurs, ont chacun leurs 105 courses d'évaluation ; le témoin ajoute 105 courses. La tentative réactive seed 4 interrompue par un redémarrage est conservée, et une relance à budget identique la remplace. Conclusion nominale appariée : « non concluant ». Les agents livrés sont choisis sur les finitions nominales de validation, puis la plus petite seed : `ppo-h1` seed 0 (7/12) et `ppo-h4` seed 1 (6/12). Résultats : `docs/results/e-18/README.md`.

## 2026-11-10 · [Entraînement] Fiabilité du réactif : adaptation et ablation de la progression

E-19 : aucun plafond de vitesse seul ne fait finir les quatre circuits de validation au réactif. Deux adaptations de deux millions d'interactions, plafond 5 m/s puis départs train diversifiés, finissent 10/12. Masquer les trois entrées de progression et de tours donne 12/12 : une dérivation met à zéro ces seules connexions, sans interaction ajoutée. Confirmation sur 120 courses nouvelles : 118/120, deux DNF à Montreal. Pourquoi : PPO est conservé pour isoler ces changements, et la fiabilité prime sur le chrono (médiane de 120,65 à 203,00 s sur les courses communes).

## 2026-11-10 · [Livraison] driver1 à 400/400 avec un plafond d'inférence de 4,75 m/s

La cible passe à 100 % observés. Les 132 anciennes courses deviennent des cas de sélection, et 400 courses nouvelles, seeds 200 à 299, sont fixées avant la confirmation. Un plafond global de 4,75 m/s corrige les deux DNF, sans réglage par circuit et avec les mêmes poids appris. Le candidat figé finit 400/400. Coût : médiane de 236,17 à 247,57 s sur 118 courses communes. Faible adhérence et forte pluie restent à 6/12 (E-23). Preuves : `docs/results/driver1/README.md`.

## 2026-11-12 · [Livraison] Modèles livrés sans LFS, provenance contrôlée

E-36 demandait Git LFS pour les modèles livrés. On garde Git simple, comme décidé dans `docs/remise.md` : les fichiers font moins de 800 Ko, et une récupération qui ignore LFS ne verrait que des pointeurs. Pour l'intention du critère, `scripts/check_deliverables.py` vérifie en CI les hashes de `export.json` et relie chaque modèle à son manifest (commit, configuration, seed), à ses ancêtres et à ses résultats. Un clone neuf passe le contrôle, et les deux modèles sont restaurés depuis les archives de la station puis ré-exportés à l'octet près. Pourquoi : la remise doit rester récupérable et prouvable sans outil en plus. Détails : `docs/remise.md`.
