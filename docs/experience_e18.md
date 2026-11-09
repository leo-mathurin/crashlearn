# Expérience E-18 : la mémoire aide-t-elle driver2 ?

**État au 09/11 : premiers PPO capables de finir des courses ; campagne de dix runs lancée.**

On entraîne un PPO qui décide sur ses 4 dernières observations, et on le compare à un PPO réactif dans les mêmes conditions. La question est de savoir si la mémoire justifie sa complexité, puisque le pilote réactif reste le repli (`docs/history.md`).

Le protocole est figé **avant** la campagne. Tout changement après le premier résultat doit être écrit ici et au memlog, avec sa raison.

## Chaîne

| Élément | Rôle |
|---|---|
| `src/crashlearn/history_env.py` | `HistoryWrapper` empile les features avec `CarHistory`, comme `agent.py` au tournoi |
| `scripts/train_ppo.py` | Entraîne un pilote et écrit `manifest.json`, `model.zip`, `progress.csv`, `checkpoints/` |
| `src/crashlearn/evaluation.py`, `scripts/evaluate_drivers.py` | Font courir n'importe quel pilote sur les mêmes courses |
| `scripts/compare_drivers.py` | Produit le rapport Markdown à partir des JSON |

`--history-depth 1` et `--history-depth 4` passent par le même code. `CarHistory` est utilisé à la place de `VecFrameStack` de SB3, qui remplit le début d'épisode avec des zéros. Toutes les courses passent par `CrashLearnEnv`, seul chemin qui expose friction et bruit LiDAR : une graine y désigne la même course pour tous les pilotes.

## Protocole

**Pilotes** :
- `ppo-h4` : 4 observations, 504 entrées, candidat pour driver2 ;
- `ppo-h1` : 1 observation, 126 entrées, baseline réactive apprise ;
- `scripted` : témoin d'E-11. Il sert de repère et de contrôle de la mesure. Le comparer seul à `ppo-h4` ne dirait rien de la mémoire.

**Entraînement** :
- 15 circuits d'entraînement, conditions nominales ;
- seeds 0 à 4, les mêmes pour h1 et h4 ;
- `PPO_CONFIG` de `train_ppo.py` pour les deux, sans réglage séparé : mêmes couches `[256, 256]`, h4 a seulement plus d'entrées ;
- budget identique, fixé avant la campagne (section suivante) ;
- le modèle évalué est celui de fin de budget, sans sélection du meilleur checkpoint.

**Évaluation** : politique déterministe, 3 tours, 8000 décisions au plus (400 s simulées), seeds 0 à 2.

| Condition | Friction | Bruit LiDAR | Circuits |
|---|---:|---:|---|
| `nominal` | Officielle | Officiel | 15 train + 4 validation |
| `low_friction` | 0,5 | Officiel | 4 validation |
| `lidar_noise` | Officielle | 0,1 | 4 validation |
| `heavy_rain` | 0,5 | 0,1 | 4 validation |
| `strict_margin` | Officielle | Officiel | 4 validation, marge de sortie de piste de 0,50 m |

Soit 105 courses par modèle. Le test scellé n'est jamais utilisé, et `campaign()` ne peut pas le renvoyer.

- **Friction 0,5 et bruit 0,1** : ce sont les bornes « forte pluie » de `env_simulation.py`, et `INSTRUCTIONS.md` annonce le même ±10 % de bruit LiDAR sous forte pluie. C'est au-delà de la plage « Champion » du document Linear « Expériences, robustesse et évaluation » (±5 %), que ce document marque lui-même « à revoir ». On garde 0,1 : l'évaluation doit couvrir la pire condition annoncée du tournoi, alors que la plage du document sert au curriculum d'entraînement (E-23).
- **`strict_margin`** : le code tolère 1,00 m au-delà du bord de piste avant le DNF, alors que `INSTRUCTIONS.md` annonce 0,50 m, et la pédagogie dit que le simulateur peut encore être corrigé (memlog du 06/10). On se fie au code pour tout le reste, mais le document Linear demande d'évaluer aussi avec la valeur annoncée, pour qu'un pilote ne dépende pas d'une tolérance fragile. `evaluation.py` change `_OFFTRACK_MARGIN_M` du simulateur pendant la course, puis le remet, sans modifier `vendor/`.

**Métriques** :
- **Fiabilité** : taux d'arrivée, DNF, TIMEOUT. Toutes les courses comptent.
- **Chronos** : temps total sur 3 tours. La comparaison h4 / h1 n'utilise que les courses finies par les deux, sinon un pilote qui échoue sur les circuits durs paraîtrait plus rapide.
- **Robustesse** : écart du taux d'arrivée par rapport au nominal, sur les circuits de validation.
- **Fiabilité par carte** : le taux d'arrivée est aussi moyenné carte par carte, pour qu'une carte ne pèse pas plus qu'une autre.
- **Contacts** : pas en contact mur et voiture, lus dans `info`, par course et pour 100 m parcourus. Les `reward_terms` de contact valent 0 puisque leur poids est nul.
- **Comportement** : pas immobiles (moins de 0,1 m/s), pas en marche arrière (moins de -0,1 m/s), variation moyenne des commandes (‖Δa‖² par pas).
- **Raccourcis** : un saut de progression est un pas où le simulateur crédite plus d'1 m au-delà de la distance roulée. L'écart entre la progression créditée et la distance roulée est aussi relevé. Il est d'environ +10 à +36 m sur 3 tours pour le témoin, qui coupe un peu les virages (validation, seed 0), et bien plus grand pour une coupe à travers un mur.
- **Cible de référence** : le document Linear vise au moins 90 % de courses nominales terminées en validation pour la baseline (E-19). C'est un repère, pas une règle de décision entre h1 et h4.

**Critère de décision** : on calcule h4 moins h1 pour chaque seed d'entraînement, puis un intervalle bootstrap à 95 % de la moyenne. Dans chaque couple partition / condition :
- h4 est plus ou moins fiable si l'intervalle du taux d'arrivée exclut 0 ;
- sinon, h4 est plus rapide ou plus lent si l'intervalle du chrono exclut 0 ;
- sinon, ou avec moins de 3 seeds appariées : **non concluant**.

Avec 5 seeds, l'intervalle reste grossier. « Non concluant » n'est pas la preuve que la mémoire est inutile.

## Échecs conservés

- `runs/` est ignoré par git : les JSON de course et les manifests vont dans `docs/results/e-18/`.
- Un run écrit son manifest dès le départ, en `running`, puis `completed` ou `failed` avec la trace d'erreur.
- `train_ppo.py` refuse d'écraser un dossier de run.
- Une seed plantée ou divergente reste listée dans le rapport. Une relance a un nouveau dossier et est mentionnée ici.
- Les DNF et TIMEOUT sont des enregistrements normaux, avec un code de sortie 0.

## Fixer le budget

### Pilotes sur poste de développement

Ces runs servent à régler la chaîne. Ils ne font pas partie de la campagne et ne disent rien de la mémoire : une seule seed, 200 000 pas, 8 environnements, aucune course finie.

| Run | Reward d'entraînement (début → fin) | Durée d'épisode (pas) | Évaluation déterministe, tours parcourus (IMS / Budapest / Sochi) |
|---|---|---|---|
| h4, `log_std_init` 0 | -226 → -237 | 237 → 552 | 0,66 / 0,15 / 0,36 |
| h1, `log_std_init` 0 | -226 → -240 | 247 → 666 | 0,31 / 0,15 / 0,21 |
| h1, `log_std_init` -1 | -216 → -188 | 244 → 444 | 0,10 / 0,15 / 0,20 |

**Décision** : `log_std_init = -1` pour h1 et h4. C'est le seul run dont la reward d'entraînement progresse, mais l'évaluation déterministe ne le confirme pas encore. Ce choix repose sur une seule seed. Il a été essayé sur h1 seulement, pour ne pas régler le candidat.

**Débit** : 2 100 pas/s pour un processus seul, sans perte notable à 4 processus indépendants. En entraînement, le débit tombe à 350-530 pas/s parce qu'un `reset` avec changement de carte coûte 600 ms et bloque tout le `SubprocVecEnv`. Le débit monte donc avec la durée des épisodes : un banc de 200 000 pas le sous-estime.

### Avant la campagne

1. **Banc** : 1 000 000 de pas sur h1, une fois en `cpu`, une fois en `cuda`, avec la commande de la section suivante. `docs/station-gpu.md` note que le CPU peut être plus rapide sur un petit MLP.
2. **Validation** : évaluer ce modèle. Si `ppo-h1` ne finit aucune course sur les circuits d'entraînement, ne pas lancer la campagne : revoir d'abord `PPO_CONFIG` ou la reward, puis noter le changement ici.
3. **Budget** : choisir une valeur qui tient dans le temps disponible pour 10 entraînements, et la noter ci-dessous avant le premier résultat.

| Mesure | Valeur |
|---|---|
| Débit station | CPU : 3 699,3 ; CUDA : 6 088,0 interactions/s, banc du 09/11 |
| Appareil retenu | RTX 5070, CUDA 13.0 |
| Budget par entraînement | 5 000 000 demandées, 5 007 360 interactions effectives |
| Temps total | Estimation avant campagne : 2 h 20 de calcul, hors évaluations |

## Commandes

Sur la station, depuis un commit propre de `develop`. #14, #15 et #17 sont mergées depuis le 01/11 : la reward v1 et les features sont figées. Toute modification invaliderait la campagne, et les `*_sha256` du manifest la rendraient visible.

```bash
export BUDGET=...   # valeur du tableau ci-dessus
export DEVICE=...
run() {
  docker run --rm --runtime=nvidia --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace \
    -e PYTHONPATH=/workspace/src -e GIT_COMMIT="$(git rev-parse HEAD)" -e HOME=/tmp \
    crashlearn-train-gpu:20261109 "$@"
}

for seed in 0 1 2 3 4; do
  for depth in 1 4; do
    run python scripts/train_ppo.py --history-depth "$depth" --seed "$seed" \
      --timesteps "$BUDGET" --device "$DEVICE" --fixed-map-workers --n-envs 15 \
      --output-dir "runs/e-18/ppo-h$depth-seed$seed"
  done
done

mkdir -p docs/results/e-18/manifests
for dir in runs/e-18/*/; do
  name=$(basename "$dir")
  cp "$dir/manifest.json" "docs/results/e-18/manifests/$name.json"
  run python scripts/evaluate_drivers.py --run-dir "$dir" --output "docs/results/e-18/$name.json"
done
run python scripts/evaluate_drivers.py --scripted --output docs/results/e-18/scripted.json
run python scripts/compare_drivers.py --results-dir docs/results/e-18 --output docs/results/e-18/rapport.md
```

Ajouter `--wandb` à `train_ppo.py` pour suivre le run en direct. Si W&B échoue, l'entraînement continue et `progress.csv` reste la référence. Un run interrompu reprend avec `--resume runs/e-18/<run>/checkpoints/<fichier>.zip`. Le manifest garde `resumed_from`, et l'état aléatoire des environnements n'étant pas restauré, ce run n'est plus reproductible à l'identique.

Vérifications hors CI, avec les dépendances d'entraînement :

```bash
uv sync --group train
uv run pytest -m slow tests/test_train_ppo.py
```

## Limites

- **Station validée le 09/11** : les deux bancs utilisent l'image Docker GPU reconstruite ;
  son identifiant et les versions effectives sont dans les manifests.
- **Premières courses terminées le 09/11** : le banc CUDA finit BrandsHatch, Budapest et IMS.
  La fiabilité sur la validation reste à mesurer dans la campagne principale.
- **Document Linear** : « Expériences, robustesse et évaluation » a été relu le 01/11. Le protocole y est aligné : marge de 0,50 m, métriques de comportement et moyenne par carte ont été ajoutées. Le seul écart est le bruit LiDAR, justifié plus haut.
- **Solo uniquement** : aucune course avec adversaires.
- **Robustesse hors entraînement** : les perturbations ne sont pas vues à l'entraînement (E-23).
- **Coût de calcul non comparé** : h4 reçoit quatre fois plus d'entrées, et le temps d'inférence n'est pas mesuré ici. Le contrôle reviendra à l'export (E-25).
- **Reward v1** : selon la review de #14, le raccourci de Sochi peut rester rentable avec γ = 0,99. Sochi étant en validation, un chrono anormalement bas s'y vérifie d'abord avec les contacts mur et le témoin.

## Résultats

Campagne lancée le 09/11/2026, résultats et manifests dans `docs/results/e-18/`.
Le rapport sera produit après les dix entraînements et leurs évaluations.

## Préparation station du 9 novembre

Avant le banc et toute campagne : cache de géométrie LiDAR limité à deux cartes par worker,
avec équivalence exacte des trajectoires testée. `--fixed-map-workers --n-envs 15` affecte un
circuit train à chaque worker. Les deux profondeurs reçoivent le même budget et le même
échantillonnage uniforme en interactions. Le banc CPU/CUDA demande 1 000 000 d'interactions
par appareil, arrondies à 1 013 760 par les rollouts de 30 720. Les threads PyTorch sont fixés
à 1 ; les 15 simulations tournent dans des processus séparés. Aucun circuit test n'est ouvert.


## Budget figé le 9 novembre avant la campagne multi-seeds

Le banc CUDA finit trois courses nominales (BrandsHatch, Budapest, IMS), seed d'évaluation 0,
sur les quinze circuits train ; le banc CPU finit IMS. La validation préalable est passée.
Les autres courses et tous les DNF restent dans `docs/results/station-20261109/`.

La campagne principale repart de zéro, seeds 0 à 4, profondeurs 1 et 4, budget demandé de
5 000 000 d'interactions par modèle. Les pilotes du banc sont exclus de la comparaison.
Le même `PPO_CONFIG`, le cache, les 15 workers fixes et le budget s'appliquent aux dix runs.
Les évaluations gardent les 105 courses prévues, et aucun circuit test n'est utilisé.

La mesure CPU a partagé les deux premières minutes avec la suite de régression locale ;
ce banc établit un ordre de grandeur sur la station, pas un classement matériel isolé.
Le débit CUDA a été mesuré après ces tests. Les modèles diffèrent entre CPU et CUDA malgré
la même seed : le protocole ne suppose pas une identité numérique entre appareils.

Reward et features restent figées. Un résultat insuffisant de la campagne sera conservé,
avec la cible de 90 % de finitions en validation toujours indiquée comme non atteinte.
