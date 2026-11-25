# E-57 · driver2 avec les recettes de fiabilité de driver1

## Protocole (figé le 2026-11-25, avant le premier run)

**Question.** Dans E-18, la mémoire de 4 observations aide nettement sur les circuits d'entraînement (93,8 % contre 76,9 %), pas sur la validation (45,0 % contre 41,7 %). E-60 le confirme sur driver2 : 100 % à l'entraînement, 33 % en validation, et ses abandons suivent le circuit d'origine, modifié ou non. Hypothèse : la mémoire sert à s'adapter (glissade, adversaire), à condition que le pilote ne puisse plus apprendre ses circuits par cœur. On lui retire donc ce qui permet de les réciter.

**Recette commune** (celle qui a fiabilisé driver1, sans correction après coup) :

| Réglage | Option de `train_ppo.py` | Effet |
|---|---|---|
| Départs aléatoires | `--random-start` | Le pilote ne part plus toujours du même point |
| Météo changeante | `--weather` | Adhérence et bruit LiDAR changent pendant la course, comme au tournoi |
| Sans progression | `--progress-blind` | Les entrées « où suis-je sur le tour » valent zéro, et leurs poids sont nuls dès la création : ils ne reçoivent aucun gradient et restent nuls |
| Plafond de vitesse | `--max-speed 5` | Le plafond d'entraînement de driver1 |
| Circuits modifiés | `--track-variants 0.3` | 30 % des épisodes sur une variante du circuit d'entraînement (E-60) : sens inverse, miroir, couloir rétréci, virage ou tronçon modifié |

Les circuits modifiés font partie de la recette : E-60 montre que driver2 récite ses circuits, et c'est l'outil fait pour l'en empêcher. Contrepartie acceptée : on ne sépare pas leur effet de celui du reste de la recette.

**Bras.** Mêmes `PPO_CONFIG`, budget de 5 007 360 interactions, 15 workers fixés sur les 15 circuits d'entraînement (`--n-envs 15 --fixed-map-workers`), seeds 0 à 4, modèle de fin de budget :

| Bras | Mémoire | Ajout | Runs |
|---|---|---|---:|
| `h1` | 1 observation | recette | 5 |
| `h4` | 4 observations | recette | 5 |
| `h4-dropout` | 4 observations | recette + `--feature-dropout 0.1` | 5 |

**Évaluation.** `evaluate_drivers.py --run-dir <run> --variants`, seeds 0 à 2 : la campagne habituelle (nominal sur entraînement et validation, perturbations sur validation) plus les 26 circuits modifiés du banc E-60, soit 195 courses par modèle.

**Critère, repris d'E-18.** Par seed d'entraînement, différence `h4 − h1` (puis `h4-dropout − h4`), intervalle bootstrap à 95 % sur les 5 seeds :
1. sur le taux d'arrivée des circuits jamais vus en nominal (validation et banc, 90 courses par modèle) : plus ou moins fiable si l'intervalle exclut 0 ;
2. sinon, sur le chrono des courses finies par les deux : plus rapide ou plus lent si l'intervalle exclut 0 ;
3. sinon, ou avec moins de 3 seeds : non concluant.

Le résultat va dans le memlog, que la mémoire gagne ou non. Une piste qui ne bat pas sa référence est abandonnée.

**Où.** Sur la station GPU de Léo : 15 runs de 5 M d'interactions, environ 3 h 30 à 6 000 interactions/s (E-18), sans doute un peu plus avec les circuits modifiés, qui ne profitent pas du cache des cartes. Sur un portable à 8 cœurs, la même campagne prendrait environ 31 h (667/s mesurés).

```bash
for seed in 0 1 2 3 4; do
  for depth in 1 4; do
    uv run python scripts/train_ppo.py --history-depth $depth --seed $seed \
      --timesteps 5007360 --n-envs 15 --fixed-map-workers --device cuda \
      --random-start --weather --progress-blind --max-speed 5 --track-variants 0.3 \
      --output-dir runs/e-57/h$depth-seed$seed --wandb --wandb-group e-57
  done
  uv run python scripts/train_ppo.py --history-depth 4 --seed $seed \
    --timesteps 5007360 --n-envs 15 --fixed-map-workers --device cuda \
    --random-start --weather --progress-blind --max-speed 5 --track-variants 0.3 \
    --feature-dropout 0.1 --output-dir runs/e-57/h4-dropout-seed$seed --wandb --wandb-group e-57
done
```

Puis, pour chaque run : `uv run python scripts/evaluate_drivers.py --run-dir runs/e-57/<run> --variants --output docs/results/e-57/<run>.json`. Le script de comparaison par bootstrap (retiré après E-18) est à réécrire avant de lire les résultats.

## Résultats

À venir.
