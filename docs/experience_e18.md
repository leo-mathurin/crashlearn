# Expérience E-18 : la mémoire aide-t-elle driver2 ?

**État : chaîne prête, campagne pas encore lancée. Aucun résultat ci-dessous ne mesure l'apport de la mémoire.**

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

Soit 93 courses par modèle. Les valeurs 0,5 et 0,1 sont les bornes « forte pluie » de `env_simulation.py`. Le test scellé n'est jamais utilisé, et `campaign()` ne peut pas le renvoyer.

**Métriques** :
- **Fiabilité** : taux d'arrivée, DNF, TIMEOUT. Toutes les courses comptent.
- **Chronos** : temps total sur 3 tours. La comparaison h4 / h1 n'utilise que les courses finies par les deux, sinon un pilote qui échoue sur les circuits durs paraîtrait plus rapide.
- **Robustesse** : écart du taux d'arrivée par rapport au nominal, sur les circuits de validation.
- **Contacts** : pas en contact mur, lus dans `info["wall_contact"]`. Les `reward_terms` de contact valent 0 puisque leur poids est nul.

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
| Débit station | À mesurer |
| Appareil retenu | À mesurer |
| Budget par entraînement | À fixer |
| Temps total | À mesurer |

## Commandes

Sur la station, depuis un commit propre. Avant le lancement, vérifier avec Lucas que #14, #15 et #17 ne changeront plus la reward ni les features : ces changements invalideraient la campagne, et les `*_sha256` du manifest le rendraient visible.

```bash
export BUDGET=...   # valeur du tableau ci-dessus
export DEVICE=...
run() {
  docker run --rm --runtime=nvidia --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace \
    -e PYTHONPATH=/workspace/src -e GIT_COMMIT="$(git rev-parse HEAD)" -e HOME=/tmp \
    crashlearn-train-gpu:20260928 "$@"
}

for seed in 0 1 2 3 4; do
  for depth in 1 4; do
    run python scripts/train_ppo.py --history-depth "$depth" --seed "$seed" \
      --timesteps "$BUDGET" --device "$DEVICE" --output-dir "runs/e-18/ppo-h$depth-seed$seed"
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

- **Station non testée** : la chaîne a été validée sur un poste de développement, pas dans l'image Docker de la station.
- **Capacité à finir non prouvée** : aucun pilote n'a encore bouclé une course. `PPO_CONFIG` n'est donc pas validé tant que l'étape de validation n'est pas passée.
- **Doc Linear non consultée** : « Expériences, robustesse et évaluation » n'était pas accessible à l'écriture de ce protocole. S'il impose d'autres conditions, aligner cette page avant la campagne.
- **Solo uniquement** : aucune course avec adversaires.
- **Robustesse hors entraînement** : les perturbations ne sont pas vues à l'entraînement (E-23).
- **Coût de calcul non comparé** : h4 reçoit quatre fois plus d'entrées, et le temps d'inférence n'est pas mesuré ici. Le contrôle reviendra à l'export (E-25).
- **Reward v1** : selon la review de #14, le raccourci de Sochi peut rester rentable avec γ = 0,99. Sochi étant en validation, un chrono anormalement bas s'y vérifie d'abord avec les contacts mur et le témoin.

## Résultats

Campagne non lancée.
