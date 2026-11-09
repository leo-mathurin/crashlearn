# Entraînements, suivi et reprise

Chaque run garde son identifiant, le commit, les empreintes du simulateur, des features,
de l'historique et de la reward, les versions, la seed et la configuration. Dans Docker,
passer `GIT_COMMIT` et `TRAIN_IMAGE_ID` pour identifier les sources et l'image exécutées.

```bash
uv sync --locked --group train
uv run python scripts/train_ppo.py --history-depth 1 --seed 0 --timesteps 1000000 \
  --device cuda --output-dir runs/baseline --wandb --wandb-group baseline
```

Le manifest est écrit avant le calcul, puis à la fin, y compris après une erreur ou SIGTERM.
Il distingue les tentatives, leurs durées et leurs checkpoints de départ. Les logs CSV vont
vers `logs/attempt-NNN/progress.csv`. Les checkpoints sont locaux, tous les 100 000 pas par
défaut ; le modèle final est `model.zip`. Les pas sont des interactions agrégées sur les
environnements. SB3 arrondit le budget au prochain rollout complet.

## W&B

`--wandb` utilise le mode **hors ligne par défaut** : métriques et artefacts de checkpoints
restent sur la station. Une panne de W&B ne supprime pas les logs CSV et n'arrête pas le run.
Pour un projet partagé déjà configuré, fournir `WANDB_API_KEY` hors Git et ajouter
`--wandb-mode online --wandb-entity <équipe>`. Le projet est `crashlearn` par défaut ;
`--wandb-project` et `--wandb-group` permettent de le choisir.

Avant de synchroniser une campagne, vérifier le quota de stockage du compte et le volume
local des checkpoints. Aucun compte, abonnement payant ni secret n'est créé par le code.

```bash
du -sh runs/baseline
# Après configuration du compte et vérification du quota :
uv run wandb sync --entity <équipe> --project crashlearn \
  runs/baseline/logs/attempt-001/wandb/offline-run-*
```

La reprise en ligne réutilise l'identifiant W&B. Les tentatives hors ligne ont des fichiers
séparés ; ne synchroniser qu'une fois chaque tentative. Les métriques ont le compteur global
d'interactions SB3. Un retour à un checkpoint antérieur peut rejouer une plage de compteurs.

## Reprise après interruption

Garder le même dossier, la même seed, profondeur et le même nombre d'environnements. Le
checkpoint doit être dans ce dossier. Une modification du prétraitement, de la reward ou du
simulateur est refusée : démarrer un nouveau run pour mesurer cette modification.

```bash
uv run python scripts/train_ppo.py --history-depth 1 --seed 0 --timesteps 2000000 \
  --device cuda --output-dir runs/baseline \
  --resume runs/baseline/checkpoints/ppo_attempt001_1000000_steps.zip --wandb
```

Le modèle recharge les poids et l'optimiseur. L'état aléatoire des environnements et le
rollout partiel ne sont pas restaurés : la reprise ne reproduit pas un run ininterrompu bit
à bit. Les checkpoints périodiques sont pris pendant les rollouts ; le modèle final est
sauvegardé après la dernière mise à jour PPO. Les anciens logs et erreurs restent conservés.

## Export et contrôle

```bash
uv run python scripts/export_agent.py --run-dir runs/baseline --output-dir submission/driver1
uv run python vendor/simulation/verify_submission.py --submission submission/driver1
```

Le dossier contient `agent.py`, `model.onnx`, `model.onnx.data` et `export.json`. L'agent
embarque les sources NumPy canoniques, isole l'historique par voiture, et charge le modèle
relativement à son propre fichier. Il ne dépend ni de PyTorch ni du dépôt. L'export vérifie
l'accord des actions normalisées à `1e-5` sur 256 entrées et conserve les empreintes des
fichiers produits. La tolérance correspond à `6e-5` m/s et `4,189e-6` rad après conversion.
Les essais de course et de prédiction sur des observations réelles complètent ce contrôle.

Le modèle ONNX utilise l'opset 17 et le chemin d'export TorchScript explicite de la version
PyTorch épinglée. Il conserve les poids dans un vrai fichier externe, même pour les petits
MLP. Les fichiers des modèles livrés se versionnent sans Git LFS (`docs/remise.md`).

La CI `Training and ONNX` exécute un entraînement court, une reprise, l'export, la comparaison
sur observations réelles et le vérificateur officiel dans l'image CPU du tournoi.

L’évaluation refuse un modèle si les empreintes de features ou d’historique diffèrent de
celles de son entraînement, même si la taille des entrées reste identique. Le rapport refuse
les courses en double et le mélange de deux modèles pour un même pilote et une même seed :
conserver les tentatives, puis sélectionner explicitement leurs fichiers avant comparaison.
Une erreur de fermeture du logger ou des workers est enregistrée avant la sortie du run.
