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
checkpoint doit être dans ce dossier et correspondre au hash que le manifest a enregistré
pour lui (`checkpoints` pour les checkpoints périodiques, `model_sha256` pour le modèle
final). Les checkpoints sans hash enregistré, par exemple ceux des runs antérieurs à ce
contrôle, ne peuvent pas reprendre. Une modification du prétraitement, de la reward ou du
simulateur est refusée : démarrer un nouveau run pour mesurer cette modification. Un budget
`--timesteps` inférieur ou égal à l'étape du checkpoint est refusé avant toute tentative,
sans modifier le manifest.

```bash
uv run python scripts/train_ppo.py --history-depth 1 --seed 0 --timesteps 2000000 \
  --device cuda --output-dir runs/baseline \
  --resume runs/baseline/checkpoints/ppo_attempt001_1000000_steps.zip --wandb
```

Le modèle recharge les poids et l'optimiseur. L'état aléatoire des environnements et le
rollout partiel ne sont pas restaurés : la reprise ne reproduit pas un run ininterrompu bit
à bit. Les checkpoints périodiques sont pris pendant les rollouts ; le modèle final est
sauvegardé après la dernière mise à jour PPO. Les anciens logs et erreurs restent conservés.

## Adapter un modèle dans un nouveau run

`--initialize-from <model.zip>` copie seulement les poids compatibles : nouvel
optimiseur, nouvelle seed de run et compteur remis à zéro. `--max-speed` borne
la commande de vitesse ; `--random-start` diversifie les départs solo sur les
circuits train. Une reprise doit conserver ces deux réglages.

```bash
uv run python scripts/train_ppo.py --history-depth 1 --seed 0 --timesteps 2000000 \
  --device cuda --n-envs 15 --fixed-map-workers --max-speed 5 --random-start \
  --initialize-from runs/baseline/model.zip --output-dir runs/adaptation
uv run python scripts/ablate_progress.py \
  --run-dir runs/adaptation --output-dir runs/adaptation-without-progress
```

L'ablation produit un modèle d'inférence réactif sans connexions depuis la
progression et les tours. Son manifest distinct conserve la source et la
transformation. Une dérivation ne peut pas reprendre son entraînement ; utiliser
ses poids avec `--initialize-from` pour un nouveau run. Les bornes d'action
enregistrées dans le modèle sont conservées par l'export ONNX.

Ajouter `--max-speed 4.75` à l'ablation pour abaisser aussi la borne de vitesse
d'inférence. Cette option conserve les poids appris et la configuration du run
d'entraînement ; le manifest dérivé enregistre les deux bornes. Le plafond doit
être positif et au plus égal à celui du modèle source.

`--weather` change la friction et le bruit LiDAR en pleine course, et `--nominal-share`
garde une part d'épisodes aux valeurs officielles ([météo](weather.md)). Une reprise doit
garder ces réglages. `--friction-feature` ajoute `friction_current` en entrée, pour une
mesure seulement : un tel modèle ne s'exporte pas. Commande complète dans
[les résultats E-23](results/e-23/README.md#provenance).

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

`--friction-exponent k` (E-49) borne en plus la vitesse de l'agent à la limite du modèle
fois `friction_current ** k`, lu dans `info` ; `export.json` enregistre la borne. driver1
utilise k = 0,5 ([résultats](results/e-49/README.md)).

Le modèle ONNX utilise l'opset 17 et le chemin d'export TorchScript explicite de la version
PyTorch épinglée. Il conserve les poids dans un vrai fichier externe, même pour les petits
MLP. Les fichiers des modèles livrés se versionnent sans Git LFS (`docs/remise.md`).

## Évaluer et confirmer

```bash
uv run python scripts/evaluate_drivers.py --run-dir runs/baseline --output runs/baseline.json
uv run python scripts/evaluate_drivers.py --scripted --output runs/scripted.json
uv run python scripts/evaluate_drivers.py --submission submission/driver1 --output runs/d1.json
```

`--submission` fait courir un dossier livré, chargé par le loader officiel : c'est l'agent
ONNX qui court, borne de friction comprise. Sous météo, ses trajectoires divergent de
celles du modèle PyTorch ; confirmer un livrable avec lui.

`evaluate_drivers.py` fait courir un run, ou le témoin scripté, sur les 117 courses de la
campagne : nominal sur train et validation, perturbations sur validation, jamais le test.
Les DNF et TIMEOUT sont des résultats. Le fichier est réécrit après chaque course. Les
perturbations suivent le sujet v1.3.0 depuis E-23 ([conditions](weather.md#évaluation)) ;
les anciens fichiers de 105 courses gardent leurs valeurs, enregistrées course par course.

`confirm_submission.py` fait courir un dossier de soumission sur les quatre circuits de
validation, dans l'image CPU du tournoi, sans Gymnasium ni PyTorch. Une seed de course y
donne la même seed moteur que dans `CrashLearnEnv`. Exemple dans
[les résultats de driver1](results/driver1/README.md).

L'évaluation, l'export et l'ablation refusent un run inachevé, un `model.zip` modifié, ou
des empreintes de features ou d'historique différentes de celles de l'entraînement, même
si la taille des entrées reste identique. Une erreur de fermeture du logger ou des workers
est enregistrée dans le manifest avant la sortie du run.

La CI `Training and ONNX` exécute un entraînement court, une reprise, l'export, la
comparaison sur observations réelles et le vérificateur officiel dans l'image CPU du
tournoi. Elle installe la pile d'entraînement aux versions de `uv.lock`.

## Résultats bruts

Les courses brutes (`.json` de `docs/results/`) ne sont pas dans Git : chaque campagne
est un artefact W&B `results-<dossier>` du projet `crashlearn/crashlearn`. Git garde le
README, les manifests et `SHA256SUMS`, qui donne la version de l'artefact et l'empreinte
de chaque fichier. Le `.gitignore` refuse les autres `.json` de `docs/results/`.

```bash
uv run python scripts/results_artifact.py fetch docs/results/e-18     # télécharge et vérifie
uv run python scripts/results_artifact.py publish docs/results/e-23   # nouvelle version
```

`publish` envoie tous les `.json` du dossier hors `manifests/` et réécrit `SHA256SUMS` ;
committer ce fichier avec le README qui cite les résultats. `fetch` échoue si un fichier
diffère de `SHA256SUMS`. L'accès demande un compte de l'équipe W&B (section W&B).
