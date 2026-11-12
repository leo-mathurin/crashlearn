# Remise finale sur le dépôt Epitech (E-7)

Rendu : **7 février 2027 à 23:42 (Europe/Paris)**. Le dépôt de remise est [EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8](https://github.com/EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8). Il ne sert qu'à la remise : le travail courant reste sur `origin`.

## Accès

```bash
git remote add epitech git@github.com:EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8.git
git ls-remote epitech                                        # lecture
git push --dry-run epitech origin/main:refs/heads/main       # écriture, rien n'est envoyé
```

L'organisation impose le SSO SAML d'Epitech. Un token GitHub (API, MCP) doit être autorisé pour l'organisation avant de pouvoir lire le dépôt. L'accès SSH a fonctionné directement.

État vérifié le 06/10/2026 pour Lucas, confirmé le 23/10/2026 pour Léo et Thibaud :

| Membre | Lecture | Écriture |
| --- | --- | --- |
| Lucas | OK | OK (dry-run) |
| Léo | OK | OK |
| Thibaud | OK | OK |

`epitech/main` contient les deux commits initiaux de Léo du 16/09 (`bb52993`, `3e184fe`). Ce sont les ancêtres de notre `main` : la remise est un simple fast-forward qui conserve l'historique et les auteurs.

## Pas de Git LFS

Les modèles livrés (`submission/driver1..4/model.onnx`) sont versionnés directement dans Git. Ce sont de petits réseaux, de moins de 1 Mo chacun, très loin de l'avertissement de GitHub (50 Mo) et de sa limite (100 Mo par fichier). LFS aurait ajouté un risque inutile : une récupération du rendu qui ne suit pas LFS ne trouverait que des pointeurs texte à la place des modèles. Si un modèle approche 50 Mo, on regarde d'abord la quantification (E-41) avant de revenir sur ce choix.

Mesuré le 12/11/2026 (E-36) : `model.onnx.data` pèse 395 296 octets pour driver1 et 782 352 octets pour driver2, `model.onnx` moins de 3 Ko. Le plus gros fichier de l'historique est une carte du simulateur de 4,2 Mo. Aucun fichier n'a d'attribut LFS (pas de `.gitattributes`). Checkpoints et runs restent hors Git : `runs/` (ignoré), archives de la station sous `/home/leo/runs/crashlearn/` et W&B.

## Provenance et restauration des modèles (E-36)

`scripts/check_deliverables.py` vérifie chaque `submission/driver*/` sans dépendance à installer ni accès à W&B. Il lit les hashes dans `export.json` et dans les manifests, il n'en fixe aucun. Il échoue si :

- un fichier livré manque, est un pointeur LFS ou diffère de son hash dans `export.json` ;
- l'export n'est pas propre (`dirty` différent de `false`) ;
- le modèle source, ou un modèle dont il dérive ou part, n'a pas de manifest sous `docs/results/*/manifests/` avec commit, configuration et seed ;
- aucun `docs/results/<campagne>/README.md` ne cite le modèle source (hash complet ou ses 12 premiers caractères), à côté d'un `SHA256SUMS` qui nomme la version de l'artefact W&B de ses résultats ([résultats bruts](experiments.md#résultats-bruts)).

Un nouvel export versionne donc son manifest, et le README et le `SHA256SUMS` de sa campagne. Le job Pytest de la CI lance ce contrôle. Pour vérifier aussi les courses, `scripts/results_artifact.py fetch` télécharge l'artefact et le compare à `SHA256SUMS`.

```bash
python scripts/check_deliverables.py          # --git : chaque commit doit exister dans le clone
```

Vérifié le 12/11/2026 :

| Contrôle | Résultat |
| --- | --- |
| Clone neuf depuis GitHub (`gh repo clone`, branche E-36), sans git-lfs installé | Contrôle OK avec `--git`, aucun pointeur LFS |
| driver1 : `cap5-random-h1-seed0` restauré depuis `e19-20261110`, puis `ablate_progress.py --max-speed 4.75` et `export_agent.py` | Modèle source conforme au manifest ; 13 tenseurs identiques au modèle livré ; `model.onnx` et `model.onnx.data` identiques octet par octet |
| driver2 : `ppo-h4-seed1` restauré depuis `e18-20261109`, puis `export_agent.py` au commit d'export `a900b2a` | Modèle source conforme au manifest ; `model.onnx` et `model.onnx.data` identiques octet par octet |
| `agent.py` des deux pilotes | Même code (AST) ; seul le formatage Ruff diffère |
| Dépôt Epitech, en lecture et `push --dry-run` de la branche vers `main` | Lecture OK ; fast-forward depuis `3e184fe` accepté |

Le `model.zip` redérivé n'a pas le même hash que l'original : l'archive zip change, pas les poids. Ré-exporter driver2 avec l'exportateur actuel donne d'autres octets, car l'export borne les actions depuis `3a7da7e` : il faut ré-exporter au commit noté dans `export.json`.

Les commits de provenance (`2e5a751`, `a900b2a`, `9b3a9cc`, `adf5a5f`) ne sont que sur la branche de PR #19. Après un squash merge, un clone de `develop` ou du dépôt Epitech ne les contient plus : `--git` échoue alors. Ils restent sur `origin` par `refs/pull/19/head`. Un tag par modèle livré les garderait visibles dans un clone normal.

## Procédure le jour de la remise

1. Promouvoir `develop` vers `main` par PR, puis tagger la version remise.
2. Depuis un clone propre de `main`, vérifier la livraison avec le vérificateur officiel :

   ```bash
   uv run python vendor/simulation/verify_submission.py --submission submission/ --grandprix
   ```

3. Vérifier que le push est un fast-forward, puis pousser `main` et le tag, sans `--force` ni `--mirror` :

   ```bash
   git fetch --multiple origin epitech
   git merge-base --is-ancestor epitech/main origin/main && echo fast-forward
   git push epitech origin/main:refs/heads/main
   git push epitech <tag>
   ```

4. Recloner le dépôt Epitech dans un dossier temporaire, puis relancer le vérificateur sur ce clone : c'est ce que la pédagogie récupérera.
