# Remise finale sur le dépôt Epitech (E-7)

Rendu : **7 février 2027 à 23:42 (Europe/Paris)**. Le dépôt de remise est [EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8](https://github.com/EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8). Il ne sert qu'à la remise : le travail courant reste sur `origin`.

## Accès

```bash
git remote add epitech git@github.com:EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8.git
git ls-remote epitech                                        # lecture
git push --dry-run epitech origin/main:refs/heads/main       # écriture, rien n'est envoyé
```

L'organisation impose le SSO SAML d'Epitech. Un token GitHub (API, MCP) doit être autorisé pour l'organisation avant de pouvoir lire le dépôt. L'accès SSH a fonctionné directement.

État vérifié le 06/10/2026 :

| Membre | Lecture | Écriture |
| --- | --- | --- |
| Lucas | OK | OK (dry-run) |
| Léo | à vérifier | à vérifier |
| Thibaud | à vérifier | à vérifier |

`epitech/main` contient les deux commits initiaux de Léo du 16/09 (`bb52993`, `3e184fe`). Ce sont les ancêtres de notre `main` : la remise est un simple fast-forward qui conserve l'historique et les auteurs.

## Pas de Git LFS

Les modèles livrés (`submission/driver1..4/model.onnx`) sont versionnés directement dans Git. Ce sont de petits réseaux, de quelques Mo chacun, très loin de l'avertissement de GitHub (50 Mo) et de sa limite (100 Mo par fichier). LFS aurait ajouté un risque inutile : une récupération du rendu qui ne suit pas LFS ne trouverait que des pointeurs texte à la place des modèles. Si un modèle approche 50 Mo, on regarde d'abord la quantification (E-41) avant de revenir sur ce choix.

## Procédure le jour de la remise

1. Promouvoir `develop` vers `main` par PR, puis tagger la version remise.
2. Depuis un clone propre de `main`, vérifier la livraison avec le vérificateur officiel :

   ```bash
   python vendor/simulation/verify_submission.py --submission submission/ --grandprix
   ```

3. Vérifier que le push est un fast-forward, puis pousser `main` et le tag, sans `--force` ni `--mirror` :

   ```bash
   git fetch origin epitech
   git merge-base --is-ancestor epitech/main origin/main && echo fast-forward
   git push epitech origin/main:refs/heads/main
   git push epitech <tag>
   ```

4. Recloner le dépôt Epitech dans un dossier temporaire, puis relancer le vérificateur sur ce clone : c'est ce que la pédagogie récupérera.
