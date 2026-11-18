# Remise finale sur le dépôt Epitech (E-7)

Rendu : **7 février 2027 à 23:42 (Europe/Paris)**. Le dépôt de remise est [EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8](https://github.com/EpitechMscProPromo2027/T-AIA-901-LYN-9-1-crashlearn-8). Le travail courant reste sur `origin` ; le dépôt Epitech en est une copie automatique.

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

La première synchronisation de `main` avance les deux commits initiaux de Léo du 16/09 (`bb52993`, `3e184fe`) jusqu'au `main` du dépôt principal. Elle conserve l'historique et les auteurs.

## Synchronisation automatique

Le [workflow Epitech mirror](../.github/workflows/mirror-epitech.yml) copie toutes les branches et tous les tags au push sur une branche qui contient le workflow et à la suppression d'une branche dans `origin`. Une exécution toutes les 15 minutes rattrape les pushes sur les anciennes branches sans workflow et les tags qui pointent vers un ancien commit. Pour une copie immédiate, lancer manuellement **Epitech mirror** depuis `main` dans GitHub Actions. La copie est atomique et supprime les branches et tags présents uniquement sur Epitech : ne pas modifier directement le dépôt Epitech.

Le secret GitHub Actions `EPITECH_MIRROR_TOKEN` contient un token classique avec les scopes `repo` et `workflow`, autorisé par le SSO de l'organisation Epitech. Il expire le 16 février 2027 ; renouveler le token et le secret avant cette date si le projet continue, puis lancer manuellement le workflow pour vérifier le nouvel accès. Le test SSH ci-dessus vérifie uniquement l'accès SSH personnel.

## Pas de Git LFS

Les modèles livrés (`submission/driver1..4/model.onnx`) sont versionnés directement dans Git. Ce sont de petits réseaux, de quelques Mo chacun, très loin de l'avertissement de GitHub (50 Mo) et de sa limite (100 Mo par fichier). LFS aurait ajouté un risque inutile : une récupération du rendu qui ne suit pas LFS ne trouverait que des pointeurs texte à la place des modèles. Si un modèle approche 50 Mo, on regarde d'abord la quantification (E-41) avant de revenir sur ce choix.

## Procédure le jour de la remise

1. Promouvoir `develop` vers `main` par PR, puis tagger la version remise.
2. Depuis un clone propre de `main`, vérifier la livraison avec le vérificateur officiel :

   ```bash
   uv run python vendor/simulation/verify_submission.py --submission submission/ --grandprix
   ```

3. Vérifier que le workflow de synchronisation du dernier push est vert, puis comparer `main` et le tag entre les deux dépôts :

   ```bash
   set -e
   git fetch --multiple origin epitech
   test "$(git rev-parse origin/main)" = "$(git rev-parse epitech/main)"
   origin_tag=$(git ls-remote --refs --exit-code origin "refs/tags/<tag>")
   epitech_tag=$(git ls-remote --refs --exit-code epitech "refs/tags/<tag>")
   test "$origin_tag" = "$epitech_tag"
   ```

4. Recloner le dépôt Epitech dans un dossier temporaire, puis relancer le vérificateur sur ce clone : c'est ce que la pédagogie récupérera.
