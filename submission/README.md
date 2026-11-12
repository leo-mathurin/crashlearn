# Pilotes solo ONNX

`driver1` est le PPO réactif ; `driver2` est le PPO avec quatre observations.
Chaque dossier contient un agent autonome, son modèle opset 17 et de vrais poids externes.
Les modèles et les prétraitements sont identifiés dans `export.json`. Résultats et
provenance : [driver1](../docs/results/driver1/README.md),
[driver2 et la campagne E-18](../docs/results/e-18/README.md).

Les fichiers générés `agent.py` ont été formatés avec Ruff après export, puis leur hash
a été mis à jour dans `export.json`. Ce formatage ne change pas le prétraitement.

Ces deux pilotes constituent la livraison solo actuelle. Le pilote SAC et le champion
multi-agent relèvent des étapes E-24 et E-26/E-31.

**Comportement avec des adversaires non mesuré.** L'entraînement et toutes les
validations se font sans adversaire : les colonnes 105 à 125 des features (adversaires)
restent nulles, et leurs poids d'entrée gardent leur initialisation aléatoire. Avec des
adversaires actifs, ces colonnes ajoutent un décalage non appris. Ces poids seront mis à
zéro avant la course commune des pilotes solo (E-28) : chaque pilote conduira alors
exactement comme seul en piste.

```bash
uv run python vendor/simulation/verify_submission.py --submission submission/driver1
uv run python vendor/simulation/verify_submission.py --submission submission/driver2
```

## Contrôle en processus partagé (E-27)

Au tournoi, tous les `agent.py` sont importés dans un seul processus Python, depuis le cwd
du harnais. `scripts/check_shared_process.py` reproduit cette situation : il charge
`driver1` à `driver4` avec le loader officiel, dry-run compris, depuis un cwd vide. Pendant
le chargement, seuls NumPy, ONNX Runtime et la bibliothèque standard sont importables. Il
lance ensuite trois courses de 600 décisions à quatre voitures sur Austin et vérifie :

| Élément | Contrôle |
| --- | --- |
| `model.onnx.data` | présent, conforme au hash de `export.json`, seule source des poids externes |
| observations et `info` | même schéma qu'au dry-run, valeurs dans les bornes de `get_space_info()` |
| actions | floats Python finis, vitesse (m/s) et braquage (rad) dans les bornes |
| mémoire | chaque instance rejouée seule refait ses actions ; une instance pour quatre voitures garde une mémoire par voiture ; un second chargement sous le même nom de module ne change rien |

Tant que `driver3` et `driver4` n'existent pas, ce sont des copies de `driver1` et
`driver2`, et la sortie le dit. Ces courses testent le chargement et le partage du
processus, pas la conduite avec adversaires. La CI (job « Training and ONNX ») lance ce
contrôle dans l'image CPU du tournoi, avec `verify_submission.py --submission` sur chaque
pilote et `--grandprix` sur les quatre dossiers contrôlés.

```bash
docker run --rm -v "$PWD:/workspace:ro" -w /tmp -e NUMBA_CACHE_DIR=/tmp/numba \
  crashlearn-sim:20260918 python /workspace/scripts/check_shared_process.py \
  --submission /workspace/submission
```
