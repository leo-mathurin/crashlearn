# E-47 : mesures des exploits du simulateur

Analyse et recommandations : [`docs/exploits.md`](../../exploits.md).

| Fichier | Contenu |
|---|---|
| `shortcuts.json` | Sites de raccourci des 23 circuits aux marges de 1,00 m et 0,50 m, avec les 52 traversées de chaque site (4 vitesses × 13 angles) et leur résumé |
| `contacts.json` | 180 mesures de contact sur Monza, Silverstone et Catalunya : poussée par l'arrière (2 à 4 voitures), poussée latérale, voiture arrêtée poussée contre un mur, choc contre un mur |
| [`src/crashlearn/exploit_scenarios.json`](../../../src/crashlearn/exploit_scenarios.json) | 13 scénarios tirés de ces mesures, avec le retour v1 de chaque raccourci |

Les deux mesures sont dans l'artefact W&B `results-e-47` : `SHA256SUMS` donne sa version et l'empreinte de chaque fichier ([récupération](../../experiments.md#résultats-bruts)). Les scénarios restent dans Git, car `exploits.py` les lit.

Chaque mesure porte son commit (`4a59688`, arbre propre), l'empreinte du simulateur et celle de `exploits.py`, la commande et la date. Elle a été écrite sous `runs/e-47-clean/`, puis copiée ici. Les mesures sont déterministes : la graine ne règle que le bruit LiDAR, que ces pilotes ne lisent pas.

## Reproduire

Durées sur la station, 4 processus :

```bash
uv run python scripts/map_exploits.py shortcuts --output docs/results/e-47/shortcuts.json    # 5 min
uv run python scripts/map_exploits.py contacts --output docs/results/e-47/contacts.json      # 1 min
uv run python scripts/map_exploits.py scenarios --shortcuts docs/results/e-47/shortcuts.json # 1 min
```

`shortcuts` et `contacts` refusent d'écraser un fichier existant. `scenarios` réécrit `src/crashlearn/exploit_scenarios.json`. Pour le relancer sur les mesures publiées, récupérer d'abord l'artefact : `uv run python scripts/results_artifact.py fetch docs/results/e-47`.
