# E-47 : résultats bruts des exploits du simulateur

Analyse et recommandations : [`docs/exploits.md`](../../exploits.md).

| Fichier | Contenu |
|---|---|
| [`shortcuts.json`](shortcuts.json) | Sites de raccourci des 23 circuits aux marges de 1,00 m et 0,50 m, avec les 52 traversées de chaque site (4 vitesses × 13 angles) et leur résumé |
| [`contacts.json`](contacts.json) | 180 mesures de contact sur Monza, Silverstone et Catalunya : poussée par l'arrière (2 à 4 voitures), poussée latérale, voiture arrêtée poussée contre un mur, choc contre un mur |
| [`../../../src/crashlearn/exploit_scenarios.json`](../../../src/crashlearn/exploit_scenarios.json) | Scénarios reproductibles tirés de ces mesures, avec le retour v1 de chaque raccourci |

Chaque JSON porte son commit, l'empreinte du simulateur et celle de `exploits.py`, la commande et la date. Les mesures sont déterministes : la graine ne règle que le bruit LiDAR, que ces pilotes ne lisent pas.

```bash
uv run python scripts/map_exploits.py shortcuts --output docs/results/e-47/shortcuts.json
uv run python scripts/map_exploits.py contacts --output docs/results/e-47/contacts.json
uv run python scripts/map_exploits.py scenarios --shortcuts docs/results/e-47/shortcuts.json
```

Durée sur la station (4 processus) : environ 5 min pour les raccourcis, 1 min pour les contacts, 1 min pour les scénarios.
