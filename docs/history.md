# Mémoire temporelle par voiture (E-13)

Les pilotes B et C décident à partir des quatre dernières observations. Le module [`src/crashlearn/history.py`](../src/crashlearn/history.py) garde cet historique **séparément pour chaque voiture**. Comme `features.py`, il n'importe que NumPy et la bibliothèque standard : l'export (E-25) le recopie dans chaque `agent.py` qui conduit avec un historique.

```python
from crashlearn.features import build_features
from crashlearn.history import CarHistory

history = CarHistory()  # DEPTH = 4


def predict(obs, info):
    stack = history.push(obs, info, build_features(obs))  # 4 x 126 float32, le plus ancien d'abord
    ...
```

## Pourquoi par voiture

Rien n'oblige le tournoi à créer une instance d'`Agent` par voiture. `sim_recorder.py` prévoit de recopier une soumission unique sur toutes les voitures (self-play). Une mémoire gardée sur l'instance mélangerait alors les observations des quatre voitures. L'historique est donc indexé par `obs["agent_id"]`.

Avec le loader de l'archive du 18/09, ce chemin self-play n'est jamais atteint : une soumission plate est refusée dès qu'il y a plus d'une voiture, et chaque dossier `driverN/` reçoit sa propre instance. On garde l'indexation par voiture parce que le harness du tournoi n'est pas connu. `test_no_memory_mixing_in_the_recorder_sharing_one_instance` remplace donc seulement la découverte des soumissions, pour faire courir **une** instance sur quatre voitures dans la vraie boucle du recorder, dry-run compris.

## Règles

- **Historique initial** : au premier appel d'un épisode, l'observation est recopiée dans les quatre emplacements. La taille de l'entrée est fixe dès la première décision. Il n'y a pas de zéros, qu'un réseau pourrait prendre pour une vraie observation.
- **Nouvel épisode**, détecté par voiture :
  - le compteur `info["step_count"]` n'est pas exactement le pas suivant du dernier appel de cette voiture. Ça couvre le recul du compteur, deux épisodes courts qui se suivent (même compteur) et un pas sauté, après lequel la mémoire ne décrit plus le présent ;
  - sans `step_count`, `time_elapsed × 20 Hz` le remplace ;
  - sans horloge, une baisse de `obs["lap_count"]`.
- **Dry-run du loader** : avant la course, `agent_loader` appelle `predict` sur une observation factice (`agent_id = 0`, 100 rayons à exactement 5 m, vitesse nulle, `step_count = 0`). Le premier appel de course arrive au pas 1, donc le compteur seul ne verrait pas de changement d'épisode. Cette observation est reconnue et ne laisse aucune trace. Un vrai scan ne donne jamais 100 rayons égaux à 5 m.
- **`agent_id` absent ou invalide** : clé `MISSING_ID = -1`, séparée de la vraie voiture 0.
- **Lecture tolérante** : une clé manquante, `None`, une chaîne ou une valeur non finie ne lève jamais d'exception. Au tournoi, une exception vaut un DNF.
- La pile renvoyée est une copie : la modifier ne touche pas la mémoire.

## Limites de la détection de reset

- **Sans `step_count` ni `time_elapsed`** (par exemple `info = {}`), seule une baisse du nombre de tours révèle un nouvel épisode. Deux épisodes qui commencent tous deux au tour 0 se suivent alors dans le même historique. Les trois premières décisions du second gardent des observations du premier.
- **Plusieurs voitures sans `agent_id`** partagent la clé `MISSING_ID` et leurs mémoires se mélangent. Aucune autre clé de l'observation n'identifie la voiture de façon sûre.
- **Changement de circuit** : il n'est vu qu'à travers le compteur. Un harness qui changerait de carte sans remettre le compteur à zéro ne serait pas détecté.
- **Dry-run** : la reconnaissance suit la forme exacte de `create_dummy_obs`. Si une archive change cette observation factice, `test_constants_match_the_simulator` casse.
- Ces règles supposent un appel par voiture et par pas, comme dans `sim_recorder.py` et `demo.py`. Une voiture arrivée n'est plus appelée, ce qui n'a pas d'effet puisque sa course est finie.

## Pilote de repli

Le pilote réactif (A) ne garde aucun état : il ne dépend d'aucune de ces règles. Il reste le repli si l'historique pose problème au tournoi.

## Entraînement

L'entraînement des pilotes B et C (E-18, E-24) doit empiler les observations avec `CarHistory`, pas avec `VecFrameStack` de SB3. `VecFrameStack` remplit le début d'épisode avec des zéros, alors que `agent.py` recopie la première observation : le réseau verrait au tournoi des débuts d'épisode qu'il n'a jamais rencontrés.
