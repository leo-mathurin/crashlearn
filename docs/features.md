# Features d'observation (E-16)

`src/crashlearn/features.py` est la source unique qui transforme une observation brute de `get_obs` en vecteur d'entrée du réseau. `CrashLearnEnv` l'utilise à l'entraînement. L'export (E-25) la recopiera dans chaque `agent.py`. Le module n'importe que `math` et NumPy, et un test le vérifie : dans la soumission, seuls NumPy et ONNX Runtime sont autorisés.

## Un seul schéma d'adversaires

Depuis l'archive du 18/09, le loader (`create_dummy_obs`), `get_space_info` et `get_obs` décrivent les adversaires avec les mêmes clés : `x_rel, y_rel, yaw_rel, speed, progress, lap_count, status`. On ne lit plus que ce schéma, et l'ancien schéma du loader (`rel_x`, `active`…) n'est plus maintenu. Un test casse si une nouvelle archive change ces clés.

## Vecteur d'entrée : 126 flottants dans [-1, 1]

| Indices | Contenu | Normalisation |
|---|---|---|
| 0-99 | 100 rayons LiDAR | / 15 m |
| 100 | vitesse (signée) | / 10 m/s |
| 101 | braquage | / 0,4189 rad |
| 102-103 | sinus et cosinus de la phase du tour | sans |
| 104 | tours bouclés | / 3 |
| 105-125 | 3 emplacements adversaires × 7 valeurs | voir ci-dessous |

Chaque emplacement adversaire contient :

| Valeur | Normalisation |
|---|---|
| masque : 1 si l'emplacement est occupé, sinon 0 | sans |
| `x_rel`, `y_rel` dans le repère de la voiture (x vers l'avant) | / 15 m, saturé |
| sinus et cosinus de `yaw_rel` | sans |
| vitesse | / 10 m/s |
| écart de course en tours : (tours + phase) de l'adversaire moins les nôtres | saturé à ±1 |

## Choix d'encodage

- **Même entrée en solo et pour le champion.** Les emplacements adversaires sont toujours présents, à zéro et masqués en solo. Une politique solo se transfère au champion sans changer d'architecture ni d'espace d'observation (ADR 0001). Les 21 entrées nulles ne coûtent presque rien à un MLP.
- **Seuls les adversaires en course sont lus.** Un emplacement n'est retenu que si `status == 1`. On écarte l'emplacement de la voiture elle-même, les voitures absentes, en DNF ou arrivées, quelles que soient leurs valeurs.
- **Tri du plus proche au plus loin.** Le réseau voit d'abord l'adversaire le plus menaçant. Le résultat ne dépend pas de l'ordre des emplacements dans le dictionnaire, ce qui évite que la politique apprenne un numéro de voiture. Les emplacements vides sont à la fin.
- **Lacet en sinus et cosinus**, comme la phase du tour, pour qu'il n'y ait pas de saut à ±π.
- **Position relative saturée à 15 m**, la portée du LiDAR. Au-delà, l'adversaire est vu « loin » dans sa direction.
- **Écart de course.** Chaque voiture compte sa progression depuis sa propre position de départ. Sur une grille échelonnée, l'écart est donc faussé de la distance entre les places, moins d'un mètre (mesuré à 0 au départ sur Budapest, IMS et Austin). Pour savoir qui est devant de près, il faut regarder `x_rel`. L'écart sert à savoir qu'une voiture a un tour d'avance ou de retard.

## Lecture tolérante

Dans le tournoi, une exception dans `agent.py` entraîne un DNF. Une donnée manquante ou invalide prend donc une valeur neutre au lieu de lever une erreur :

- clé absente, `None`, texte ou valeur non finie : 0 ;
- rayon LiDAR manquant ou non fini : 15 m, donc espace libre ;
- `opponents` absent ou mal formé : aucun adversaire ;
- emplacement actif sans données : adversaire présent, avec des valeurs nulles.

Les tests (`tests/test_features.py`) couvrent le cas nominal, les emplacements vides et inactifs, l'ordre des emplacements et les données incomplètes ou invalides. Un dernier test comparera les features de chaque `agent.py` livré à celles de l'entraînement dès que `submission/` existera.
