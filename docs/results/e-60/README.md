# E-60 · Généralisation de driver1 et driver2 sur les circuits modifiés

## Protocole (figé avant les résultats, 2026-11-25)

**Question.** Nos pilotes finissent-ils un circuit qu'ils n'ont jamais vu, ou ont-ils appris leurs circuits par cœur ? Leurs taux d'arrivée viennent des circuits qui ont servi à les choisir : ils ne disent rien d'un circuit nouveau ou modifié, ce que le tournoi testera sans doute.

**Pilotes.** Le témoin scripté, `submission/driver1` et `submission/driver2`, tels que livrés : l'agent ONNX qui court au tournoi, post-traitements compris.

**Courses.** `evaluate_drivers.py --condition nominal --variants`, seeds 0, 1 et 2, soit 135 courses par pilote :

| Partition | Circuits | Courses |
|---|---|---:|
| `train` | les 15 circuits d'entraînement | 45 |
| `validation` | les 4 circuits de validation | 12 |
| `variant` | `tracks.variant_test_tracks()` : Norisring, et Montreal, Norisring, Sochi, Spielberg, YasMarina en `reverse`, `mirror`, `narrow`, `bend1`, `section1` | 78 |

Aucun pilote n'a roulé sur une variante avant cette campagne, et leurs bases n'ont jamais servi à l'entraînement.

**Indicateurs.**
1. Le taux d'arrivée par partition, puis par transformation sur `variant` (15 courses par transformation, 3 sur Norisring).
2. Le chrono rapporté à celui du témoin scripté sur la même course, quand les deux finissent.

**Lecture, fixée d'avance.** L'écart de généralisation d'un pilote est son taux d'arrivée en `validation` moins son taux en `variant`. Une transformation qui fait perdre plus de 10 points d'arrivée par rapport à la validation est une faiblesse à corriger à l'entraînement (randomisation des circuits, `--track-variants`). Avec 15 courses par transformation, une différence de 1 ou 2 courses reste du bruit : on lit les tendances, pas les décimales.

## Résultats (2026-11-25)

Taux d'arrivée, et chrono médian rapporté au témoin scripté sur les mêmes courses :

| Partition | Scripté | driver1 | driver2 |
|---|---|---|---|
| `train` (45) | 100 % | 93,3 % · ×1,29 | 100 % · ×0,72 |
| `validation` (12) | 100 % | 100 % · ×1,25 | 33,3 % · ×0,72 |
| `variant` (78) | 100 % | 93,6 % · ×1,25 | 52,6 % · ×0,72 |
| ↳ `reverse` (15) | 100 % | **86,7 %** | 60,0 % |
| ↳ `mirror` (15) | 100 % | 100 % | 60,0 % |
| ↳ `narrow` (15) | 100 % | **80,0 %** | 46,7 % |
| ↳ `bend1` (15) | 100 % | 100 % | 46,7 % |
| ↳ `section1` (15) | 100 % | 100 % | 40,0 % |
| ↳ Norisring (3) | 100 % | 100 % | 100 % |

Le témoin finit les 135 courses : chaque variante du banc est praticable.

**driver1 généralise bien, avec deux faiblesses.** Il finit tous les miroirs, virages et tronçons modifiés, et Norisring. Selon la règle fixée d'avance, il perd plus de 10 points sur deux transformations :
- `narrow` : ses 3 abandons sont tous sur `Montreal@narrow`, le couloir le plus étroit du banc (1,08 m), vers 90 % du premier tour, après plus de 100 pas de contact avec les murs ;
- `reverse` : 2 abandons sur 3 sur `YasMarina@reverse`, au même endroit (73 % du premier tour).

Il reste 25 % plus lent que le témoin.

**driver2 ne perd rien sur les variantes, parce qu'il échoue déjà sur les circuits de base.** Ses abandons suivent le circuit d'origine, pas la transformation : 0 arrivée sur 18 courses pour Montreal et YasMarina, officiels ou modifiés, 18 sur 18 pour Spielberg et Norisring, 9 sur 18 pour Sochi. Il finit 100 % de ses circuits d'entraînement et 33 % de la validation : c'est le sur-apprentissage noté dans E-57. Quand il finit, il est 28 % plus rapide que le témoin.

**Anomalie hors sujet.** driver1 abandonne Hockenheim (entraînement) sur les 3 seeds, au même endroit du troisième tour (progression 2,46, pas 3 866 à 3 871), dont une fois sans aucun contact avec un mur. À examiner à part.

**Suite.** driver1 : entraîner avec `--track-variants` et mesurer à nouveau `narrow` et `reverse` sur ce banc. driver2 : E-57 (recettes de driver1, puis randomisation des circuits).

Courses brutes : `scripted.json`, `driver1.json`, `driver2.json` (135 courses chacun), à publier comme artefact W&B `results-e-60` (`scripts/results_artifact.py publish docs/results/e-60`).
