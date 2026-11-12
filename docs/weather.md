# Météo changeante (E-23)

## Conditions du tournoi

Le sujet v1.3.0 (p. 3 et 9) fixe deux plages : friction dans [0,7 ; 1,0] (0,7 = forte
pluie) et demi-amplitude du bruit LiDAR relatif dans [0,001 ; 0,1], tiré à chaque pas.
Les deux changent **pendant la course**, ensemble ou séparément, à des moments et pour des
durées inconnus. `info["friction_current"]` donne la friction ; le niveau de bruit n'est
pas observable. Le tournoi utilise sa propre copie du simulateur.

La copie officielle du 18/09 fige les deux valeurs : `_update_friction` construit les
paramètres sans jamais les appliquer, et le bruit reste à 0,001 (`TODO` en amont).
`vendor/` ne change pas : `CrashLearnEnv` applique la météo de notre côté.

## Calendrier

`crashlearn.weather.schedule` tire, une fois par épisode, une suite de segments constants :

- premier segment au pas 0, friction et bruit uniformes dans les plages du tournoi ;
- durée de chaque segment uniforme entre 5 et 40 s de course ;
- à chaque changement, un tiers des cas change la friction seule, un tiers le bruit seul,
  un tiers les deux.

Un segment s'applique avant la décision de son premier pas. La friction passe par
`update_params(mu=…)` sur toutes les voitures, et `_friction` est mis à jour pour que
`friction_current` la rapporte. Le bruit remplace celui du simulateur après chaque pas,
comme les overrides existants.

**Seeds.** Le calendrier d'entraînement vient de la seed moteur de l'épisode, par un flux
séparé du bruit LiDAR (`WEATHER_STREAM`). L'évaluation tire le sien de la seed de course,
par un autre flux (`EVAL_WEATHER_STREAM`) : les courses `changing_weather` sont les mêmes
pour tous les pilotes et ne sont jamais tirées à l'entraînement. Rafales et pertes de
rayons LiDAR ne font pas partie des conditions du tournoi : elles ne sont pas simulées.

**Distributions mesurées** sur 2000 épisodes d'entraînement de 8000 pas
(`--nominal-share 0.25`) :

| Mesure | Valeur |
|---|---|
| Épisodes nominaux (friction 1,0, bruit 0,001) | 23,5 % |
| Segments par épisode non nominal | médiane 18 (13 à 25) |
| Changements friction seule / bruit seul / les deux | 8856 / 8823 / 8886 |
| Friction pondérée par le temps, 250 premières s (déciles 10, 50, 90) | 0,73 ; 0,85 ; 0,97 |
| Part du temps sous 0,8 de friction | 34 % |
| Bruit pondéré par le temps (déciles 10, 50, 90) | 0,01 ; 0,05 ; 0,09 |

Les plages d'entraînement sont celles du tournoi. La campagne n'a pas montré de besoin de
les élargir : les échecs restants se concentrent sur Montreal, quelle que soit la
condition (voir les résultats).

## Évaluation

| Condition | Friction | Bruit LiDAR |
|---|---|---|
| `nominal` | officielle (1,0) | officiel (0,001) |
| `low_friction` | 0,7 | officiel |
| `lidar_noise` | officielle | 0,1 |
| `heavy_rain` | 0,7 | 0,1 |
| `changing_weather` | calendrier par seed de course | calendrier par seed de course |
| `strict_margin` | officielle, marge de sortie de 0,50 m | officiel |

Chaque course `changing_weather` enregistre ses segments (`weather_segments`).

## Friction en entrée

`friction_current` n'entre pas dans les features : la mesure fixée avant les résultats ne
montre pas de gain suffisant ([résultats](results/e-23/README.md)). Les features restent à
126 colonnes, et le transfert vers le champion reste inchangé. `--friction-feature` reste
disponible pour une nouvelle mesure.
