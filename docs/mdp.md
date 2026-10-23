# MDP et reward (E-15)

Ce document décrit le problème tel que l'agent le voit dans `CrashLearnEnv` (`src/crashlearn/env.py`) : observations, actions, reward et fins d'épisode. La reward est versionnée (`RewardConfig.version`). Chaque changement donne une nouvelle version et une entrée dans `docs/memlog.md`.

## Observations

105 flottants dans [-1, 1], pour la voiture apprenante seulement :

| Indices | Contenu | Normalisation |
|---|---|---|
| 0-99 | 100 rayons LiDAR | / 15 m |
| 100 | vitesse (signée) | / 10 m/s |
| 101 | braquage | / 0,4189 rad |
| 102-103 | sinus et cosinus de la phase du tour | sans |
| 104 | tours bouclés | / 3 |

La phase passe par un sinus et un cosinus pour que le passage de 0,999 à 0 ne soit pas un saut. Les adversaires et l'historique des observations relèvent d'E-16 et d'E-13.

## Actions

Deux flottants dans [-1, 1], remappés de façon affine vers une vitesse cible de [-2, 10] m/s et un braquage de [-0,4189, 0,4189] rad. La marche arrière est conservée pour se dégager d'un mur.

## Fins d'épisode

- **terminated** : la voiture a fini ses 3 tours, ou elle est en DNF (moins de 1 m de progression en 10 s, ou sortie du couloir toléré). C'est le simulateur qui décide de ces deux cas.
- **truncated** : `max_episode_steps` décisions (8000 par défaut, soit 400 s) atteintes avant.
- Un contact avec un mur ou une voiture ne termine jamais l'épisode.

## Reward v1

`RaceReward` (`src/crashlearn/reward.py`) calcule chaque terme séparément et les additionne simplement. Les termes de chaque pas sont dans `info["reward_terms"]`, pour pouvoir les journaliser à l'entraînement (E-20).

| Terme | Valeur | Quand |
|---|---|---|
| `progress` | +1 par mètre | progression signée le long de la ligne centrale, `progress_delta × longueur de piste` |
| `lap` | +20 | tour enregistré par le simulateur |
| `finish` | +50 | arrivée (3 tours) |
| `dnf` | -200 | abandon |
| `wall_contact`, `vehicle_contact` | 0 | contact (journalisé seulement) |
| `slow` | -0,1 par pas | voiture en course qui progresse de moins de 1 m/s |
| `reverse` | -0,1 par pas | marche arrière (vitesse < -0,1 m/s) au-delà de 1 s d'affilée |
| `action_change` | -0,02 × ‖Δa‖² | variation de l'action normalisée depuis le pas précédent |

### Pourquoi ces choix

- **La progression en mètres plutôt qu'en fraction de tour.** La v0 donnait `1000 × progress_delta`. Or les circuits d'entraînement mesurent de 293 m (IMS) à 487 m (Sepang) : à mètre égal, la v0 payait 1,7 fois plus sur IMS que sur Sepang, et le même raccourci valait plus sur un petit circuit. En mètres, la même décision de conduite vaut la même chose partout. Un tour vaut alors la longueur de la piste, soit 293 à 487 points.
- **Progression signée.** Reculer sur un tronçon reprend exactement ce que l'avoir parcouru a rapporté. Faire des allers-retours ne rapporte rien.
- **Événements uniques.** `lap` et `finish` reposent sur les compteurs du simulateur, qui sont verrouillés : un tour n'est jamais compté deux fois, même si la voiture recule à travers la ligne puis la refranchit. `finish` et `dnf` terminent l'épisode, donc ils ne tombent qu'une fois. Les primes restent petites devant la progression (20 et 50, contre environ 400 par tour), parce qu'elles ne font que marquer l'événement.
- **Pas de survie passive.** Aucun terme n'est positif sans progression. Une voiture à l'arrêt prend `slow` à chaque pas, puis le DNF au bout de 10 s.
- **Pas de vitesse brute.** On paie les mètres crédités par le simulateur, pas la vitesse. Une voiture qui tourne en rond ou qui roule en travers ne gagne rien. C'est l'actualisation (γ < 1) qui pousse à aller vite : les mêmes mètres, gagnés plus tôt, valent plus.
- **Contacts non pénalisés (ADR 0004).** Un contact peut rapporter (mur fin traversé, poussée). Il est journalisé avec un poids nul jusqu'à ce qu'E-47 mesure qu'il coûte plus qu'il ne rapporte.
- **Raccourcis.** Un raccourci qui laisse la voiture en course est payé, parce que le simulateur crédite les mètres gagnés : +78 m pour le mur traversé à Zandvoort. Un raccourci mortel coûte : à Sochi, on gagne +126 m mais on prend -200 pour le DNF, d'où -74. Le DNF est fixé à -200, soit environ un demi-tour, précisément pour que ce cas reste perdant.
- **Rang.** Le rang n'entre pas dans la reward solo : en course solo il ne bouge pas, et un terme de rang se gagnerait et se perdrait à chaque échange de place. Les termes du champion (dépassement confirmé, perte durable d'une place) sont prévus pour E-26.

### Ordres de grandeur mesurés

`uv run python scripts/reward_scale.py` joue un épisode complet par circuit d'entraînement (15 circuits, seed 0, troncature à 8000 pas) avec cinq politiques de référence. Moyennes par épisode :

| Politique | Issues | Pas | Retour | progress | lap | finish | dnf | slow | reverse | action_change |
|---|---|---|---|---|---|---|---|---|---|---|
| poursuite 6 m/s | 15 arrivées | 4081 | 1327,6 | 1218,6 | 60,0 | 50,0 | 0 | -0,6 | 0 | -0,4 |
| poursuite 3 m/s | 7 arrivées, 8 tronqués | 7572 | 1218,4 | 1146,2 | 49,3 | 23,3 | 0 | -0,2 | 0 | -0,3 |
| à l'arrêt | 15 DNF | 201 | -220,0 | 0 | 0 | 0 | -200 | -20,0 | 0 | 0 |
| marche arrière | 15 DNF | 201 | -254,7 | -18,1 | 0 | 0 | -200 | -20,0 | -16,6 | 0 |
| aléatoire | 15 DNF | 235 | -222,8 | 4,2 | 0 | 0 | -200 | -20,7 | 0 | -6,2 |

La poursuite suit la ligne centrale sans rien exploiter, et ne touche jamais de mur. Les termes de contact restent donc à 0 dans cette mesure.

Ce qu'on en retient :

- **La progression domine** : 92 % du retour d'un pilote propre. Les pénalités de forme pèsent moins de 0,1 % de ce retour, donc elles ne gênent pas un pilote qui roule bien.
- **Toute conduite qui ne progresse pas coûte** entre -220 et -255, et le DNF en fait l'essentiel.
- **Par pas**, la progression vaut environ 0,3 à 6 m/s et 0,5 à 10 m/s. Les termes de forme valent au plus 0,1 à 0,2 : l'échelle reste de l'ordre de 1, ce qui convient à PPO et à SAC sans renormaliser.

### Ce qui reste à régler

Les coefficients v1 sont des valeurs de départ vérifiées en ordre de grandeur, pas un résultat d'entraînement. Ils seront ajustés à partir des expériences sur les circuits d'entraînement et de validation (E-18, E-19, E-24), jamais sur le test scellé. Les poids des contacts attendent les mesures d'E-47.
