## Résultats par pilote

| Pilote | Partition | Condition | Courses | Arrivées | Arrivées, moyenne par carte | Écart au nominal | DNF | TIMEOUT | Chrono médian (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ppo-h1 | train | nominal | 225 | 76.9 % | 76.9 % | - | 52 | 0 | 148.60 |
| ppo-h1 | validation | nominal | 60 | 41.7 % | 41.7 % | - | 35 | 0 | 159.50 |
| ppo-h1 | validation | low_friction | 60 | 8.3 % | 8.3 % | -33.3 pts | 55 | 0 | 203.30 |
| ppo-h1 | validation | lidar_noise | 60 | 48.3 % | 48.3 % | +6.7 pts | 31 | 0 | 159.45 |
| ppo-h1 | validation | heavy_rain | 60 | 6.7 % | 6.7 % | -35.0 pts | 56 | 0 | 200.50 |
| ppo-h1 | validation | strict_margin | 60 | 41.7 % | 41.7 % | +0.0 pts | 35 | 0 | 159.50 |
| ppo-h4 | train | nominal | 225 | 93.8 % | 93.8 % | - | 14 | 0 | 146.95 |
| ppo-h4 | validation | nominal | 60 | 45.0 % | 45.0 % | - | 33 | 0 | 141.35 |
| ppo-h4 | validation | low_friction | 60 | 11.7 % | 11.7 % | -33.3 pts | 53 | 0 | 135.35 |
| ppo-h4 | validation | lidar_noise | 60 | 45.0 % | 45.0 % | +0.0 pts | 33 | 0 | 141.20 |
| ppo-h4 | validation | heavy_rain | 60 | 10.0 % | 10.0 % | -35.0 pts | 54 | 0 | 130.15 |
| ppo-h4 | validation | strict_margin | 60 | 45.0 % | 45.0 % | +0.0 pts | 33 | 0 | 141.35 |
| scripted | train | nominal | 45 | 100.0 % | 100.0 % | - | 0 | 0 | 196.75 |
| scripted | validation | nominal | 12 | 100.0 % | 100.0 % | - | 0 | 0 | 213.62 |
| scripted | validation | low_friction | 12 | 100.0 % | 100.0 % | +0.0 pts | 0 | 0 | 214.85 |
| scripted | validation | lidar_noise | 12 | 100.0 % | 100.0 % | +0.0 pts | 0 | 0 | 220.55 |
| scripted | validation | heavy_rain | 12 | 100.0 % | 100.0 % | +0.0 pts | 0 | 0 | 221.62 |
| scripted | validation | strict_margin | 12 | 100.0 % | 100.0 % | +0.0 pts | 0 | 0 | 213.62 |

## Comportement

Moyennes par course. Contacts par 100 m parcourus. Un saut de progression est un pas où le simulateur crédite plus d'1 m au-delà de la distance roulée, comme une coupe à travers un mur. L'écart crédité moins roulé est un peu positif pour une trajectoire qui coupe les virages, et très positif pour un raccourci.

| Pilote | Partition | Condition | Mur / 100 m | Voiture / 100 m | Immobile (pas) | Marche arrière (pas) | Variation des commandes | Sauts de progression | Crédité moins roulé (m) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ppo-h1 | train | nominal | 0.32 | 0.00 | 33.6 | 3.5 | 0.0158 | 0.35 | +12.9 |
| ppo-h1 | validation | nominal | 7.18 | 0.00 | 83.7 | 18.5 | 0.0129 | 0.35 | +5.7 |
| ppo-h1 | validation | low_friction | 31.22 | 0.00 | 136.4 | 30.4 | 0.0135 | 0.17 | -1.7 |
| ppo-h1 | validation | lidar_noise | 6.17 | 0.00 | 86.4 | 20.2 | 0.0137 | 0.50 | +6.0 |
| ppo-h1 | validation | heavy_rain | 30.89 | 0.00 | 135.3 | 32.0 | 0.0148 | 0.12 | -1.9 |
| ppo-h1 | validation | strict_margin | 7.18 | 0.00 | 83.7 | 18.5 | 0.0129 | 0.35 | +5.7 |
| ppo-h4 | train | nominal | 0.58 | 0.00 | 9.3 | 3.0 | 0.0095 | 0.08 | +10.7 |
| ppo-h4 | validation | nominal | 8.04 | 0.00 | 75.4 | 27.6 | 0.0079 | 0.10 | +1.8 |
| ppo-h4 | validation | low_friction | 33.89 | 0.00 | 124.7 | 50.4 | 0.0089 | 0.00 | -5.7 |
| ppo-h4 | validation | lidar_noise | 8.38 | 0.00 | 72.8 | 26.2 | 0.0082 | 0.07 | +1.8 |
| ppo-h4 | validation | heavy_rain | 36.86 | 0.00 | 120.6 | 47.6 | 0.0096 | 0.02 | -5.3 |
| ppo-h4 | validation | strict_margin | 8.04 | 0.00 | 75.4 | 27.6 | 0.0079 | 0.10 | +1.8 |
| scripted | train | nominal | 0.00 | 0.00 | 1.0 | 0.0 | 0.0145 | 0.18 | +35.8 |
| scripted | validation | nominal | 0.00 | 0.00 | 1.0 | 0.0 | 0.0160 | 0.00 | +26.4 |
| scripted | validation | low_friction | 0.00 | 0.00 | 1.0 | 0.0 | 0.0144 | 0.00 | +22.3 |
| scripted | validation | lidar_noise | 0.00 | 0.00 | 1.0 | 0.0 | 0.0296 | 0.00 | +25.5 |
| scripted | validation | heavy_rain | 0.00 | 0.00 | 1.0 | 0.0 | 0.0280 | 0.00 | +22.1 |
| scripted | validation | strict_margin | 0.00 | 0.00 | 1.0 | 0.0 | 0.0160 | 0.00 | +26.4 |

## Apport de la mémoire : ppo-h4 moins ppo-h1

Intervalles bootstrap à 95 % par seed d'entraînement. Gain de chrono positif : ppo-h4 plus rapide, sur les courses finies par les deux pilotes.

| Partition | Condition | Seeds | Arrivées (pts) | Gain de chrono (s) | Conclusion |
|---|---|---:|---|---|---|
| train | nominal | 5 | [+2.2 ; +36.9] | [-0.1 ; +23.4] | h4 plus fiable |
| validation | nominal | 5 | [-15.0 ; +20.0] | [-1.2 ; +34.4] | non concluant |
| validation | low_friction | 5 | [-3.3 ; +10.0] | seeds insuffisantes | non concluant |
| validation | lidar_noise | 5 | [-26.7 ; +15.0] | [-1.3 ; +33.7] | non concluant |
| validation | heavy_rain | 5 | [-3.3 ; +11.7] | seeds insuffisantes | non concluant |
| validation | strict_margin | 5 | [-15.0 ; +20.0] | [-1.2 ; +34.4] | non concluant |

## Échecs conservés

484 courses sur 1155 non finies.

| Pilote | Seed d'entraînement | Carte | Condition | Seed | Statut | Tours |
|---|---:|---|---|---:|---|---:|
| ppo-h1 | 0 | Montreal | nominal | 0 | DNF | 0.62 |
| ppo-h1 | 0 | Montreal | nominal | 2 | DNF | 0.62 |
| ppo-h1 | 0 | YasMarina | nominal | 0 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | nominal | 1 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | nominal | 2 | DNF | 0.71 |
| ppo-h1 | 0 | Montreal | low_friction | 0 | DNF | 0.08 |
| ppo-h1 | 0 | Montreal | low_friction | 1 | DNF | 0.08 |
| ppo-h1 | 0 | Montreal | low_friction | 2 | DNF | 0.08 |
| ppo-h1 | 0 | Sochi | low_friction | 0 | DNF | 0.98 |
| ppo-h1 | 0 | Sochi | low_friction | 1 | DNF | 0.98 |
| ppo-h1 | 0 | Sochi | low_friction | 2 | DNF | 0.98 |
| ppo-h1 | 0 | Spielberg | low_friction | 0 | DNF | 0.33 |
| ppo-h1 | 0 | Spielberg | low_friction | 1 | DNF | 0.33 |
| ppo-h1 | 0 | Spielberg | low_friction | 2 | DNF | 0.33 |
| ppo-h1 | 0 | YasMarina | low_friction | 0 | DNF | 0.28 |
| ppo-h1 | 0 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h1 | 0 | YasMarina | low_friction | 2 | DNF | 0.28 |
| ppo-h1 | 0 | YasMarina | lidar_noise | 0 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | lidar_noise | 1 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | lidar_noise | 2 | DNF | 0.71 |
| ppo-h1 | 0 | Montreal | heavy_rain | 0 | DNF | 0.08 |
| ppo-h1 | 0 | Montreal | heavy_rain | 1 | DNF | 0.08 |
| ppo-h1 | 0 | Montreal | heavy_rain | 2 | DNF | 0.08 |
| ppo-h1 | 0 | Sochi | heavy_rain | 0 | DNF | 0.98 |
| ppo-h1 | 0 | Sochi | heavy_rain | 1 | DNF | 0.98 |
| ppo-h1 | 0 | Sochi | heavy_rain | 2 | DNF | 0.83 |
| ppo-h1 | 0 | Spielberg | heavy_rain | 0 | DNF | 0.33 |
| ppo-h1 | 0 | Spielberg | heavy_rain | 1 | DNF | 0.33 |
| ppo-h1 | 0 | Spielberg | heavy_rain | 2 | DNF | 0.33 |
| ppo-h1 | 0 | YasMarina | heavy_rain | 0 | DNF | 0.28 |
| ppo-h1 | 0 | YasMarina | heavy_rain | 1 | DNF | 0.28 |
| ppo-h1 | 0 | YasMarina | heavy_rain | 2 | DNF | 0.28 |
| ppo-h1 | 0 | Montreal | strict_margin | 0 | DNF | 0.62 |
| ppo-h1 | 0 | Montreal | strict_margin | 2 | DNF | 0.62 |
| ppo-h1 | 0 | YasMarina | strict_margin | 0 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | strict_margin | 1 | DNF | 0.71 |
| ppo-h1 | 0 | YasMarina | strict_margin | 2 | DNF | 0.71 |
| ppo-h1 | 1 | Montreal | nominal | 0 | DNF | 1.30 |
| ppo-h1 | 1 | Montreal | nominal | 1 | DNF | 1.30 |
| ppo-h1 | 1 | Montreal | nominal | 2 | DNF | 1.30 |
| ppo-h1 | 1 | YasMarina | nominal | 0 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | nominal | 1 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | nominal | 2 | DNF | 0.28 |
| ppo-h1 | 1 | Montreal | low_friction | 0 | DNF | 0.17 |
| ppo-h1 | 1 | Montreal | low_friction | 1 | DNF | 0.17 |
| ppo-h1 | 1 | Montreal | low_friction | 2 | DNF | 0.17 |
| ppo-h1 | 1 | Sochi | low_friction | 0 | DNF | 0.36 |
| ppo-h1 | 1 | Sochi | low_friction | 1 | DNF | 0.81 |
| ppo-h1 | 1 | Sochi | low_friction | 2 | DNF | 0.81 |
| ppo-h1 | 1 | Spielberg | low_friction | 0 | DNF | 1.92 |
| ppo-h1 | 1 | Spielberg | low_friction | 1 | DNF | 0.92 |
| ppo-h1 | 1 | Spielberg | low_friction | 2 | DNF | 1.92 |
| ppo-h1 | 1 | YasMarina | low_friction | 0 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | low_friction | 2 | DNF | 0.28 |
| ppo-h1 | 1 | Montreal | lidar_noise | 0 | DNF | 1.30 |
| ppo-h1 | 1 | Montreal | lidar_noise | 1 | DNF | 2.30 |
| ppo-h1 | 1 | Montreal | lidar_noise | 2 | DNF | 2.30 |
| ppo-h1 | 1 | YasMarina | lidar_noise | 0 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | lidar_noise | 1 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | lidar_noise | 2 | DNF | 0.28 |
| ppo-h1 | 1 | Montreal | heavy_rain | 0 | DNF | 0.17 |
| ppo-h1 | 1 | Montreal | heavy_rain | 1 | DNF | 0.17 |
| ppo-h1 | 1 | Montreal | heavy_rain | 2 | DNF | 0.17 |
| ppo-h1 | 1 | Sochi | heavy_rain | 0 | DNF | 0.81 |
| ppo-h1 | 1 | Sochi | heavy_rain | 1 | DNF | 1.36 |
| ppo-h1 | 1 | Sochi | heavy_rain | 2 | DNF | 0.81 |
| ppo-h1 | 1 | Spielberg | heavy_rain | 0 | DNF | 1.92 |
| ppo-h1 | 1 | Spielberg | heavy_rain | 1 | DNF | 0.92 |
| ppo-h1 | 1 | Spielberg | heavy_rain | 2 | DNF | 0.92 |
| ppo-h1 | 1 | YasMarina | heavy_rain | 0 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | heavy_rain | 1 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | heavy_rain | 2 | DNF | 0.28 |
| ppo-h1 | 1 | Montreal | strict_margin | 0 | DNF | 1.30 |
| ppo-h1 | 1 | Montreal | strict_margin | 1 | DNF | 1.30 |
| ppo-h1 | 1 | Montreal | strict_margin | 2 | DNF | 1.30 |
| ppo-h1 | 1 | YasMarina | strict_margin | 0 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | strict_margin | 1 | DNF | 0.28 |
| ppo-h1 | 1 | YasMarina | strict_margin | 2 | DNF | 0.28 |
| ppo-h1 | 2 | Austin | nominal | 0 | DNF | 0.78 |
| ppo-h1 | 2 | Hockenheim | nominal | 0 | DNF | 0.46 |
| ppo-h1 | 2 | Hockenheim | nominal | 1 | DNF | 0.46 |
| ppo-h1 | 2 | Hockenheim | nominal | 2 | DNF | 0.46 |
| ppo-h1 | 2 | Monza | nominal | 0 | DNF | 2.17 |
| ppo-h1 | 2 | Monza | nominal | 1 | DNF | 2.17 |
| ppo-h1 | 2 | Sakhir | nominal | 1 | DNF | 0.14 |
| ppo-h1 | 2 | Sakhir | nominal | 2 | DNF | 0.14 |
| ppo-h1 | 2 | Montreal | nominal | 0 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | nominal | 1 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | nominal | 2 | DNF | 0.62 |
| ppo-h1 | 2 | Spielberg | nominal | 0 | DNF | 0.32 |
| ppo-h1 | 2 | Spielberg | nominal | 1 | DNF | 0.33 |
| ppo-h1 | 2 | Spielberg | nominal | 2 | DNF | 1.33 |
| ppo-h1 | 2 | YasMarina | nominal | 0 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | nominal | 1 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | nominal | 2 | DNF | 0.28 |
| ppo-h1 | 2 | Montreal | low_friction | 0 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | low_friction | 1 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | low_friction | 2 | DNF | 0.62 |
| ppo-h1 | 2 | Spielberg | low_friction | 0 | DNF | 1.93 |
| ppo-h1 | 2 | Spielberg | low_friction | 1 | DNF | 0.92 |
| ppo-h1 | 2 | Spielberg | low_friction | 2 | DNF | 0.93 |
| ppo-h1 | 2 | YasMarina | low_friction | 0 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | low_friction | 2 | DNF | 0.28 |
| ppo-h1 | 2 | Montreal | lidar_noise | 0 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | lidar_noise | 1 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | lidar_noise | 2 | DNF | 0.62 |
| ppo-h1 | 2 | Spielberg | lidar_noise | 0 | DNF | 0.33 |
| ppo-h1 | 2 | YasMarina | lidar_noise | 0 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | lidar_noise | 1 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | lidar_noise | 2 | DNF | 0.28 |
| ppo-h1 | 2 | Montreal | heavy_rain | 0 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | heavy_rain | 1 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | heavy_rain | 2 | DNF | 0.62 |
| ppo-h1 | 2 | Sochi | heavy_rain | 2 | DNF | 0.98 |
| ppo-h1 | 2 | Spielberg | heavy_rain | 0 | DNF | 0.92 |
| ppo-h1 | 2 | Spielberg | heavy_rain | 1 | DNF | 0.92 |
| ppo-h1 | 2 | Spielberg | heavy_rain | 2 | DNF | 0.92 |
| ppo-h1 | 2 | YasMarina | heavy_rain | 0 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | heavy_rain | 1 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | heavy_rain | 2 | DNF | 0.28 |
| ppo-h1 | 2 | Montreal | strict_margin | 0 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | strict_margin | 1 | DNF | 0.62 |
| ppo-h1 | 2 | Montreal | strict_margin | 2 | DNF | 0.62 |
| ppo-h1 | 2 | Spielberg | strict_margin | 0 | DNF | 0.32 |
| ppo-h1 | 2 | Spielberg | strict_margin | 1 | DNF | 0.33 |
| ppo-h1 | 2 | Spielberg | strict_margin | 2 | DNF | 1.33 |
| ppo-h1 | 2 | YasMarina | strict_margin | 0 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | strict_margin | 1 | DNF | 0.28 |
| ppo-h1 | 2 | YasMarina | strict_margin | 2 | DNF | 0.28 |
| ppo-h1 | 3 | Austin | nominal | 0 | DNF | 0.12 |
| ppo-h1 | 3 | Austin | nominal | 1 | DNF | 0.12 |
| ppo-h1 | 3 | Austin | nominal | 2 | DNF | 0.12 |
| ppo-h1 | 3 | Budapest | nominal | 0 | DNF | 0.15 |
| ppo-h1 | 3 | Budapest | nominal | 1 | DNF | 0.15 |
| ppo-h1 | 3 | Budapest | nominal | 2 | DNF | 0.15 |
| ppo-h1 | 3 | Catalunya | nominal | 0 | DNF | 0.75 |
| ppo-h1 | 3 | Catalunya | nominal | 1 | DNF | 0.75 |
| ppo-h1 | 3 | Catalunya | nominal | 2 | DNF | 0.75 |
| ppo-h1 | 3 | Hockenheim | nominal | 0 | DNF | 0.46 |
| ppo-h1 | 3 | Hockenheim | nominal | 1 | DNF | 0.46 |
| ppo-h1 | 3 | Hockenheim | nominal | 2 | DNF | 0.46 |
| ppo-h1 | 3 | Melbourne | nominal | 0 | DNF | 0.89 |
| ppo-h1 | 3 | Melbourne | nominal | 1 | DNF | 0.89 |
| ppo-h1 | 3 | Melbourne | nominal | 2 | DNF | 0.89 |
| ppo-h1 | 3 | MexicoCity | nominal | 0 | DNF | 0.84 |
| ppo-h1 | 3 | MexicoCity | nominal | 1 | DNF | 0.84 |
| ppo-h1 | 3 | MexicoCity | nominal | 2 | DNF | 0.84 |
| ppo-h1 | 3 | Monza | nominal | 0 | DNF | 0.16 |
| ppo-h1 | 3 | Monza | nominal | 1 | DNF | 0.16 |
| ppo-h1 | 3 | Monza | nominal | 2 | DNF | 0.16 |
| ppo-h1 | 3 | Nuerburgring | nominal | 0 | DNF | 0.08 |
| ppo-h1 | 3 | Nuerburgring | nominal | 1 | DNF | 0.08 |
| ppo-h1 | 3 | Nuerburgring | nominal | 2 | DNF | 0.08 |
| ppo-h1 | 3 | Sakhir | nominal | 0 | DNF | 0.13 |
| ppo-h1 | 3 | Sakhir | nominal | 1 | DNF | 0.13 |
| ppo-h1 | 3 | Sakhir | nominal | 2 | DNF | 0.13 |
| ppo-h1 | 3 | Sepang | nominal | 0 | DNF | 0.14 |
| ppo-h1 | 3 | Sepang | nominal | 1 | DNF | 0.14 |
| ppo-h1 | 3 | Sepang | nominal | 2 | DNF | 0.14 |
| ppo-h1 | 3 | Silverstone | nominal | 0 | DNF | 0.18 |
| ppo-h1 | 3 | Silverstone | nominal | 1 | DNF | 0.18 |
| ppo-h1 | 3 | Silverstone | nominal | 2 | DNF | 0.18 |
| ppo-h1 | 3 | Montreal | nominal | 0 | DNF | 0.06 |
| ppo-h1 | 3 | Montreal | nominal | 1 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | nominal | 2 | DNF | 0.03 |
| ppo-h1 | 3 | Spielberg | nominal | 0 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | nominal | 1 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | nominal | 2 | DNF | 0.32 |
| ppo-h1 | 3 | YasMarina | nominal | 0 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | nominal | 1 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | nominal | 2 | DNF | 0.27 |
| ppo-h1 | 3 | Montreal | low_friction | 0 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | low_friction | 1 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | low_friction | 2 | DNF | 0.03 |
| ppo-h1 | 3 | Sochi | low_friction | 0 | DNF | 2.21 |
| ppo-h1 | 3 | Sochi | low_friction | 1 | DNF | 2.21 |
| ppo-h1 | 3 | Sochi | low_friction | 2 | DNF | 1.21 |
| ppo-h1 | 3 | Spielberg | low_friction | 0 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | low_friction | 1 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | low_friction | 2 | DNF | 0.32 |
| ppo-h1 | 3 | YasMarina | low_friction | 0 | DNF | 0.28 |
| ppo-h1 | 3 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h1 | 3 | YasMarina | low_friction | 2 | DNF | 0.28 |
| ppo-h1 | 3 | Montreal | lidar_noise | 0 | DNF | 0.06 |
| ppo-h1 | 3 | Montreal | lidar_noise | 1 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | lidar_noise | 2 | DNF | 0.06 |
| ppo-h1 | 3 | Spielberg | lidar_noise | 0 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | lidar_noise | 1 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | lidar_noise | 2 | DNF | 0.32 |
| ppo-h1 | 3 | YasMarina | lidar_noise | 0 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | lidar_noise | 1 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | lidar_noise | 2 | DNF | 0.27 |
| ppo-h1 | 3 | Montreal | heavy_rain | 0 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | heavy_rain | 1 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | heavy_rain | 2 | DNF | 0.03 |
| ppo-h1 | 3 | Sochi | heavy_rain | 0 | DNF | 1.21 |
| ppo-h1 | 3 | Sochi | heavy_rain | 1 | DNF | 2.21 |
| ppo-h1 | 3 | Sochi | heavy_rain | 2 | DNF | 2.21 |
| ppo-h1 | 3 | Spielberg | heavy_rain | 0 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | heavy_rain | 1 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | heavy_rain | 2 | DNF | 0.32 |
| ppo-h1 | 3 | YasMarina | heavy_rain | 0 | DNF | 0.28 |
| ppo-h1 | 3 | YasMarina | heavy_rain | 1 | DNF | 0.28 |
| ppo-h1 | 3 | YasMarina | heavy_rain | 2 | DNF | 0.28 |
| ppo-h1 | 3 | Montreal | strict_margin | 0 | DNF | 0.06 |
| ppo-h1 | 3 | Montreal | strict_margin | 1 | DNF | 0.03 |
| ppo-h1 | 3 | Montreal | strict_margin | 2 | DNF | 0.03 |
| ppo-h1 | 3 | Spielberg | strict_margin | 0 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | strict_margin | 1 | DNF | 0.32 |
| ppo-h1 | 3 | Spielberg | strict_margin | 2 | DNF | 0.32 |
| ppo-h1 | 3 | YasMarina | strict_margin | 0 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | strict_margin | 1 | DNF | 0.27 |
| ppo-h1 | 3 | YasMarina | strict_margin | 2 | DNF | 0.27 |
| ppo-h1 | 4 | Austin | nominal | 0 | DNF | 0.12 |
| ppo-h1 | 4 | Austin | nominal | 1 | DNF | 0.12 |
| ppo-h1 | 4 | Austin | nominal | 2 | DNF | 0.12 |
| ppo-h1 | 4 | Hockenheim | nominal | 0 | DNF | 1.47 |
| ppo-h1 | 4 | Hockenheim | nominal | 1 | DNF | 1.46 |
| ppo-h1 | 4 | Hockenheim | nominal | 2 | DNF | 1.46 |
| ppo-h1 | 4 | Melbourne | nominal | 0 | DNF | 0.88 |
| ppo-h1 | 4 | MexicoCity | nominal | 0 | DNF | 0.84 |
| ppo-h1 | 4 | MexicoCity | nominal | 1 | DNF | 0.84 |
| ppo-h1 | 4 | MexicoCity | nominal | 2 | DNF | 0.84 |
| ppo-h1 | 4 | Sepang | nominal | 2 | DNF | 0.57 |
| ppo-h1 | 4 | Montreal | nominal | 0 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | nominal | 1 | DNF | 1.62 |
| ppo-h1 | 4 | Montreal | nominal | 2 | DNF | 0.62 |
| ppo-h1 | 4 | YasMarina | nominal | 0 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | nominal | 1 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | nominal | 2 | DNF | 0.27 |
| ppo-h1 | 4 | Montreal | low_friction | 0 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | low_friction | 1 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | low_friction | 2 | DNF | 0.62 |
| ppo-h1 | 4 | Sochi | low_friction | 1 | DNF | 0.51 |
| ppo-h1 | 4 | Spielberg | low_friction | 0 | DNF | 1.33 |
| ppo-h1 | 4 | Spielberg | low_friction | 1 | DNF | 1.33 |
| ppo-h1 | 4 | Spielberg | low_friction | 2 | DNF | 0.33 |
| ppo-h1 | 4 | YasMarina | low_friction | 0 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | low_friction | 1 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | low_friction | 2 | DNF | 0.27 |
| ppo-h1 | 4 | Montreal | lidar_noise | 0 | DNF | 1.62 |
| ppo-h1 | 4 | Montreal | lidar_noise | 1 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | lidar_noise | 2 | DNF | 0.62 |
| ppo-h1 | 4 | YasMarina | lidar_noise | 0 | DNF | 0.28 |
| ppo-h1 | 4 | YasMarina | lidar_noise | 1 | DNF | 0.28 |
| ppo-h1 | 4 | YasMarina | lidar_noise | 2 | DNF | 0.28 |
| ppo-h1 | 4 | Montreal | heavy_rain | 0 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | heavy_rain | 1 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | heavy_rain | 2 | DNF | 0.62 |
| ppo-h1 | 4 | Sochi | heavy_rain | 0 | DNF | 0.87 |
| ppo-h1 | 4 | Spielberg | heavy_rain | 0 | DNF | 1.33 |
| ppo-h1 | 4 | Spielberg | heavy_rain | 1 | DNF | 1.33 |
| ppo-h1 | 4 | Spielberg | heavy_rain | 2 | DNF | 1.33 |
| ppo-h1 | 4 | YasMarina | heavy_rain | 0 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | heavy_rain | 1 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | heavy_rain | 2 | DNF | 0.27 |
| ppo-h1 | 4 | Montreal | strict_margin | 0 | DNF | 0.62 |
| ppo-h1 | 4 | Montreal | strict_margin | 1 | DNF | 1.62 |
| ppo-h1 | 4 | Montreal | strict_margin | 2 | DNF | 0.62 |
| ppo-h1 | 4 | YasMarina | strict_margin | 0 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | strict_margin | 1 | DNF | 0.27 |
| ppo-h1 | 4 | YasMarina | strict_margin | 2 | DNF | 0.27 |
| ppo-h4 | 0 | Montreal | nominal | 0 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | nominal | 1 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | nominal | 2 | DNF | 0.08 |
| ppo-h4 | 0 | Sochi | nominal | 0 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | nominal | 1 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | nominal | 2 | DNF | 0.81 |
| ppo-h4 | 0 | YasMarina | nominal | 0 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | nominal | 1 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | nominal | 2 | DNF | 0.71 |
| ppo-h4 | 0 | Montreal | low_friction | 0 | DNF | 0.90 |
| ppo-h4 | 0 | Montreal | low_friction | 1 | DNF | 1.90 |
| ppo-h4 | 0 | Montreal | low_friction | 2 | DNF | 0.90 |
| ppo-h4 | 0 | Sochi | low_friction | 0 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | low_friction | 1 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | low_friction | 2 | DNF | 0.81 |
| ppo-h4 | 0 | Spielberg | low_friction | 0 | DNF | 1.11 |
| ppo-h4 | 0 | Spielberg | low_friction | 1 | DNF | 1.11 |
| ppo-h4 | 0 | Spielberg | low_friction | 2 | DNF | 2.11 |
| ppo-h4 | 0 | YasMarina | low_friction | 0 | DNF | 0.51 |
| ppo-h4 | 0 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h4 | 0 | YasMarina | low_friction | 2 | DNF | 0.27 |
| ppo-h4 | 0 | Montreal | lidar_noise | 0 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | lidar_noise | 1 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | lidar_noise | 2 | DNF | 0.91 |
| ppo-h4 | 0 | Sochi | lidar_noise | 0 | DNF | 2.81 |
| ppo-h4 | 0 | Sochi | lidar_noise | 1 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | lidar_noise | 2 | DNF | 0.81 |
| ppo-h4 | 0 | YasMarina | lidar_noise | 0 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | lidar_noise | 1 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | lidar_noise | 2 | DNF | 0.71 |
| ppo-h4 | 0 | Montreal | heavy_rain | 0 | DNF | 0.90 |
| ppo-h4 | 0 | Montreal | heavy_rain | 1 | DNF | 0.90 |
| ppo-h4 | 0 | Montreal | heavy_rain | 2 | DNF | 0.90 |
| ppo-h4 | 0 | Sochi | heavy_rain | 0 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | heavy_rain | 1 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | heavy_rain | 2 | DNF | 0.81 |
| ppo-h4 | 0 | Spielberg | heavy_rain | 1 | DNF | 2.11 |
| ppo-h4 | 0 | YasMarina | heavy_rain | 0 | DNF | 0.25 |
| ppo-h4 | 0 | YasMarina | heavy_rain | 1 | DNF | 0.28 |
| ppo-h4 | 0 | YasMarina | heavy_rain | 2 | DNF | 0.25 |
| ppo-h4 | 0 | Montreal | strict_margin | 0 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | strict_margin | 1 | DNF | 0.08 |
| ppo-h4 | 0 | Montreal | strict_margin | 2 | DNF | 0.08 |
| ppo-h4 | 0 | Sochi | strict_margin | 0 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | strict_margin | 1 | DNF | 0.81 |
| ppo-h4 | 0 | Sochi | strict_margin | 2 | DNF | 0.81 |
| ppo-h4 | 0 | YasMarina | strict_margin | 0 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | strict_margin | 1 | DNF | 0.71 |
| ppo-h4 | 0 | YasMarina | strict_margin | 2 | DNF | 0.71 |
| ppo-h4 | 1 | Montreal | nominal | 0 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | nominal | 1 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | nominal | 2 | DNF | 0.62 |
| ppo-h4 | 1 | YasMarina | nominal | 0 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | nominal | 1 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | nominal | 2 | DNF | 0.71 |
| ppo-h4 | 1 | Montreal | low_friction | 0 | DNF | 0.08 |
| ppo-h4 | 1 | Montreal | low_friction | 1 | DNF | 0.08 |
| ppo-h4 | 1 | Montreal | low_friction | 2 | DNF | 0.08 |
| ppo-h4 | 1 | Sochi | low_friction | 0 | DNF | 0.21 |
| ppo-h4 | 1 | Sochi | low_friction | 1 | DNF | 0.21 |
| ppo-h4 | 1 | Sochi | low_friction | 2 | DNF | 0.21 |
| ppo-h4 | 1 | Spielberg | low_friction | 0 | DNF | 0.11 |
| ppo-h4 | 1 | Spielberg | low_friction | 1 | DNF | 0.11 |
| ppo-h4 | 1 | Spielberg | low_friction | 2 | DNF | 0.11 |
| ppo-h4 | 1 | YasMarina | low_friction | 0 | DNF | 0.25 |
| ppo-h4 | 1 | YasMarina | low_friction | 1 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | low_friction | 2 | DNF | 0.27 |
| ppo-h4 | 1 | Montreal | lidar_noise | 0 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | lidar_noise | 1 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | lidar_noise | 2 | DNF | 0.62 |
| ppo-h4 | 1 | Sochi | lidar_noise | 1 | DNF | 0.21 |
| ppo-h4 | 1 | YasMarina | lidar_noise | 0 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | lidar_noise | 1 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | lidar_noise | 2 | DNF | 0.28 |
| ppo-h4 | 1 | Montreal | heavy_rain | 0 | DNF | 0.17 |
| ppo-h4 | 1 | Montreal | heavy_rain | 1 | DNF | 0.17 |
| ppo-h4 | 1 | Montreal | heavy_rain | 2 | DNF | 0.62 |
| ppo-h4 | 1 | Sochi | heavy_rain | 0 | DNF | 0.21 |
| ppo-h4 | 1 | Sochi | heavy_rain | 1 | DNF | 0.21 |
| ppo-h4 | 1 | Sochi | heavy_rain | 2 | DNF | 0.21 |
| ppo-h4 | 1 | Spielberg | heavy_rain | 0 | DNF | 0.11 |
| ppo-h4 | 1 | Spielberg | heavy_rain | 1 | DNF | 0.11 |
| ppo-h4 | 1 | Spielberg | heavy_rain | 2 | DNF | 0.11 |
| ppo-h4 | 1 | YasMarina | heavy_rain | 0 | DNF | 0.24 |
| ppo-h4 | 1 | YasMarina | heavy_rain | 1 | DNF | 0.25 |
| ppo-h4 | 1 | YasMarina | heavy_rain | 2 | DNF | 0.25 |
| ppo-h4 | 1 | Montreal | strict_margin | 0 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | strict_margin | 1 | DNF | 0.62 |
| ppo-h4 | 1 | Montreal | strict_margin | 2 | DNF | 0.62 |
| ppo-h4 | 1 | YasMarina | strict_margin | 0 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | strict_margin | 1 | DNF | 0.28 |
| ppo-h4 | 1 | YasMarina | strict_margin | 2 | DNF | 0.71 |
| ppo-h4 | 2 | Monza | nominal | 0 | DNF | 2.17 |
| ppo-h4 | 2 | Monza | nominal | 1 | DNF | 2.17 |
| ppo-h4 | 2 | Monza | nominal | 2 | DNF | 1.17 |
| ppo-h4 | 2 | Montreal | nominal | 0 | DNF | 1.62 |
| ppo-h4 | 2 | Montreal | nominal | 1 | DNF | 2.30 |
| ppo-h4 | 2 | Montreal | nominal | 2 | DNF | 2.91 |
| ppo-h4 | 2 | YasMarina | nominal | 0 | DNF | 0.71 |
| ppo-h4 | 2 | YasMarina | nominal | 1 | DNF | 1.71 |
| ppo-h4 | 2 | YasMarina | nominal | 2 | DNF | 2.70 |
| ppo-h4 | 2 | Montreal | low_friction | 0 | DNF | 0.62 |
| ppo-h4 | 2 | Montreal | low_friction | 1 | DNF | 0.62 |
| ppo-h4 | 2 | Montreal | low_friction | 2 | DNF | 1.91 |
| ppo-h4 | 2 | Sochi | low_friction | 0 | DNF | 1.88 |
| ppo-h4 | 2 | Sochi | low_friction | 1 | DNF | 0.88 |
| ppo-h4 | 2 | Spielberg | low_friction | 0 | DNF | 1.33 |
| ppo-h4 | 2 | Spielberg | low_friction | 1 | DNF | 1.33 |
| ppo-h4 | 2 | YasMarina | low_friction | 0 | DNF | 0.27 |
| ppo-h4 | 2 | YasMarina | low_friction | 1 | DNF | 0.51 |
| ppo-h4 | 2 | YasMarina | low_friction | 2 | DNF | 0.51 |
| ppo-h4 | 2 | Montreal | lidar_noise | 0 | DNF | 1.18 |
| ppo-h4 | 2 | Montreal | lidar_noise | 1 | DNF | 2.62 |
| ppo-h4 | 2 | Montreal | lidar_noise | 2 | DNF | 0.91 |
| ppo-h4 | 2 | YasMarina | lidar_noise | 0 | DNF | 0.71 |
| ppo-h4 | 2 | YasMarina | lidar_noise | 1 | DNF | 2.51 |
| ppo-h4 | 2 | YasMarina | lidar_noise | 2 | DNF | 0.71 |
| ppo-h4 | 2 | Montreal | heavy_rain | 0 | DNF | 0.19 |
| ppo-h4 | 2 | Montreal | heavy_rain | 1 | DNF | 1.08 |
| ppo-h4 | 2 | Montreal | heavy_rain | 2 | DNF | 0.62 |
| ppo-h4 | 2 | Sochi | heavy_rain | 0 | DNF | 0.99 |
| ppo-h4 | 2 | Sochi | heavy_rain | 1 | DNF | 0.88 |
| ppo-h4 | 2 | Sochi | heavy_rain | 2 | DNF | 0.99 |
| ppo-h4 | 2 | Spielberg | heavy_rain | 0 | DNF | 2.33 |
| ppo-h4 | 2 | Spielberg | heavy_rain | 2 | DNF | 2.33 |
| ppo-h4 | 2 | YasMarina | heavy_rain | 0 | DNF | 0.51 |
| ppo-h4 | 2 | YasMarina | heavy_rain | 1 | DNF | 0.51 |
| ppo-h4 | 2 | YasMarina | heavy_rain | 2 | DNF | 0.51 |
| ppo-h4 | 2 | Montreal | strict_margin | 0 | DNF | 1.62 |
| ppo-h4 | 2 | Montreal | strict_margin | 1 | DNF | 2.30 |
| ppo-h4 | 2 | Montreal | strict_margin | 2 | DNF | 2.91 |
| ppo-h4 | 2 | YasMarina | strict_margin | 0 | DNF | 0.71 |
| ppo-h4 | 2 | YasMarina | strict_margin | 1 | DNF | 1.71 |
| ppo-h4 | 2 | YasMarina | strict_margin | 2 | DNF | 2.70 |
| ppo-h4 | 3 | Austin | nominal | 0 | DNF | 0.97 |
| ppo-h4 | 3 | Austin | nominal | 1 | DNF | 0.97 |
| ppo-h4 | 3 | Austin | nominal | 2 | DNF | 0.97 |
| ppo-h4 | 3 | SaoPaulo | nominal | 0 | DNF | 1.08 |
| ppo-h4 | 3 | SaoPaulo | nominal | 1 | DNF | 1.08 |
| ppo-h4 | 3 | Silverstone | nominal | 0 | DNF | 0.94 |
| ppo-h4 | 3 | Silverstone | nominal | 1 | DNF | 0.94 |
| ppo-h4 | 3 | Silverstone | nominal | 2 | DNF | 0.94 |
| ppo-h4 | 3 | Montreal | nominal | 0 | DNF | 0.90 |
| ppo-h4 | 3 | Montreal | nominal | 1 | DNF | 1.62 |
| ppo-h4 | 3 | Montreal | nominal | 2 | DNF | 0.62 |
| ppo-h4 | 3 | YasMarina | nominal | 0 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | nominal | 1 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | nominal | 2 | DNF | 0.07 |
| ppo-h4 | 3 | Montreal | low_friction | 0 | DNF | 0.47 |
| ppo-h4 | 3 | Montreal | low_friction | 1 | DNF | 0.47 |
| ppo-h4 | 3 | Montreal | low_friction | 2 | DNF | 0.08 |
| ppo-h4 | 3 | Sochi | low_friction | 0 | DNF | 0.81 |
| ppo-h4 | 3 | Sochi | low_friction | 1 | DNF | 0.81 |
| ppo-h4 | 3 | Sochi | low_friction | 2 | DNF | 0.81 |
| ppo-h4 | 3 | Spielberg | low_friction | 0 | DNF | 1.33 |
| ppo-h4 | 3 | Spielberg | low_friction | 2 | DNF | 0.33 |
| ppo-h4 | 3 | YasMarina | low_friction | 0 | DNF | 0.51 |
| ppo-h4 | 3 | YasMarina | low_friction | 1 | DNF | 0.24 |
| ppo-h4 | 3 | YasMarina | low_friction | 2 | DNF | 0.51 |
| ppo-h4 | 3 | Montreal | lidar_noise | 0 | DNF | 1.62 |
| ppo-h4 | 3 | Montreal | lidar_noise | 1 | DNF | 1.90 |
| ppo-h4 | 3 | Montreal | lidar_noise | 2 | DNF | 0.63 |
| ppo-h4 | 3 | YasMarina | lidar_noise | 0 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | lidar_noise | 1 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | lidar_noise | 2 | DNF | 0.07 |
| ppo-h4 | 3 | Montreal | heavy_rain | 0 | DNF | 0.17 |
| ppo-h4 | 3 | Montreal | heavy_rain | 1 | DNF | 0.47 |
| ppo-h4 | 3 | Montreal | heavy_rain | 2 | DNF | 0.08 |
| ppo-h4 | 3 | Sochi | heavy_rain | 0 | DNF | 0.81 |
| ppo-h4 | 3 | Sochi | heavy_rain | 1 | DNF | 0.81 |
| ppo-h4 | 3 | Sochi | heavy_rain | 2 | DNF | 0.81 |
| ppo-h4 | 3 | Spielberg | heavy_rain | 0 | DNF | 2.33 |
| ppo-h4 | 3 | Spielberg | heavy_rain | 1 | DNF | 0.33 |
| ppo-h4 | 3 | Spielberg | heavy_rain | 2 | DNF | 0.33 |
| ppo-h4 | 3 | YasMarina | heavy_rain | 0 | DNF | 0.51 |
| ppo-h4 | 3 | YasMarina | heavy_rain | 1 | DNF | 0.51 |
| ppo-h4 | 3 | YasMarina | heavy_rain | 2 | DNF | 0.24 |
| ppo-h4 | 3 | Montreal | strict_margin | 0 | DNF | 0.90 |
| ppo-h4 | 3 | Montreal | strict_margin | 1 | DNF | 1.62 |
| ppo-h4 | 3 | Montreal | strict_margin | 2 | DNF | 0.62 |
| ppo-h4 | 3 | YasMarina | strict_margin | 0 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | strict_margin | 1 | DNF | 0.07 |
| ppo-h4 | 3 | YasMarina | strict_margin | 2 | DNF | 0.07 |
| ppo-h4 | 4 | MexicoCity | nominal | 0 | DNF | 1.26 |
| ppo-h4 | 4 | MexicoCity | nominal | 1 | DNF | 0.26 |
| ppo-h4 | 4 | MexicoCity | nominal | 2 | DNF | 0.26 |
| ppo-h4 | 4 | Montreal | nominal | 0 | DNF | 0.91 |
| ppo-h4 | 4 | Montreal | nominal | 1 | DNF | 0.91 |
| ppo-h4 | 4 | Montreal | nominal | 2 | DNF | 2.91 |
| ppo-h4 | 4 | YasMarina | nominal | 0 | DNF | 0.71 |
| ppo-h4 | 4 | YasMarina | nominal | 1 | DNF | 0.70 |
| ppo-h4 | 4 | YasMarina | nominal | 2 | DNF | 0.70 |
| ppo-h4 | 4 | Montreal | low_friction | 0 | DNF | 0.46 |
| ppo-h4 | 4 | Montreal | low_friction | 1 | DNF | 0.62 |
| ppo-h4 | 4 | Montreal | low_friction | 2 | DNF | 0.46 |
| ppo-h4 | 4 | Sochi | low_friction | 1 | DNF | 0.82 |
| ppo-h4 | 4 | Sochi | low_friction | 2 | DNF | 1.51 |
| ppo-h4 | 4 | YasMarina | low_friction | 0 | DNF | 0.24 |
| ppo-h4 | 4 | YasMarina | low_friction | 1 | DNF | 0.25 |
| ppo-h4 | 4 | YasMarina | low_friction | 2 | DNF | 0.25 |
| ppo-h4 | 4 | Montreal | lidar_noise | 0 | DNF | 1.91 |
| ppo-h4 | 4 | Montreal | lidar_noise | 2 | DNF | 2.62 |
| ppo-h4 | 4 | YasMarina | lidar_noise | 0 | DNF | 0.70 |
| ppo-h4 | 4 | YasMarina | lidar_noise | 1 | DNF | 0.71 |
| ppo-h4 | 4 | YasMarina | lidar_noise | 2 | DNF | 0.70 |
| ppo-h4 | 4 | Montreal | heavy_rain | 0 | DNF | 0.31 |
| ppo-h4 | 4 | Montreal | heavy_rain | 1 | DNF | 0.47 |
| ppo-h4 | 4 | Montreal | heavy_rain | 2 | DNF | 0.47 |
| ppo-h4 | 4 | Sochi | heavy_rain | 0 | DNF | 0.82 |
| ppo-h4 | 4 | Sochi | heavy_rain | 1 | DNF | 0.81 |
| ppo-h4 | 4 | Sochi | heavy_rain | 2 | DNF | 1.81 |
| ppo-h4 | 4 | YasMarina | heavy_rain | 0 | DNF | 0.25 |
| ppo-h4 | 4 | YasMarina | heavy_rain | 1 | DNF | 0.24 |
| ppo-h4 | 4 | YasMarina | heavy_rain | 2 | DNF | 0.24 |
| ppo-h4 | 4 | Montreal | strict_margin | 0 | DNF | 0.91 |
| ppo-h4 | 4 | Montreal | strict_margin | 1 | DNF | 0.91 |
| ppo-h4 | 4 | Montreal | strict_margin | 2 | DNF | 2.91 |
| ppo-h4 | 4 | YasMarina | strict_margin | 0 | DNF | 0.71 |
| ppo-h4 | 4 | YasMarina | strict_margin | 1 | DNF | 0.70 |
| ppo-h4 | 4 | YasMarina | strict_margin | 2 | DNF | 0.70 |

1 entraînements non terminés.

- ppo-h1 seed 4 : interrupted
