# Circuits absents de l'archive de l'école

`Norisring.csv` vient de la [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database) (commit `e59595d1f3573b30d1ded6a08984935b957688e0`), copié sans modification. Cette base de l'université technique de Munich est la source des 23 circuits fournis : leurs centerlines reprennent ses points un à un, réduits à l'échelle. Norisring et Suzuka n'ont pas été repris. Suzuka n'est pas inclus ici : son tracé se croise (un pont), et le simulateur, qui projette la voiture sur le point de centerline le plus proche, fait sauter la progression de 41 % d'un tour au croisement puis déclare l'abandon.

| Fichier | sha256 |
|---|---|
| `Norisring.csv` | `8857d3c362ad2923c1f93c8d257498f50459770b9021adcc7969b71085c31d9a` |

Format : `x_m, y_m, w_tr_right_m, w_tr_left_m`, en mètres réels. `crashlearn.variants.load_new` le met à l'échelle des cartes de l'école (voir `docs/tracks.md`).

Licence : la base est publiée sous LGPL-3.0. Ses centerlines viennent d'OpenStreetMap (© contributeurs OpenStreetMap, ODbL).
