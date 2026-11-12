# Dashboard de course

Lecteur web local de replays de course, dans une scène 3D. Il lit les fichiers `.npz` de `vendor/simulation/sim_recorder.py`, sans compte, sans base distante et sans streaming. Vocabulaire et périmètre : [Dashboard de course et vocabulaire](https://linear.app/e-hamilton/document/dashboard-de-course-et-vocabulaire-0987cb2ff84a).

## Démarrer

Prérequis : [Bun](https://bun.sh) 1.4 ou plus récent, et `uv` pour enregistrer des replays.

```bash
uv run python scripts/record_replay.py --demo    # une fois, depuis la racine : 3 replays dans replays/
cd dashboard && bun install && bun start         # puis ouvrir http://localhost:5173
```

La page liste les replays du dossier `replays/` (ignoré par Git). On peut aussi choisir un fichier `.npz` ou le déposer dans la fenêtre. Un replay ouvert depuis la liste a une adresse stable : `http://localhost:5173/?replay=demo-4-voitures-spielberg.npz`.

Options du serveur : `bun start --port 8080 --replays <dossier> --maps <dossier>`. `bun dev` recharge la page à chaque modification du code.

### Enregistrer un replay

`scripts/record_replay.py` lance le `sim_recorder.py` fourni, sans le modifier, avec les pilotes de `submission/`. Il refuse les circuits du test scellé.

```bash
uv run python scripts/record_replay.py --map Sochi --drivers driver1 driver2 --seed 4
uv run python scripts/record_replay.py --map Austin --drivers driver1 driver2 driver1 crash-test
```

`crash-test` n'est pas un pilote livré : cette voiture roule tout droit à 3 m/s et finit en DNF. Elle sert à enregistrer un vrai abandon. Le replay `demo-dnf-sochi` en contient une.

## Utiliser

| Action | Souris ou clavier |
| --- | --- |
| Lecture, pause | bouton ou `Espace` |
| Avancer ou reculer d'une frame | `→` `←` (20 frames, soit 1 s, avec `Maj`) |
| Début, fin | `Début`, `Fin` |
| Vitesse de lecture, de 0,25× à 8× | boutons ou `↑` `↓` |
| Zoom | molette ou pincement, centré sur le curseur |
| Déplacement | glisser |
| Revenir à la vue d'ensemble | double-clic ou `R` |
| Afficher ou masquer la limite de DNF | interrupteur ou `L` |
| Ouvrir un replay | bouton ou `O` |
| Afficher les mesures de fluidité | `P` |

Vu de loin, une voiture de 0,58 m fait moins d'un pixel : une pastille de couleur marque alors sa position. Elle disparaît quand on zoome assez pour voir la voiture.

## Ce que la scène montre

Tout ce qui est sur la piste vient des données du simulateur. Rien n'est inventé.

| Élément | Source |
| --- | --- |
| Bitume | centerline et demi-largeurs de `<Carte>_centerline.csv` |
| Lignes de bord, vibreurs | même bord de piste ; rouge et blanc là où la centerline tourne à plus de 0,14 rad/m. Les vibreurs sont de la peinture sur le bord mesuré, sans relief |
| Ligne de départ | premier point de la centerline, la pose de départ du simulateur |
| Murs, 0,25 m de haut | pixels occupés de `<Carte>_map.png`, au seuil du simulateur (valeur ≤ 128). Un rayon LiDAR s'arrête sur ces mêmes pixels |
| Limite de DNF, tirets rouges | bord de piste + 1 m (`_OFFTRACK_MARGIN_M`), avec le calcul de `_is_off_track` |
| Voitures | poses enregistrées, à la taille de la boîte de collision : 0,58 × 0,31 m |

Une Voiture inactive (statut DNF ou arrivée) est masquée : le simulateur la retire de la piste. Entre deux frames (20 Hz), la pose est interpolée pour un mouvement fluide ; les valeurs affichées sont celles de la frame enregistrée.

Si la carte du replay est introuvable, les voitures sont affichées sur une grille neutre, avec un avertissement. Si le fichier n'est pas un replay valide, la page dit pourquoi et reste utilisable.

## Technique

| Choix | Raison |
| --- | --- |
| Three.js, `WebGPURenderer` | un seul code de scène : WebGPU quand le navigateur l'offre, WebGL 2 sinon (Firefox et Chrome sous Linux aujourd'hui) |
| TypeScript strict, sans framework d'interface | le HUD est mis à jour à chaque frame : du DOM direct coûte moins qu'un rendu virtuel, et il n'y a que deux dépendances à l'exécution (`three`, `fflate`) |
| Bun pour tout : serveur, bundler, tests | `Bun.serve` sert la page compilée à la volée, les replays et les cartes de `vendor/` sans copie. Pas de Vite, pas de Node |
| Décodage NPZ dans le navigateur | `fflate` pour le ZIP, lecteur `.npy` maison : le format du recorder reste la seule source |
| Textures procédurales | bitume, herbe et béton sont dessinés au chargement avec une graine fixe : aucun fichier image, même rendu partout |
| oxlint, oxfmt, `tsc`, `bun test`, Playwright | mêmes outils que les autres projets de l'équipe |

```
server.ts            serveur local : page, /api/replays, /api/maps
src/replay/          décodage NPZ, validation, horloge de lecture
src/track/           données de piste : centerline, limite de DNF, murs (sans Three.js, testé)
src/scene/           rendu : renderer, lumières, circuit, voitures, caméra
src/ui/              HUD : timeline, choix du replay, pastilles, mesures
tests/               tests unitaires (bun test)
e2e/                 tests navigateur, Chromium et Firefox (Playwright)
```

## Qualité

Ce sont les commandes de la CI (`.github/workflows/dashboard.yml`).

```bash
bun run lint && bun run format:check && bun run typecheck && bun run test && bun run build
```

Tests navigateur, hors CI : ils enregistrent les replays de démonstration s'ils manquent.

```bash
bunx playwright install chromium firefox    # une fois
bun run test:e2e                            # sur une machine sans écran : xvfb-run -a bun run test:e2e
```

Sans écran, Firefox n'a pas de WebGL : `xvfb-run` lui en donne un.

`window.dashboard` donne l'état de la page aux tests (`state()`) et mesure le coût d'une frame sans la limite de l'écran (`benchmark()`).
