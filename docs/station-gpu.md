# Station Ubuntu et image Docker GPU (E-5)

## Machine contrôlée le 28 septembre 2026

| Élément | Valeur observée |
| --- | --- |
| OS | Ubuntu Server 26.04.1 LTS, noyau `7.0.0-34-generic` |
| CPU / mémoire | Ryzen 7 7800X3D / 32 Go |
| GPU / VRAM | NVIDIA GeForce RTX 5070 / 12 227 MiB |
| Pilote / CUDA annoncé par `nvidia-smi` | `595.91.07` / `13.2` |
| Docker / Compose | `29.8.1` / `v5.5.1` |
| Accès | SSH par clé via Tailscale ; Ubuntu démarre par défaut, Windows reste disponible |

`nvidia-smi` sur l'hôte et depuis `docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi -L` détectent tous deux la RTX 5070. Ce contrôle de visibilité du périphérique est distinct d'un calcul CUDA ; le test PyTorch ci-dessous vérifie le calcul.

## Construire l'image

La source du simulateur est la copie non modifiée du 18/09/2026 dans `vendor/simulation/` ([provenance](../vendor/PROVENANCE.md)). Son Dockerfile installe PyTorch **CPU** et les versions ONNX du tournoi. La couche `docker/Dockerfile.gpu` remplace uniquement PyTorch par `2.13.0+cu130` depuis l'[index CUDA officiel](https://download.pytorch.org/whl/cu130), puis ajoute Gymnasium, SB3 et W&B à versions exactes. Le build vérifie que `onnx==1.22.0` et `onnxruntime==1.29.0` n'ont pas changé.

```bash
git rev-parse HEAD
docker build -t crashlearn-sim:20260918 \
  -f vendor/simulation/Dockerfile vendor/simulation
docker build -t crashlearn-train-gpu:20260928 \
  -f docker/Dockerfile.gpu .
docker image inspect --format '{{.Id}}' crashlearn-train-gpu:20260928
```

La première image doit être reconstruite depuis la version officielle conservée dans ce dépôt avant de construire la seconde. Les tags indiqués sont locaux. Pour le contrôle du 28 septembre, la source était `31f5442`, l'ID de l'image de base `sha256:8437590a2837dd15d96ad358ffffe563bba68c573f00ef800728db813e2889b0` et celui de l'image GPU `sha256:e2412c7b844693c88859af65b3322b6fe34e520514fb0b571ffafc5fb1382376`. Conserver l'ID de l'image avec chaque run : le Dockerfile de base utilise `python:3.11-slim` sans digest et ses dépendances indirectes peuvent évoluer lors d'une reconstruction.

## Calcul, run court et checkpoint

Depuis la racine du dépôt sur la station :

```bash
mkdir -p runs/station-smoke
docker run --rm --runtime=nvidia --gpus all \
  -v "$PWD/runs/station-smoke:/results" \
  crashlearn-train-gpu:20260928 \
  python /usr/local/bin/station_smoke.py --device cuda --output-dir /results \
  --matrix-size 2048 --repetitions 100 --timesteps 8192
docker run --rm --runtime=nvidia --gpus all \
  -v "$PWD/runs/station-smoke:/results" \
  crashlearn-train-gpu:20260928 \
  python /usr/local/bin/station_smoke.py --device cpu --output-dir /results \
  --matrix-size 2048 --repetitions 100 --timesteps 8192
```

Le script utilise les mêmes matrices `float32`, dimensions, nombre de répétitions, graine, environnement CartPole et paramètres PPO sur les deux appareils. Il vérifie le résultat du produit matriciel CUDA contre le CPU, mesure le débit de chaque appareil, entraîne PPO pendant 8 192 interactions, enregistre un checkpoint et confirme qu'un modèle rechargé donne la même action déterministe. Résultats et checkpoints sont écrits dans `runs/station-smoke/`, hors Git.

| Mesure du 28/09/2026 | CPU | GPU RTX 5070 |
| --- | ---: | ---: |
| Matmul 2048², 100 répétitions | 1,669 s ; 1 029 GFLOP/s | 0,072 s ; 23 772 GFLOP/s |
| PPO CartPole, 8 192 étapes | 1,276 s ; 6 422 étapes/s | 2,575 s ; 3 181 étapes/s |

Le calcul matriciel est environ 23 fois plus rapide sur GPU dans ce test. Pour ce petit PPO MLP, le CPU est environ deux fois plus rapide. L'image exécutée contient Python 3.11.16, PyTorch `2.13.0+cu130`, CUDA runtime 13.0, Stable-Baselines3 2.9.0, W&B 0.30.0, ONNX 1.22.0 et ONNX Runtime 1.29.0. Les deux checkpoints ont été rechargés dans le conteneur. Le checkpoint GPU a ensuite été copié sur un SSD externe, rapatrié dans un autre dossier de la station et rechargé sur CUDA avec `PPO.load` ; son SHA-256 après retour était `da60d672779f1b9c0a55272fd7626f4d204e98643b48a535f21f536457ef0283`. La reprise d'entraînement a fait passer le modèle de 8 192 à 8 448 étapes.

Ce run est un **test de la station**, pas un entraînement du pilote Crash & Learn. L'entraînement du pilote avec le wrapper Gymnasium devra être validé séparément une fois la boucle d'entraînement du projet disponible. Le débit PPO CartPole ne prédit pas celui du simulateur de course : la simulation peut être limitée par le CPU, et une petite politique MLP peut être plus lente sur GPU.

## Démarrage et arrêt de la station

Sur le Mac de Léo :

```bash
tour status
tour on             # Ubuntu par défaut
tour on windows     # Windows au prochain démarrage
tour off            # arrêt propre, puis coupure de la prise
```

La connexion SSH utilise Tailscale et une clé, sans mot de passe dans le dépôt. La commande `tour` attend que le système choisi réponde ; l'allumage depuis une prise coupée passe par Home Assistant. Les identifiants W&B et les autres secrets restent hors Git. Pour interrompre un entraînement long, attendre la confirmation d'un checkpoint avant `tour off` ; la politique de checkpoints périodiques du véritable entraîneur reste à implémenter.
