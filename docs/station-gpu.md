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

Le 29 septembre, `scripts/station_race_smoke.py` a utilisé le wrapper Gymnasium de la [PR E-14](https://github.com/leo-mathurin/crashlearn/pull/9) avec la carte `example` et PPO sur CUDA. Il a entraîné 1 024 étapes en 7,711 s et créé des checkpoints périodiques aux étapes 256, 512, 768 et 1 024. Un checkpoint final, enregistré **après** la dernière mise à jour PPO, a été rechargé : les actions déterministes étaient identiques. L'entraînement a ensuite repris jusqu'à 1 280 étapes et un nouveau checkpoint a été enregistré. Le résultat et les empreintes SHA-256 restent sur la station dans `/home/leo/runs/station-race-periodic-v2/result.json`. Le script renseigne `gym.__version__` pour les métadonnées de sauvegarde Stable-Baselines3 : le paquet `gym` du simulateur est un espace de noms qui n'expose pas cette métadonnée.

Le test a été exécuté avec un export temporaire de E-14, ensuite retiré. Pour le reproduire, placer la branche E-14 dans un répertoire temporaire sur la station, puis lancer depuis ce répertoire :

```bash
docker run --rm --runtime=nvidia --gpus all --user 1000:1000 \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace/src:/workspace/vendor/simulation:/workspace/vendor/simulation/gym \
  -v "$PWD:/workspace:ro" -v "$HOME/runs:/home/leo/runs" -w /workspace \
  crashlearn-train-gpu:20260928 \
  python /home/leo/runs/station_race_smoke.py \
  --output-dir /home/leo/runs/station-race-periodic-v2 \
  --device cuda --steps 1024 --resume-steps 256 --checkpoint-every 256
```

Ces runs valident la station et la sauvegarde puis reprise d'un petit pilote de course. Le wrapper E-14 reste une PR séparée ; la politique de checkpoints de ce script est une validation courte, pas encore celle de l'entraîneur de production. Le débit PPO CartPole ne prédit pas celui du simulateur de course : la simulation peut être limitée par le CPU, et une petite politique MLP peut être plus lente sur GPU.

## Démarrage et arrêt de la station

Sur le Mac de Léo :

```bash
tour status
tour on             # Ubuntu par défaut
tour on windows     # Windows au prochain démarrage
tour off            # arrêt propre, puis coupure de la prise
```

La connexion SSH utilise Tailscale et une clé, sans mot de passe dans le dépôt. La commande `tour` attend que le système choisi réponde ; l'allumage depuis une prise coupée passe par Home Assistant. Les identifiants W&B et les autres secrets restent hors Git. Pour interrompre un entraînement long, attendre la confirmation d'un checkpoint avant `tour off` ; la politique de checkpoints périodiques du véritable entraîneur reste à implémenter.

### Extinction après inactivité

Sur Ubuntu, `agent-awake.service` vérifie toutes les 15 secondes les sessions de travail, les agents, les conteneurs Docker, les calculs GPU et l'activité CPU. Il demande un arrêt propre après une heure sans signal d'activité. Les connexions SSH maintenues par Mutagen ne comptent pas comme du travail. Le compteur repart de zéro au démarrage du service. La commande `agent-awake acquire|heartbeat|release` permet aux hooks des agents de signaler une tâche en cours ; les leases expirent après dix minutes sans heartbeat. Un échec de lecture de Docker ou d'une base T3 existante bloque l'extinction par précaution.

Le service s'installe avec `sudo ops/agent-awake/install.sh` sur la station. Vérification : `systemctl is-active agent-awake`, puis `sudo /usr/local/lib/tour/agent-awake-daemon --diagnose`. Il s'applique à Ubuntu ; Windows garde sa gestion d'alimentation séparée. L'arrêt automatique éteint l'OS, mais laisse la prise connectée sous tension pour permettre le prochain réveil à distance.

### Migration ciblée depuis dev-tower

Le seul dépôt de code restauré sur la tour est le monorepo `/home/leo/code/wondday/wondday`, à la demande de Léo. Les sauvegardes non reproductibles sont dans `/home/leo/backups/dev-tower/` sur Ubuntu : deux exports SQL Supabase, une archive du volume PostgreSQL Stremlist et l'archive complète des données utilisateur T3. La base T3 restaurée a passé `PRAGMA integrity_check` et est utilisée par T3 Code `0.0.42`. Deux worktrees wondday référencés par son historique ont été reconstruits depuis leurs branches GitHub dans `/home/leo/.t3/worktrees/wondday/`. La clé SOPS/age est dans `/home/leo/.config/sops/age/keys.txt` avec des permissions privées. Le VHDX original et l'archive `linux-projects-config.tar.gz` restent sur le SSD externe du Mac ; ils constituent la source de récupération si un autre fichier est nécessaire.

Mutagen synchronise les réglages partagés des agents depuis le Mac vers la tour par l'alias SSH `tour`. Les sessions sont `tour-agents-shared`, `tour-claude-config`, `tour-codex-config` et `tour-t3-worktree-clean`. Les configurations Claude et Codex incluent les hooks `agent-awake`. Seuls les fichiers de configuration et compétences nécessaires sont synchronisés ; les historiques, caches, dépôts et réglages Codex propres à macOS restent hors de la synchronisation. Un `~/.codex/config.toml` Linux réduit est installé séparément pour éviter les chemins macOS. `mutagen sync list` sur le Mac affiche l'état des sessions. Sur Ubuntu, Codex CLI est en version `0.159.0` et Claude Code en `2.1.284`. Un appel réel à Codex a réussi ; Claude demande une nouvelle connexion OAuth propre à la tour.

T3 Code `0.0.42` tourne comme service utilisateur systemd avec linger activé. Son interface locale répond sur le port 3773 ; Tailscale Serve la rend disponible sur `https://tour.tail86ad70.ts.net/` pour les appareils du tailnet. L'historique restauré comprend 18 projets et 185 fils. Le jumelage d'un nouveau navigateur passe par `t3 pair --tailscale` sur la tour.

La partition temporaire `LINUXBOOT` de 8 Gio a été supprimée après sauvegarde de la table GPT sur la tour et le SSD externe. Les partitions Windows, EFI et Ubuntu sont intactes, et la vérification GPT ne signale pas d'erreur. Secure Boot reste désactivé ; une éventuelle réactivation nécessitera un essai au démarrage dans l'UEFI, devant la tour.
