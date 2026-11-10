# Campagne solo PPO du 9 et 10 novembre 2026

Dix modèles de fin de budget : cinq seeds (0 à 4), avec une ou quatre observations. Chaque modèle reçoit 5 000 000 interactions demandées, soit 5 007 360 après arrondi SB3. Les 15 workers couvrent les 15 circuits train. Reward, features, historique, paramètres PPO et image Docker restent identiques pour toute la campagne.

**1 050 courses d’évaluation des PPO et 105 courses du témoin scripté sont enregistrées.** Chaque fichier contient les 105 courses attendues, sans doublon. Les DNF et TIMEOUT restent dans le rapport. Les modèles de la campagne ont été évalués sur les circuits train et validation.

## Résultats

[Rapport complet](rapport.md), [sélection et hashes](selection/agents.json), [preuve des soumissions](selection/submission-proof.json), [relecture](review.md).

Conclusion du protocole sur la validation nominale : **non concluant**. Le bootstrap utilise les cinq seeds d’entraînement appariées. Une conclusion non concluante ne prouve pas l’inutilité de la mémoire.

![Fiabilité sur validation](comparaison.png)

La cible interne de 90 % de finitions nominales est évaluée sur les 12 courses de validation de chaque modèle. Le choix de livraison maximise ce nombre de finitions ; la plus petite seed départage les égalités. Le modèle reste celui de fin de budget.

| Agent | Modèle | Seed | Arrivées nominales sur validation |
|---|---|---:|---:|
| submission/driver1 | ppo-h1 | 0 | 7/12 |
| submission/driver2 | ppo-h4 | 1 | 6/12 |

**La cible nominale de 90 % n’est pas atteinte. E-19 reste ouvert.**

L’entraînement est nominal. La baisse sous faible adhérence et forte pluie reste à traiter dans le curriculum de robustesse E-23.

## Interruption et provenance

Le redémarrage du serveur a interrompu `ppo-h1-seed4` après le checkpoint `3099690`. Sa tentative et ses checkpoints sont conservés ; son manifest est `manifests/ppo-h1-seed4-interrupted.json`. Une relance propre `ppo-h1-seed4-retry1` repart de zéro avec le même budget. Seule cette relance fournit les courses du réactif seed 4. Les dix modèles comparés ont chacun une tentative complète sans reprise de poids.

Commit d’entraînement figé : `adf5a5ffbb49d8a6fe067bc9b229ec06914b8aa1`. Image : `sha256:ff88d08af5fd4a202ba98f0b86f027af167df440efd027a79350eadd869e5df2`. Le rapport est recalculé avec les gardes de comparaison de la PR #19. Chaque hash de modèle et chaque empreinte de prétraitement ont été comparés au manifest de leur entraînement.

Les JSON racine sont des tableaux standards, avec une course par ligne pour limiter le volume des diffs. Les valeurs mesurées sont inchangées. Les manifests et fichiers de sélection gardent leur mise en forme indentée.

## Archive locale durable

Les modèles PyTorch, checkpoints périodiques, CSV, données W&B et le code figé sont conservés dans
`/home/leo/runs/crashlearn/e18-20261109`. Les dix modèles de l’archive ont été revérifiés par SHA-256.
Les modèles ONNX de livraison sont aussi versionnés dans `submission/`.

## Soumissions et suivi

Les deux agents ONNX opset 17 passent le vérificateur officiel et une course de trois tours sur IMS dans l’image CPU du tournoi, depuis `/tmp`. Le même processus charge les deux agents et les appelle pour deux voitures sans interférence. Ni PyTorch ni Stable-Baselines3 ne sont importés. Le prétraitement est incorporé depuis les sources NumPy canoniques.

W&B est hors ligne : métriques et checkpoints sont conservés sur la station, avec les CSV de chaque tentative. Aucun compte partagé W&B n’est configuré ; la synchronisation cloud et son quota n’ont pas été validés. E-20 reste ouvert pour cette configuration.

Recalcul du rapport :

```bash
uv run python scripts/compare_drivers.py --results-dir docs/results/e-18 \
  --output docs/results/e-18/rapport.md
```
