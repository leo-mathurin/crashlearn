# E-19 : atteindre 90 % de courses terminées

Protocole fixé le 10 novembre 2026 avant le premier entraînement d'adaptation.

La référence est le PPO réactif seed 0 de E-18 : 7 arrivées sur 12 courses
nominales de validation. Ses poids et ses résultats sont conservés dans
`/home/leo/runs/crashlearn/e18-20261109` et `docs/results/e-18/`.

La cible est au moins 11 arrivées sur les 12 courses initiales : Montreal,
Sochi, Spielberg et YasMarina, seeds 0, 1 et 2, trois tours, 8000 décisions.
Les DNF et les TIMEOUT restent dans le dénominateur. Après sélection, le modèle
est figé et confirmé sur les mêmes quatre circuits avec les seeds 100 à 129,
soit 120 courses supplémentaires. Cette confirmation doit aussi dépasser 90 %.
Les circuits test restent scellés. Aucun circuit de validation ne sert à entraîner.

Le premier essai reprend seulement les poids du meilleur réactif, avec un nouvel
optimiseur et un nouveau compteur. Il demande 2 000 000 interactions supplémentaires
sur les 15 circuits train, seed 0, 15 workers fixes, RTX 5070, reward v1 et
features inchangées. La seule modification du contrôle est un plafond de 5 m/s,
enregistré dans l'espace d'action et conservé dans l'export ONNX. Les anciens
modèles gardent leur plafond de 10 m/s. La fiabilité prime ici sur le chrono.

Le diagnostic préalable a essayé des plafonds de 10, 8, 7, 6, 5 et 4 m/s sans
réentraînement, seed 0. Aucun ne finit les quatre circuits : ralentir seul ne
suffit pas. Les résultats sont conservés dans `runs/reliability-20261110/`.

Si ce premier essai échoue, le changement suivant et sa raison seront ajoutés
ici avant le prochain entraînement. Chaque essai conserve son manifest, ses
checkpoints et ses échecs. E-18 reste une expérience historique distincte ; ses
mesures ne seront pas réécrites avec les nouveaux modèles.
