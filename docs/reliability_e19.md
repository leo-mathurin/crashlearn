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

## Deuxième essai : diversifier les départs

Le premier essai finit 10/12 courses : Montreal 3/3, Sochi 3/3, Spielberg 3/3,
YasMarina 1/3. Des plafonds supplémentaires de 4,5 et 4 m/s sans réentraînement
font aussi échouer Montreal et YasMarina, seed 0. Le contrôle doit donc s'adapter
aux états rencontrés, au-delà d'un simple ralentissement.

Le deuxième essai reprend les mêmes poids initiaux de E-18 et tous les paramètres
du premier, dont le budget de deux millions et le plafond de 5 m/s. Seuls les
départs train changent : waypoint aléatoire, décalage latéral dans ±0,2 m et
cap dans ±0,1 rad de la tangente. La progression repart de zéro depuis ce point.
Cela décorrèle la phase du tour de la géométrie locale et expose la correction
de petits écarts. La grille officielle est rétablie après chaque reset, et les
évaluations gardent leurs départs officiels.

## Troisième essai : retirer les connexions de progression

Le deuxième entraînement finit aussi 10/12 courses : YasMarina 1/3. Son modèle
figé est `cap5-random-h1-seed0`, SHA-256
`f283194ea0334cfbde14c0f4925698d5226b335b89533c39dd1e3b940fc359aa`. Un diagnostic masque les colonnes
102 à 104 (sinus et cosinus de progression, tours accomplis) à l'inférence : les
12 courses initiales sont alors finies. Ce résultat motive une ablation explicite.

`scripts/ablate_progress.py` met à zéro les trois colonnes correspondantes dans
la première couche de l'acteur et du critique, sans modifier les autres poids,
les features canoniques ni le plafond appris. Un contrôle sur 256 entrées
compare ce modèle à l'original recevant des progressions masquées. Le modèle
dérivé a son propre hash et son propre manifest ; l'original reste conservé.
Il n'y a aucune interaction ni optimisation supplémentaire.

Le candidat est figé avant la confirmation seeds 100 à 129. Cette confirmation
porte sur les mêmes quatre circuits, en conditions nominales et départ officiel.
Elle mesure la répétabilité face au bruit du simulateur sur ces cartes ; elle
ne constitue pas une évaluation sur des cartes nouvelles. Aucun résultat de
confirmation ne sert à modifier ce candidat.

## Résultat du candidat figé

L'agent ONNX finit les 12 cas de sélection dans l'image CPU officielle, puis
118 des 120 courses de confirmation, soit 98,3 %. Montreal : 28/30 ; Sochi,
Spielberg et YasMarina : 30/30 chacun. Les deux DNF restent comptés. Aucun
fichier du candidat ne change entre le gel et la fin de ces courses.

La cible nominale est atteinte. Les [résultats détaillés](results/e-19/README.md)
conservent aussi le coût en chrono, les essais non retenus et les limites sous
faible adhérence et forte pluie. `submission/driver1` reçoit cet agent ; les
résultats et l'ancien agent de E-18 restent conservés.
