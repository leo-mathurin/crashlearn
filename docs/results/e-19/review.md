# Relecture de l'adaptation E-19

`sol-loop`, gpt-6.1-sol, raisonnement xhigh. L'usage en tokens n'est pas exposé.
Le périmètre supplémentaire est comparé par `git diff 779d61b` ; la chaîne
antérieure a ses trois revues dans `../e-18/review.md`.

| Tour | Head relu | Périmètre | Verdict |
|---|---|---|---|
| 1 | `9b3a9cc` | Plafond d'action, initialisation de poids, départs diversifiés, reprise et export | APPROVE, aucun signalement |
| 2 | `7017d3c` | Ablation, contrôle CPU natif, provenance du candidat, résultats et agent livré | APPROVE, aucun signalement |

Le tour 2 vérifie aussi l'égalité des paramètres hors colonnes 102 à 104, l'erreur
d'équivalence nulle avec les entrées masquées, la filiation des manifests et les
hashes source, gelés et livrés. Les 118/120 courses forment 120 paires carte/seed
uniques ; les deux DNF sont conservés. Les décomptes des 105 autres courses et
les médianes de temps correspondent aux enregistrements bruts.

La différence passe de 11 fichiers, 263 insertions et 9 suppressions au tour 1 à
35 fichiers, 2370 insertions et 17 suppressions au tour 2, mesures générées et
artefacts compris. Aucun signalement n'a été corrigé ou refusé. Les deux revues
n'ont modifié aucun fichier et n'ont lancé ni entraînement ni test lourd.

L'approbation couvre le code et les preuves dans leur périmètre nominal solo.
Elle n'étend pas la mesure aux cartes scellées, à d'autres seeds d'entraînement
ou aux conditions de faible adhérence. La livraison conserve ces limites.
