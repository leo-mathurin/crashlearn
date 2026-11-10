# Relecture de la calibration pour la cible de 100 %

`sol-loop`, gpt-6.1-sol, raisonnement xhigh. Usage en tokens non exposé.
Base de cette nouvelle modification : `80a2c02`, déjà relu pour E-19.
Commande de comparaison : `git diff 80a2c02`.

| Tour | Head relu | Périmètre | Verdict |
|---|---|---|---|
| 1 | `2e5a751` | Plafond d'inférence optionnel, sauvegarde/rechargement, export et provenance | APPROVE, aucun signalement |
| 2 | `0f05d92` | Agent final, 400 courses nouvelles, filiation, hashes et rapports | APPROVE, aucun signalement |

La revue confirme que le plafond affecte l'inférence, survit au rechargement et
atteint le graphe ONNX. Le comportement sans option, l'ablation de progression
et le run source sont préservés. Les deux bornes sont enregistrées. Aucun test
ou garde inutile n'est signalé ; aucune correction ou objection n'est ouverte.

La différence relue contient cinq fichiers, 198 insertions et trois suppressions,
diagnostic compris. Le log de tests observé indique 548 succès. Le reviewer n'a
modifié aucun fichier et n'a lancé ni entraînement ni test lourd.

Le tour 2 vérifie les 132 cas de sélection et les 400 cas nouveaux, uniques et
complets, les trois tours par arrivée, l'égalité des fichiers gelés, archivés
et livrés, et les 13 tenseurs appris inchangés. L'accord ONNX/PyTorch est aussi
reproduit sur 256 entrées : erreur maximale 4,768 × 10⁻⁷. Les décomptes
complémentaires et les médianes appariées correspondent aux enregistrements.

La différence du tour 2 contient 22 fichiers, 1260 insertions et 15 suppressions,
mesures et artefacts compris. Les logs relus indiquent 548 tests source passés,
puis 472 tests par défaut passés après livraison. Aucun signalement, aucune
correction et aucune objection ne restent ouverts. La revue ne relance ni
entraînement ni course.

Ces approbations couvrent le code et les preuves dans le périmètre nominal
observé. Les mesures E-19 précédentes restent historiques ; elles ne sont pas
ajoutées à la confirmation du nouveau candidat.
