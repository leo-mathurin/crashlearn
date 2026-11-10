# Relecture et validation

Trois tours de `sol-loop`, avec **gpt-6.1-sol**, raisonnement **xhigh**, dans T3 Code/Codex.
Les trois verdicts sont **APPROVE**.
L'usage en tokens n'est pas exposé par le harnais.

| Tour | Head relu | Périmètre | Verdict |
|---|---|---|---|
| 1 | `9e22571` | Suivi, reprise, export et cache ajoutés à la chaîne E-18 | APPROVE, aucun signalement |
| 2 | `e7a344f` | Chaîne E-18 reprise dans la PR et protocole complet | APPROVE, aucun signalement |
| 3 | `a900b2a` | Isolation CI et corrections Greptile | APPROVE, quatre points résolus |

La différence initiale ajoutée à `ffdb512` comportait 14 fichiers, 754 insertions et
81 suppressions. Au head `a900b2a`, après les corrections et la documentation, hors données générées :
18 fichiers, 939 insertions et 96 suppressions. Les ajouts de revue couvrent des défauts
reproduits et leurs tests ; aucun signalement n'a été refusé et aucun nit n'est resté ouvert.

| Point | Défaut vérifié | Correction et preuve |
|---|---|---|
| CI1 | Fixture de session fermée par un autre test | État de deux voitures par test ; quatre échecs reproduits puis corrigés |
| GT1 | Ancien modèle évalué avec des features ou un historique modifiés | Deux hashes contrôlés avant chargement ; deux cas de rejet testés |
| GT2 | Courses dupliquées ou modèles de deux tentatives mélangés | Validation du jeu de résultats complet avant calcul ; deux cas de rejet testés |
| GT3 | Erreur de nettoyage pouvant empêcher le manifest final | Fermeture et écriture dans des `finally` distincts ; erreurs logger et workers testées |

Les trois conversations Greptile sont résolues dans la
[PR #19](https://github.com/leo-mathurin/crashlearn/pull/19).
Les fichiers d'agents et leurs poids de fin de budget sont générés après cette relecture.
Leur preuve directe est [submission-proof.json](selection/submission-proof.json) : vérificateur
officiel, course autonome, processus partagé et absence des frameworks d'entraînement.
Les courses et manifests de toute la campagne ont aussi été contrôlés avant export.
