# Pilote réactif : viser 100 % de finitions nominales

Protocole du 10 novembre 2026, avant sélection du nouveau candidat.

Le modèle livré au commit `80a2c02` termine 118/120 courses de confirmation.
Il échoue à Montreal, seeds 104 et 118, au même passage. Le modèle et toutes
ces mesures restent conservés dans `docs/results/e-19/` et dans l'archive
`/home/leo/runs/crashlearn/e19-20261110`.

La nouvelle cible est **100 % observés en validation nominale**, sur les quatre
circuits Montreal, Sochi, Spielberg et YasMarina, trois tours, départ officiel,
8000 décisions au plus. Les DNF et TIMEOUT comptent tous. Cela ne constitue pas
une garantie de réussite pour toutes les seeds possibles ou de nouvelles cartes.

## Sélection

Les 12 courses initiales et les 120 anciennes courses de confirmation deviennent
des cas de sélection, soit 132 paires carte/seed distinctes. Elles ne serviront
plus de confirmation indépendante pour cette nouvelle cible.

Le premier diagnostic reproduit les deux échecs avec l'agent ONNX réel, puis
essaie des plafonds globaux de 4,75, 4,50 et 4,25 m/s. La politique apprise et
la suppression des connexions de progression restent identiques. Aucun réglage
ne dépend du circuit ou de la seed. On évalue ensuite les 132 cas avec le plus
haut plafond qui corrige les deux échecs ; s'il échoue, on conserve ses résultats
et essaie le plafond suivant. Un éventuel changement d'apprentissage sera décrit
ici avant de l'entraîner sur les quinze circuits train, avec le GPU.

## Confirmation

Après sélection, l'agent est figé par ses hashes avant la première course :
**seeds 200 à 299 sur chaque carte, soit 400 courses nouvelles**. Le vérificateur
officiel et l'agent ONNX sont exécutés dans l'image CPU du tournoi, sans importer
PyTorch ni Stable-Baselines3. Le candidat doit terminer les 400 courses.

Si la confirmation échoue, les échecs restent enregistrés et le candidat n'est
pas déclaré à 100 %. Une nouvelle sélection nécessitera une nouvelle plage de
seeds de confirmation, fixée avant de l'exécuter. Les circuits test restent
scellés ; l'adhérence dégradée est une question distincte, suivie dans E-23.
