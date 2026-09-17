# Provenance du simulateur

| Champ | Valeur |
|---|---|
| Archive | `simulation.zip` |
| Origine | Ressources du projet Epitech « Crash & Learn » (fournie par l'équipe pédagogique) |
| Taille | 4 567 078 octets |
| sha256 | `7110e6580d267d6bdfe6d6530b08b93e9c01679100af16bfa65896330ef8ba2e` |
| Dates des entrées | 2026-09-08 |
| Date d'import | 2026-09-16 (ticket E-8) |
| Version amont | `f110_gym` 0.2.1 (fork F1TENTH, d'après `setup.py`) ; commit amont inconnu |
| Emplacement | `vendor/simulation/` |

## Méthode d'import

```bash
unzip -q simulation.zip -x '__MACOSX/*' -d vendor/
sha256sum simulation.zip
```

Seules les métadonnées macOS (`__MACOSX/`) ont été écartées.

**Aucune modification à ce stade.** Tous les autres fichiers sont identiques à l'archive.

## Remarques

- Les écarts connus (friction non appliquée, doublon `Mexico City`/`MexicoCity`, détection de carte, etc.) seront corrigés dans le ticket E-12. Chaque correctif fera l'objet d'un commit séparé et visible dans le diff.
- `Dockerfile` et `docker-compose.yml` sont conservés tels quels, mais ne servent **pas** de référence : ils n'ont pas de configuration GPU, et SB3 y est annoncé sans être installé. L'image reproductible fait l'objet d'un ticket dédié.
- Ce dossier est exclu de Ruff et de pytest (voir `pyproject.toml`).
