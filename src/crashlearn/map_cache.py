"""Cache immutable map geometry in training workers, without changing vendor files.

The official scanner rebuilds its distance transform twice on every seeded reset. Its
set_map method only assigns geometry from the map files, independent of the ray count or
RNG. Keep two maps per worker; scans read these arrays and never modify them.
"""

from functools import lru_cache
from types import SimpleNamespace


def cached_map_loader(original):
    @lru_cache(maxsize=2)
    def geometry(path, extension):
        state = SimpleNamespace()
        original(state, path, extension)
        return vars(state)

    def set_map(scanner, path, extension):
        scanner.__dict__.update(geometry(path, extension))
        return True

    return set_map
