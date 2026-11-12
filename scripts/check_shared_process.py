"""Load driver1..driver4 in one process, the way the tournament does, and race them.

The tournament imports every agent.py into one shared Python process, from its own cwd.
This check builds that situation from any cwd: it loads four driver folders with the
official loader (dry-run included) while only NumPy, ONNX Runtime and the standard library
can be imported, then runs real races with four cars. It checks what verify_submission.py
does not: the external weights file, the race observations against the dry-run schema,
action types and units, inputs left unchanged, and memory that no other car or instance
can disturb.

A missing driver3 or driver4 is replaced by a copy of an existing driver, and the output
says so. The races test loading and process sharing, not racing between our drivers: the
format of the final races is unknown, and behaviour with opponents stays unmeasured.

Runs in the tournament CPU image without the repository package, for example:
    docker run --rm -v "$PWD:/workspace:ro" -w /tmp -e NUMBA_CACHE_DIR=/tmp/numba \
      crashlearn-sim:20260918 python /workspace/scripts/check_shared_process.py \
      --submission /workspace/submission
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.abc
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime  # noqa: F401  loaded before the import guard, as in the tournament

SIM_DIR = Path(__file__).resolve().parents[1] / "vendor" / "simulation"
sys.path.insert(0, str(SIM_DIR))
import agent_loader  # noqa: E402  only NumPy and the standard library

DRIVERS = ("driver1", "driver2", "driver3", "driver4")
DELIVERY_FILES = ("agent.py", "model.onnx", "model.onnx.data")
ALLOWED_PACKAGES = {"numpy", "onnxruntime"}
TRAINING_PACKAGES = ("torch", "onnx", "stable_baselines3", "gymnasium", "wandb", "crashlearn")
STATUS_FINISHED = 2

# onnx.proto field numbers: ModelProto.graph, GraphProto.initializer,
# TensorProto.external_data and data_location (value 1 is EXTERNAL).
MODEL_GRAPH, GRAPH_INITIALIZER, TENSOR_EXTERNAL_DATA, TENSOR_DATA_LOCATION = 7, 5, 13, 14
LOCATION_EXTERNAL = 1


class CheckError(Exception):
    pass


# --- delivery files ----------------------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7


def _fields(buf: bytes):
    """(field number, value) of a protobuf message; length-delimited values stay bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i : i + size], i + size
        elif wire in (1, 5):
            value, i = None, i + (8 if wire == 1 else 4)
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > len(buf):
            raise ValueError("truncated protobuf message")
        yield key >> 3, value


def external_initializers(model: bytes) -> list[dict[str, str]]:
    """external_data entries (location, offset, length) of each initializer kept outside."""
    tensors = []
    try:
        for field, graph in _fields(model):
            if field != MODEL_GRAPH:
                continue
            for field, tensor in _fields(graph):
                if field != GRAPH_INITIALIZER:
                    continue
                entries, external = {}, False
                for field, value in _fields(tensor):
                    if field == TENSOR_EXTERNAL_DATA:
                        entry = dict(_fields(value))
                        entries[entry.get(1, b"").decode()] = entry.get(2, b"").decode()
                    elif field == TENSOR_DATA_LOCATION:
                        external = value == LOCATION_EXTERNAL
                if external:
                    tensors.append(entries)
    except (IndexError, ValueError, UnicodeDecodeError) as error:
        raise CheckError(f"model.onnx is not a readable ONNX model: {error}") from error
    return tensors


def check_external_data(folder: Path) -> int:
    """Every external weight must lie inside folder/model.onnx.data. Returns their count."""
    data = folder / "model.onnx.data"
    if not data.is_file():
        raise CheckError(f"{folder.name}: model.onnx.data is missing")
    tensors = external_initializers((folder / "model.onnx").read_bytes())
    if not tensors:
        raise CheckError(f"{folder.name}: model.onnx does not read model.onnx.data")
    size = data.stat().st_size
    for entries in tensors:
        if entries.get("location") != data.name:
            raise CheckError(f"{folder.name}: weights read from {entries.get('location')!r}")
        offset = int(entries.get("offset", 0))
        length = int(entries.get("length", size - offset))
        if offset < 0 or length < 0 or offset + length > size:
            raise CheckError(f"{folder.name}: weights outside model.onnx.data ({size} bytes)")
    return len(tensors)


def check_export_record(folder: Path) -> None:
    """The delivery files are the ones export.json records, hashes read from that file."""
    record = json.loads((folder / "export.json").read_text())["files"]
    for name in DELIVERY_FILES:
        if name not in record:
            raise CheckError(f"{folder.name}: export.json does not record {name}")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != record[name]:
            raise CheckError(f"{folder.name}: {name} does not match export.json")


def build_grand_prix(submission: Path, target: Path) -> dict[str, str]:
    """Copy the delivery files of driver1..driver4 into target; return {driver: source}.

    A missing driver is a copy of an existing one, so the same agent.py is loaded twice.
    """
    present = [d for d in DRIVERS if (submission / d / "agent.py").is_file()]
    if not present:
        raise CheckError(f"no driver folder in {submission}")
    sources = {}
    for i, driver in enumerate(DRIVERS):
        source = driver if driver in present else present[i % len(present)]
        (target / driver).mkdir(parents=True)
        for name in DELIVERY_FILES:
            if not (submission / source / name).is_file():
                raise CheckError(f"{source}: {name} is missing")
            shutil.copy2(submission / source / name, target / driver / name)
        sources[driver] = source
    return sources


# --- one shared process --------------------------------------------------------


class _TournamentImports(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.partition(".")[0]
        if top in sys.stdlib_module_names or top in ALLOWED_PACKAGES:
            return None
        raise ImportError(f"{name} is not in the tournament: NumPy, ONNX Runtime, stdlib only")


@contextlib.contextmanager
def tournament_imports():
    """Refuse any new import outside NumPy, ONNX Runtime and the standard library."""
    finder = _TournamentImports()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def schema(value):
    """Keys, Python types and array shapes of an observation or info dict."""
    if isinstance(value, dict):
        return {key: schema(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape)
    return type(value).__name__


def same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, list | tuple):
        return type(a) is type(b) and len(a) == len(b) and all(map(same, a, b))
    return type(a) is type(b) and a == b


def check_in_bounds(name: str, value: float, bounds: tuple[float, float]) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise CheckError(f"{name} = {value} outside {bounds}")


def check_action(action, actions: dict) -> tuple[float, float]:
    """(target speed in m/s, steering in rad): Python floats, finite, within the bounds."""
    if not isinstance(action, tuple) or len(action) != 2:
        raise CheckError(f"predict() returned {action!r}, not a (speed, steering) tuple")
    for value, (name, space) in zip(action, actions.items(), strict=True):
        if type(value) is not float or not math.isfinite(value):
            raise CheckError(f"{name} = {value!r} is not a finite Python float")
        check_in_bounds(f"{name} ({space['unit']})", value, space["bounds"])
    return action


def check_observation(obs: dict, observations: dict) -> None:
    lidar = obs["lidar"]
    check_in_bounds("lidar min (m)", float(lidar.min()), observations["lidar"]["bounds"])
    check_in_bounds("lidar max (m)", float(lidar.max()), observations["lidar"]["bounds"])
    for key in ("velocity", "steering", "progress"):
        check_in_bounds(
            f"{key} ({observations[key]['unit']})", obs[key], observations[key]["bounds"]
        )


def race(sim, agents: list, track: str, seed: int, steps: int) -> list[tuple]:
    """One race in the order of sim_recorder.py; returns (car, obs, info, action) per call.

    Every car still racing, or out of the race, gets predict() each step, all with the
    same info object, as in the official recorder.
    """
    space = sim.get_space_info()
    dry_obs, dry_info = schema(agent_loader.create_dummy_obs()), agent_loader.create_dummy_info()
    sim.set_map(track)
    sim.reset(num_cars=len(agents), seed=seed)
    calls = []
    for _ in range(steps):
        info = sim.get_step_info()
        if schema(info) != schema(dry_info):
            raise CheckError(f"race info differs from the dry-run schema: {schema(info)}")
        info_before = copy.deepcopy(info)
        for car, agent in enumerate(agents):
            if info["agent_status"][car] == STATUS_FINISHED:
                continue
            obs = sim.get_obs(car)
            if schema(obs) != dry_obs:
                raise CheckError(f"race observation differs from the dry-run: {schema(obs)}")
            check_observation(obs, space["observations"])
            obs_before = copy.deepcopy(obs)
            action = check_action(agent.predict(obs, info), space["actions"])
            if not same(obs, obs_before):
                raise CheckError(f"car {car}: predict() changed its observation")
            calls.append((car, obs_before, info_before, action))
            sim.apply_action(car, *action)
        if not same(info, info_before):
            raise CheckError("predict() changed the shared info dict")
        sim.simulation_step()
        if all(status != 1 for status in sim.get_step_info()["agent_status"].values()):
            break
    return calls


def check_replay(new_agent, calls: list[tuple], car: int) -> None:
    """A fresh instance fed only the calls of `car` reproduces its actions exactly.

    It got no dry-run either: the dry-run must leave no trace in the memory.
    """
    agent = new_agent()
    for caller, obs, info, action in calls:
        if caller == car and agent.predict(obs, info) != action:
            raise CheckError(f"car {car}: memory depends on other cars, instances or dry-run")


def summary(sim, names: list[str], calls: list[tuple]) -> str:
    info = sim.get_step_info()
    status = {0: "DNF", 1: "racing", 2: "finished"}
    return (
        ", ".join(
            f"{name} {status.get(info['agent_status'][car])} {info['max_progress'][car]:.2f} lap"
            for car, name in enumerate(names)
        )
        + f" ({len(calls)} predict calls)"
    )


def run(submission: Path, workdir: Path, track: str, seed: int, steps: int) -> None:
    grand_prix = workdir / "submission"
    sources = build_grand_prix(submission, grand_prix)
    for driver, source in sources.items():
        note = "" if driver == source else f" (substitute: copy of {source}, {driver} missing)"
        print(f"{driver}: {source}{note}")
    for source in sorted(set(sources.values())):
        check_export_record(submission / source)
        print(f"{source}: delivery files match export.json")
    for driver in DRIVERS:
        count = check_external_data(grand_prix / driver)
        print(f"{driver}: {count} weights inside model.onnx.data")

    cwd = workdir / "cwd"
    cwd.mkdir()
    with contextlib.chdir(cwd):  # any cwd but the agent folders
        race_in_one_process(grand_prix, sources, cwd, track, seed, steps)
    imported = [name for name in TRAINING_PACKAGES if name in sys.modules]
    if imported:
        raise CheckError(f"training packages in the tournament process: {imported}")


def race_in_one_process(grand_prix, sources, cwd, track, seed, steps) -> None:
    with tournament_imports():
        # Grand Prix loader: dry-run of each agent, module ids 0..3.
        loaded = agent_loader.discover_and_load_submissions(grand_prix, max_teams=len(DRIVERS))
        # Load each agent again with the same module name as its first load.
        twins = {
            source: agent_loader.validate_and_load_agent(
                grand_prix / source, team_name=source, unique_id=DRIVERS.index(source)
            )[1]
            for source in sorted(set(sources.values()))
        }
    names = [name for name, _ in loaded]
    print(f"loaded in one process from {cwd}: {', '.join(names)}, dry-runs passed")
    print(f"loaded again under the same module names: {', '.join(twins)}")

    import env_simulation as sim  # after the guard: the simulator is not in the agent world

    try:
        agents = [agent for _, agent in loaded]
        calls = race(sim, agents, track, seed, steps)
        print(f"race {track} seed {seed}: {summary(sim, names, calls)}")
        for car, agent in enumerate(agents):
            check_replay(type(agent), calls, car)
        print("each instance replays its car alone: no shared state between instances")
        for source, twin in twins.items():
            shared = [twin] * len(DRIVERS)
            calls = race(sim, shared, track, seed + 1, steps)
            print(
                f"race {track} seed {seed + 1}, one {source} instance for 4 cars: "
                f"{summary(sim, [source] * len(DRIVERS), calls)}"
            )
            for car in range(len(DRIVERS)):
                check_replay(type(agents[DRIVERS.index(source)]), calls, car)
            print(f"{source}: each car replays alone: memory kept per car")
    finally:
        sim.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--track", default="Austin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600, help="decisions per race (20 Hz)")
    parser.add_argument("--grand-prix-copy", type=Path, help="write the four checked folders there")
    args = parser.parse_args()
    if args.grand_prix_copy and args.grand_prix_copy.exists():
        parser.error("--grand-prix-copy already exists")
    submission = args.submission.resolve()
    copy_to = args.grand_prix_copy and args.grand_prix_copy.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run(submission, Path(tmp), args.track, args.seed, args.steps)
        except Exception as error:  # the loader raises plain exceptions too
            print(f"FAIL: {error}")
            return 1
        if copy_to:
            shutil.copytree(Path(tmp) / "submission", copy_to)
    print("OK: four drivers share one process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
