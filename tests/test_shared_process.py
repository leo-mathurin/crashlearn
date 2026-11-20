"""Self-tests of scripts/check_shared_process.py. The full check runs in CI (Training job)."""

import importlib.util
import shutil

import env_simulation
import numpy as np
import pytest
from conftest import ROOT, SUBMISSION_DIRS

spec = importlib.util.spec_from_file_location(
    "check_shared_process", ROOT / "scripts" / "check_shared_process.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

ACTIONS = env_simulation.get_space_info()["actions"]


def message(field: int, payload: bytes) -> bytes:
    """One length-delimited protobuf field (payloads here stay under 128 bytes)."""
    return bytes([field << 3 | 2, len(payload)]) + payload


def model(*initializers: bytes) -> bytes:
    return message(check.MODEL_GRAPH, b"".join(message(5, i) for i in initializers))


def external(location: str, offset: int = 0, length: int = 4) -> bytes:
    entries = {"location": location, "offset": str(offset), "length": str(length)}
    data = b"".join(
        message(13, message(1, k.encode()) + message(2, v.encode())) for k, v in entries.items()
    )
    return data + bytes([check.TENSOR_DATA_LOCATION << 3, check.LOCATION_EXTERNAL])


@pytest.fixture
def driver(tmp_path):
    if not SUBMISSION_DIRS:
        pytest.skip("no submission/<driver>/ yet")
    return shutil.copytree(SUBMISSION_DIRS[0], tmp_path / "driver1")


def test_delivered_weights_lie_in_model_onnx_data():
    if not SUBMISSION_DIRS:
        pytest.skip("no submission/<driver>/ yet")
    for folder in SUBMISSION_DIRS:
        assert check.check_external_data(folder) > 0
        check.check_export_record(folder)


@pytest.mark.parametrize(
    ("onnx_bytes", "error"),
    [
        (model(external("model.onnx.data")), None),
        (model(b""), "does not read"),
        (model(external("../model.onnx.data")), "weights read from"),
        (model(external("model.onnx.data", offset=8)), "outside"),
        (model(external("model.onnx.data"))[:-3], "not a readable"),
    ],
)
def test_external_data_check(tmp_path, onnx_bytes, error):
    (tmp_path / "model.onnx").write_bytes(onnx_bytes)
    (tmp_path / "model.onnx.data").write_bytes(b"\0" * 8)
    if error is None:
        assert check.check_external_data(tmp_path) == 1
    else:
        with pytest.raises(check.CheckError, match=error):
            check.check_external_data(tmp_path)


def test_missing_data_file_fails(driver):
    (driver / "model.onnx.data").unlink()
    with pytest.raises(check.CheckError, match="missing"):
        check.check_external_data(driver)


def test_changed_file_fails_its_export_record(driver):
    with (driver / "model.onnx.data").open("ab") as data:
        data.write(b"\0")
    with pytest.raises(check.CheckError, match="does not match export.json"):
        check.check_export_record(driver)


def test_missing_drivers_are_copies_of_existing_ones(tmp_path):
    for name in ("driver1", "driver2"):
        (tmp_path / "in" / name).mkdir(parents=True)
        for file in (*check.DELIVERY_FILES, "export.json"):
            (tmp_path / "in" / name / file).write_text(name)
    sources = check.build_grand_prix(tmp_path / "in", tmp_path / "out")
    assert sources == {
        "driver1": "driver1",
        "driver2": "driver2",
        "driver3": "driver1",
        "driver4": "driver2",
    }
    for driver, source in sources.items():
        copied = sorted(p.name for p in (tmp_path / "out" / driver).iterdir())
        assert copied == sorted(check.DELIVERY_FILES)  # only what the subject delivers
        assert (tmp_path / "out" / driver / "agent.py").read_text() == source


def test_only_tournament_packages_can_be_imported():
    finder = check._TournamentImports()
    for allowed in ("os", "json", "numpy.linalg", "onnxruntime"):
        assert finder.find_spec(allowed) is None
    with check.tournament_imports(), pytest.raises(ImportError, match="not in the tournament"):
        importlib.import_module("torch_or_any_training_package")


class PerCarMemory:
    def __init__(self):
        self.steps = {}

    def predict(self, obs, info):
        car = obs["agent_id"]
        self.steps[car] = self.steps.get(car, 0) + 1
        return float(self.steps[car]), 0.0


class SharedMemory(PerCarMemory):
    def predict(self, obs, info):
        return super().predict({"agent_id": 0}, info)


def calls_of(agent, cars=2, steps=3):
    return [
        (car, {"agent_id": car}, {}, agent.predict({"agent_id": car}, {}))
        for _ in range(steps)
        for car in range(cars)
    ]


def test_replay_accepts_memory_kept_per_car():
    calls = calls_of(PerCarMemory())
    for car in (0, 1):
        check.check_replay(PerCarMemory, calls, car)


def test_replay_detects_memory_shared_by_cars():
    with pytest.raises(check.CheckError, match="memory depends"):
        check.check_replay(SharedMemory, calls_of(SharedMemory()), 1)


@pytest.mark.parametrize(
    "action",
    [[1.0, 0.0], (np.float32(1.0), 0.0), (1, 0.0), (float("nan"), 0.0), (10.5, 0.0), (1.0, 0.5)],
)
def test_action_check_rejects(action):
    with pytest.raises(check.CheckError):
        check.check_action(action, ACTIONS)


def test_action_check_accepts_physical_units():
    assert check.check_action((10.0, -0.4189), ACTIONS) == (10.0, -0.4189)


def test_same_sees_changed_inputs():
    obs = {"lidar": np.ones(3, dtype=np.float32), "opponents": {0: {"x": 1.0}}}
    changed = {"lidar": np.ones(3, dtype=np.float32), "opponents": {0: {"x": 2.0}}}
    assert check.same(obs, {**obs})
    assert not check.same(obs, changed)
    assert not check.same({"v": 1.0}, {"v": np.float64(1.0)})
