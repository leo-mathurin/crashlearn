from crashlearn import SIM_DIR, add_simulator_to_path


def test_vendored_simulator_is_importable():
    assert (SIM_DIR / "env_simulation.py").is_file()
    add_simulator_to_path()
    import env_simulation

    assert "Austin" in env_simulation.get_available_maps()
