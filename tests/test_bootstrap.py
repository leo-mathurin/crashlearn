from crashlearn import SIM_DIR, add_simulator_to_path


def test_vendored_simulator_is_importable():
    assert (SIM_DIR / "env_simulation.py").is_file()
    add_simulator_to_path()
    import env_simulation

    assert "Austin" in env_simulation.get_available_maps()


def test_provenance_records_archive_hash():
    provenance = (SIM_DIR.parent / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "7110e6580d267d6bdfe6d6530b08b93e9c01679100af16bfa65896330ef8ba2e" in provenance
