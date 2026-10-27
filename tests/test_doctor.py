"""doctor command: offline environment check (deterministic, no network by default)."""

from ddswarm import cli, doctor

_ALL_PREP = {"meeko": "1.0", "scipy": "1.0", "gemmi": "1.0", "pdbfixer": "1.0", "openbabel-wheel": "1.0"}


def test_doctor_offline_reports_gaps(monkeypatch):
    monkeypatch.setattr(doctor, "_vina_status", lambda: (False, "not found"))
    monkeypatch.setattr(doctor, "_docking_prep", lambda: {**_ALL_PREP, "meeko": "missing"})
    monkeypatch.setattr(doctor, "_obabel_status", lambda: (False, "/none/obabel"))
    monkeypatch.setattr(doctor, "_openmm_status", lambda: (False, "unknown"))
    monkeypatch.setattr(doctor, "_hermes_status", lambda: (False, ""))
    monkeypatch.setattr(doctor, "_pdb_cache_status", lambda: ("/tmp/cache", 0))

    report = doctor.run_doctor()  # network defaults to False

    assert "python" in report
    assert "core deps" in report
    assert "vina binary    : MISSING" in report
    assert "docking prep   : MISSING meeko" in report
    assert "obabel binary  : MISSING" in report
    assert "openmm         : MISSING" in report
    assert "hermes NOT FOUND" in report
    assert "pdb cache" in report
    assert "network" not in report  # offline by default
    assert "summary" in report
    assert "docking/selectivity" in report
    assert "docking prep" in report
    assert "receptor MD" in report
    assert "reasoning (knowledge agents)" in report


def test_doctor_all_ok(monkeypatch):
    monkeypatch.setattr(doctor, "_vina_status", lambda: (True, "AutoDock Vina v1.2.7"))
    monkeypatch.setattr(doctor, "_docking_prep", lambda: dict(_ALL_PREP))
    monkeypatch.setattr(doctor, "_obabel_status", lambda: (True, "/venv/bin/obabel"))
    monkeypatch.setattr(doctor, "_openmm_status", lambda: (True, "8.6.1"))
    monkeypatch.setattr(doctor, "_hermes_status", lambda: (True, "/usr/bin/hermes"))
    monkeypatch.setattr(doctor, "_pdb_cache_status", lambda: ("/tmp/cache", 3))

    report = doctor.run_doctor()

    assert "vina binary    : OK" in report
    assert "openmm         : OK" in report
    assert "reasoning      : OK" in report
    assert "all compute paths available" in report
    assert "MISSING" not in report


def test_doctor_network_is_opt_in(monkeypatch):
    monkeypatch.setattr(doctor, "_vina_status", lambda: (True, "v"))
    monkeypatch.setattr(doctor, "_docking_prep", lambda: dict(_ALL_PREP))
    monkeypatch.setattr(doctor, "_obabel_status", lambda: (True, "/o"))
    monkeypatch.setattr(doctor, "_openmm_status", lambda: (True, "8"))
    monkeypatch.setattr(doctor, "_hermes_status", lambda: (True, "/h"))
    monkeypatch.setattr(doctor, "_pdb_cache_status", lambda: ("/tmp/cache", 0))

    monkeypatch.setattr(doctor, "_network_status", lambda: (True, ""))
    assert "network (RCSB) : reachable" in doctor.run_doctor(network=True)

    monkeypatch.setattr(doctor, "_network_status", lambda: (False, "timeout"))
    report = doctor.run_doctor(network=True)
    assert "network (RCSB) : unreachable (timeout)" in report


def test_doctor_cli(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_vina_status", lambda: (True, "v"))
    monkeypatch.setattr(doctor, "_docking_prep", lambda: dict(_ALL_PREP))
    monkeypatch.setattr(doctor, "_obabel_status", lambda: (True, "/o"))
    monkeypatch.setattr(doctor, "_openmm_status", lambda: (True, "8"))
    monkeypatch.setattr(doctor, "_hermes_status", lambda: (True, "/h"))
    monkeypatch.setattr(doctor, "_pdb_cache_status", lambda: ("/tmp/cache", 0))

    rc = cli.main(["doctor"])
    assert rc == 0
    assert "ddswarm environment check" in capsys.readouterr().out
