"""setup command: install [docking] extras + fetch the AutoDock Vina binary (offline-testable)."""

from ddswarm import cli, setup


def test_asset_for_platforms():
    assert setup._asset_for("linux", "x86_64") == "vina_1.2.7_linux_x86_64"
    assert setup._asset_for("linux", "amd64") == "vina_1.2.7_linux_x86_64"
    assert setup._asset_for("linux", "aarch64") == "vina_1.2.7_linux_aarch64"
    assert setup._asset_for("darwin", "x86_64") == "vina_1.2.7_mac_x86_64"
    assert setup._asset_for("darwin", "arm64") == "vina_1.2.7_mac_aarch64"
    assert setup._asset_for("win32", "AMD64") == "vina_1.2.7_win.exe"
    assert setup._asset_for("freebsd", "x86_64") == ""


def test_setup_nothing_to_do(monkeypatch, capsys):
    monkeypatch.setattr(setup, "_needs_deps", lambda: False)
    monkeypatch.setattr(setup, "_needs_vina", lambda: False)
    rc = setup.run_setup()
    assert rc == 0
    assert "nothing to do" in capsys.readouterr().out


def test_setup_dry_run(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(setup, "_needs_deps", lambda: True)
    monkeypatch.setattr(setup, "_needs_vina", lambda: True)
    monkeypatch.setattr(setup, "_vina_asset", lambda: "vina_1.2.7_linux_x86_64")
    monkeypatch.setattr(setup, "_pip_install_docking", lambda: calls.append("pip") or (0, "ok"))
    monkeypatch.setattr(setup, "_download_vina", lambda *a: calls.append("download") or (True, "wrote"))

    rc = setup.run_setup(dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "install [docking] extras" in out
    assert "download AutoDock Vina 1.2.7" in out
    assert "dry run" in out
    assert calls == []  # dry run makes no changes


def test_setup_proceeds(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(setup, "_needs_deps", lambda: True)
    monkeypatch.setattr(setup, "_needs_vina", lambda: True)
    monkeypatch.setattr(setup, "_vina_asset", lambda: "vina_1.2.7_linux_x86_64")
    monkeypatch.setattr(setup, "_pip_install_docking", lambda: calls.append("pip") or (0, "installed"))
    monkeypatch.setattr(setup, "_download_vina", lambda *a: calls.append("download") or (True, "wrote /tmp/vina"))
    monkeypatch.setattr("builtins.input", lambda *a: "")

    rc = setup.run_setup()
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == ["pip", "download"]
    assert "installing [docking] extras" in out
    assert "downloading AutoDock Vina 1.2.7" in out


def test_setup_abort(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(setup, "_needs_deps", lambda: True)
    monkeypatch.setattr(setup, "_needs_vina", lambda: True)
    monkeypatch.setattr(setup, "_vina_asset", lambda: "vina_1.2.7_linux_x86_64")
    monkeypatch.setattr(setup, "_pip_install_docking", lambda: calls.append("pip") or (0, "ok"))
    monkeypatch.setattr(setup, "_download_vina", lambda *a: calls.append("download") or (True, "wrote"))
    monkeypatch.setattr("builtins.input", lambda *a: "no")

    rc = setup.run_setup()
    out = capsys.readouterr().out
    assert rc == 1
    assert "aborted" in out
    assert calls == []  # nothing ran


def test_setup_cli(monkeypatch, capsys):
    monkeypatch.setattr(setup, "_needs_deps", lambda: False)
    monkeypatch.setattr(setup, "_needs_vina", lambda: False)
    rc = cli.main(["setup"])
    assert rc == 0
    assert "ddswarm setup" in capsys.readouterr().out
