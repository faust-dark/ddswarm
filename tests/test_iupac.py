"""Tests for the opt-in IUPAC naming module (NIH CACTUS, network — mocked offline)."""

from ddswarm.chemistry.iupac import fetch_iupac_name


class _FakeResp:
    def __init__(self, text: str):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._text.encode()


def test_fetch_iupac_name_ok(monkeypatch):
    captured = {}

    def fake_open(req, timeout):
        captured["url"] = req.full_url
        return _FakeResp("8-chloro-1-methyl-6-phenyl-[1,2,4]triazolo[4,3-a][1,4]benzodiazepine")

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    name = fetch_iupac_name("Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2")
    assert name == "8-chloro-1-methyl-6-phenyl-[1,2,4]triazolo[4,3-a][1,4]benzodiazepine"
    assert captured["url"].endswith("/iupac_name")


def test_fetch_iupac_name_error_returns_none(monkeypatch):
    def fake_open(url, timeout):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    assert fetch_iupac_name("c1ccccc1") is None


def test_fetch_iupac_name_empty_returns_none(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: _FakeResp("  "))
    assert fetch_iupac_name("c1ccccc1") is None


def test_fetch_iupac_name_pubchem_fallback(monkeypatch):
    def fake_open(req, timeout):
        url = req.full_url
        if "cactus" in url:
            return _FakeResp("")  # CACTUS cannot name it
        if "pubchem" in url:
            return _FakeResp('{"PropertyTable":{"Properties":[{"IUPACName":"benzene"}]}}')
        raise OSError(f"unexpected url {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    assert fetch_iupac_name("c1ccccc1") == "benzene"
