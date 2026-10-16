import json, zipfile, importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("download", Path(__file__).parents[1] / "data" / "download.py")
dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "src"; src.mkdir()
    zp = src / "x.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("some_other_name.csv", "a,b,class\n1,2,UP\n3,4,DOWN\n")     # name differs from `main`
    (src / "f.npz").write_bytes(b"npzdata")
    raw = tmp_path / "raw"
    monkeypatch.setattr(dl, "RAW", raw); monkeypatch.setattr(dl, "MANIFEST", raw / "MANIFEST.json")
    monkeypatch.setattr(dl, "ROOT", tmp_path)
    monkeypatch.setattr(dl, "SOURCES", {
        "z": dict(url=zp.as_uri(), kind="zip", dest="zz", main="data_main.csv", used_by="t"),
        "f": dict(url=(src / "f.npz").as_uri(), kind="file", dest="ad", main="f.npz", used_by="t")})
    return raw


def test_download_manifest_verify(env):
    dl.get("z"); dl.get("f")
    assert (env / "zz" / "data_main.csv").exists()               # renamed to expected name
    assert (env / "ad" / "f.npz").read_bytes() == b"npzdata"
    m = json.loads((env / "MANIFEST.json").read_text()); assert set(m) == {"z", "f"}
    assert dl.verify() == 0
    dl.get("f")                                                  # idempotent skip
    (env / "ad" / "f.npz").write_bytes(b"tampered")
    assert dl.verify() == 1                                      # hash mismatch detected


def test_zip_slip_blocked(tmp_path):
    zp = tmp_path / "bad.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        dl.safe_extract(zp, tmp_path / "out")
