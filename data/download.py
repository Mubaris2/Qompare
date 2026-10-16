"""Download every raw dataset into data/raw/ (inside the project, nothing in ~/river_data).

  python data/download.py                 # fetch everything missing
  python data/download.py --only elec2    # fetch a subset
  python data/download.py --force         # re-download
  python data/download.py --verify        # check files against data/raw/MANIFEST.json
  python data/download.py --list

MANIFEST.json (sha256 per file) is committed to git. If the upstream file ever changes,
--verify on the other member's machine will flag it, so both of you use identical data.
If a download fails (blocked network, site down), the script prints the URL and the path
where you can place the file by hand; re-run it once to record the hash.
"""
import argparse, hashlib, json, shutil, sys, time, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ADB = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"

SOURCES = {
    "elec2": dict(url="https://maxhalford.github.io/files/datasets/electricity.zip", kind="zip",
                  dest="elec2", main="electricity.csv", used_by="T3, P1 reproduction"),
    "seoul_bike": dict(url="https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip",
                       kind="zip", dest="seoul_bike", main="SeoulBikeData.csv", used_by="T1"),
    "ad_shuttle": dict(url=ADB + "32_shuttle.npz", kind="file", dest="ad", main="32_shuttle.npz",
                       used_by="T2 (49,097 x 9, 7.2% anomalies)"),
    "ad_mammography": dict(url=ADB + "23_mammography.npz", kind="file", dest="ad", main="23_mammography.npz",
                           used_by="T2 (11,183 x 6, 2.3% anomalies)"),
    "ad_cardio": dict(url=ADB + "6_cardio.npz", kind="file", dest="ad", main="6_cardio.npz",
                      used_by="T2 (1,831 x 21, 9.6% anomalies, small)"),
}
MANIFEST = RAW / "MANIFEST.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, retries=3):
    """Download url to dest atomically. Browser-like UA because some hosts block urllib's default."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-project)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            part.replace(dest)
            return
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed after {retries} tries: {last}")


def safe_extract(zpath, out):
    out = out.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if not (out / m).resolve().is_relative_to(out):
                raise RuntimeError(f"unsafe path in zip: {m}")
        z.extractall(out)


def ensure_main(dest, main):
    """Make sure dest/main exists. If the zip used another csv name, copy the best candidate."""
    if (dest / main).exists():
        return True
    csvs = sorted(dest.rglob("*.csv"))
    stem = Path(main).stem.lower()
    pick = [c for c in csvs if stem[:5] in c.name.lower()] or (csvs if len(csvs) == 1 else [])
    if pick:
        shutil.copy(pick[0], dest / main)
        print(f"   note: copied {pick[0].name} -> {main}")
        return True
    return False


def hash_tree(dest, only_main=None):
    files = [dest / only_main] if only_main else [p for p in dest.rglob("*") if p.is_file() and p.suffix != ".part"]
    return {p.relative_to(RAW).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()}


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m):
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def get(name, force=False):
    s = SOURCES[name]
    dest = RAW / s["dest"]
    main = dest / s["main"]
    man = load_manifest()
    if main.exists() and not force:
        print(f"[skip] {name}: {main.relative_to(ROOT)} already present")
    else:
        print(f"[get ] {name}: {s['url']}")
        if s["kind"] == "zip":
            z = dest / (name + ".zip")
            fetch(s["url"], z)
            safe_extract(z, dest)
            z.unlink()
        else:
            fetch(s["url"], main)
        if not ensure_main(dest, s["main"]):
            raise RuntimeError(f"{s['main']} not found after download; place it in {dest}")
    files = hash_tree(dest, None if s["kind"] == "zip" else s["main"])
    old = man.get(name, {}).get("files")
    if old and old != files:
        print(f"   WARNING: {name} differs from MANIFEST.json (upstream changed or file edited)."
              + (" Manifest updated (--force)." if force else " Manifest kept; fix the file or re-run with --force."))
    if force or not old:
        man[name] = {"url": s["url"], "used_by": s["used_by"], "files": files,
                     "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_manifest(man)


def verify():
    man, bad = load_manifest(), 0
    if not man:
        print("no MANIFEST.json yet"); return 1
    for name, rec in man.items():
        for rel, info in rec["files"].items():
            p = RAW / rel
            if not p.exists():
                print(f"[MISSING] {rel}"); bad += 1
            elif sha256(p) != info["sha256"]:
                print(f"[CHANGED] {rel}"); bad += 1
            else:
                print(f"[ok     ] {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SOURCES)); ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, s in SOURCES.items():
            print(f"{n:16s} -> data/raw/{s['dest']}/{s['main']:22s} {s['used_by']}")
        sys.exit(0)
    if a.verify:
        sys.exit(verify())
    failed = []
    for n in (a.only or SOURCES):
        try:
            get(n, a.force)
        except Exception as e:
            s = SOURCES[n]
            print(f"[FAIL] {n}: {e}\n       manual: download {s['url']}\n       place at: data/raw/{s['dest']}/{s['main']}, then re-run to record its hash")
            failed.append(n)
    print("done." if not failed else f"done with failures: {failed}")
    sys.exit(1 if failed else 0)
