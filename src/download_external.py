"""Download the external (other-games) datasets used for training and validation.

Nothing here touches the competition match. Two public datasets, both about OTHER games:

  * IDSSE / DFL open dataset -- 7 German Bundesliga matches (2022/23) with official 25 Hz
    position data. The ball FrameSet carries per-frame ``BallStatus`` (dead/alive) and
    ``BallPossession`` (home/guest) annotated by the league's data provider. CC-BY 4.0.
    Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event
    data in elite soccer", Scientific Data. Mirror: huggingface.co/datasets/pysport/idsse-data
  * Metrica Sports sample data -- 3 anonymised matches with 25 Hz tracking and human event
    logs. github.com/metrica-sports/sample-data

Usage:
    python src/download_external.py            # everything (about 3 GB)
    python src/download_external.py --only metrica
"""

import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")

DFL_BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
DFL_MATCHES = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
DFL_FILES = (
    "DFL_02_01_matchinformation_{comp}_DFL-MAT-{mid}.xml",
    "DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{mid}.xml",
)

METRICA_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
METRICA_FILES = (
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
)


def jobs(only=None):
    out = []
    if only in (None, "metrica"):
        for rel in METRICA_FILES:
            out.append((METRICA_BASE + rel, os.path.join(EXTERNAL_DIR, "metrica", *rel.split("/"))))
    if only in (None, "dfl"):
        for mid, comp in DFL_MATCHES.items():
            for template in DFL_FILES:
                name = template.format(comp=comp, mid=mid)
                out.append((DFL_BASE + name, os.path.join(EXTERNAL_DIR, "dfl", name)))
    return out


def fetch(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "have  " + os.path.basename(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, path)
    return "got   {} ({:.0f} MB)".format(os.path.basename(path), os.path.getsize(path) / 1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("metrica", "dfl"), default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(fetch, jobs(args.only)):
            print(message)
            sys.stdout.flush()
    print("done ->", EXTERNAL_DIR)
