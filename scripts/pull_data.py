"""Download NFL play-by-play data and save it as a partitioned parquet dataset."""

from pathlib import Path

import nfl_data_py as nfl
import pyarrow as pa
import pyarrow.parquet as pq

SEASONS = list(range(2020, 2025))
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pbp"


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.parquet")):
        print(f"Data already exists at {OUTPUT_DIR}, skipping download.")
        return

    print(f"Downloading play-by-play data for seasons {SEASONS[0]}-{SEASONS[-1]}...")
    pbp = nfl.import_pbp_data(SEASONS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(pbp)
    pq.write_to_dataset(table, root_path=str(OUTPUT_DIR), partition_cols=["season"])

    print(f"Saved {len(pbp)} rows to {OUTPUT_DIR}, partitioned by season.")


if __name__ == "__main__":
    main()
