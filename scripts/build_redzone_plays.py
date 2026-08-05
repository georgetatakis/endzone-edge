"""Filter play-by-play data down to red zone plays."""

from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pbp"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "redzone_plays.parquet"

COLUMNS = [
    "season",
    "week",
    "game_id",
    "drive",
    "yardline_100",
    "receiver_player_name",
    "rusher_player_name",
    "play_type",
    "posteam",
    "touchdown",
]


def main() -> None:
    pbp = pd.read_parquet(INPUT_DIR, columns=COLUMNS)
    print(f"Rows before filtering: {len(pbp)}")

    redzone = pbp[pbp["yardline_100"] <= 20].copy()
    redzone["goal_line"] = redzone["yardline_100"] <= 5
    redzone["touchdown"] = redzone["touchdown"].astype(bool)
    redzone["season"] = redzone["season"].astype(int)

    print(f"Rows after filtering: {len(redzone)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    redzone.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(redzone)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
