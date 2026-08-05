"""Compute team-level red zone trip/play rates and join them into player_week."""

from pathlib import Path

import pandas as pd

REDZONE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "redzone_plays.parquet"
PLAYER_WEEK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_week.parquet"
TEAM_WEEK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "team_week.parquet"

GROUP_KEYS = ["posteam", "season", "week"]


def build_team_week(redzone: pd.DataFrame) -> pd.DataFrame:
    plays = redzone.groupby(GROUP_KEYS).agg(
        games=("game_id", "nunique"),
        redzone_plays=("game_id", "size"),
    )

    with_drive = redzone.dropna(subset=["drive"]).copy()
    with_drive["game_drive"] = with_drive["game_id"] + "_" + with_drive["drive"].astype(int).astype(str)
    trips = with_drive.groupby(GROUP_KEYS)["game_drive"].nunique().rename("redzone_trips")

    team_week = plays.join(trips, how="left").reset_index()
    team_week["redzone_trips"] = team_week["redzone_trips"].fillna(0).astype(int)

    team_week["redzone_trips_per_game"] = team_week["redzone_trips"] / team_week["games"]
    team_week["redzone_plays_per_game"] = team_week["redzone_plays"] / team_week["games"]

    return team_week


def main() -> None:
    redzone = pd.read_parquet(REDZONE_PATH)
    print(f"Redzone plays in: {len(redzone)}")
    dropped = redzone["drive"].isna().sum()
    if dropped:
        print(f"Skipping {dropped} plays with no drive id when counting trips (still counted as plays).")

    team_week = build_team_week(redzone)
    print(f"Team-week rows: {len(team_week)}")
    assert (team_week["games"] == 1).all(), "expected exactly one game per team/season/week"

    TEAM_WEEK_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_week.to_parquet(TEAM_WEEK_PATH, index=False)
    print(f"Saved {len(team_week)} rows to {TEAM_WEEK_PATH}")

    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    print(f"Player-week rows before join: {len(player_week)}")

    team_cols = GROUP_KEYS + ["redzone_trips", "redzone_plays", "redzone_trips_per_game", "redzone_plays_per_game"]
    player_week = player_week.merge(team_week[team_cols], on=GROUP_KEYS, how="left", suffixes=("", "_team"))
    print(f"Player-week rows after join: {len(player_week)}")

    player_week.to_parquet(PLAYER_WEEK_PATH, index=False)
    print(f"Saved {len(player_week)} rows to {PLAYER_WEEK_PATH}")


if __name__ == "__main__":
    main()
