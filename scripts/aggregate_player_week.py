"""Aggregate red zone plays into per-player, per-week red zone usage stats."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "redzone_plays.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_week.parquet"

GROUP_KEYS = ["player", "posteam", "season", "week"]


def main() -> None:
    redzone = pd.read_parquet(INPUT_PATH)
    print(f"Rows in: {len(redzone)}")

    targets = redzone[redzone["receiver_player_name"].notna()].copy()
    targets["player"] = targets["receiver_player_name"]
    targets = (
        targets.groupby(GROUP_KEYS)
        .agg(
            redzone_targets=("player", "size"),
            goal_line_targets=("goal_line", "sum"),
            receiving_tds=("touchdown", "sum"),
        )
        .reset_index()
    )

    carries = redzone[redzone["rusher_player_name"].notna()].copy()
    carries["player"] = carries["rusher_player_name"]
    carries = (
        carries.groupby(GROUP_KEYS)
        .agg(
            redzone_carries=("player", "size"),
            goal_line_carries=("goal_line", "sum"),
            rushing_tds=("touchdown", "sum"),
        )
        .reset_index()
    )

    player_week = targets.merge(carries, on=GROUP_KEYS, how="outer")

    count_cols = [
        "redzone_targets",
        "goal_line_targets",
        "receiving_tds",
        "redzone_carries",
        "goal_line_carries",
        "rushing_tds",
    ]
    player_week[count_cols] = player_week[count_cols].fillna(0).astype(int)

    player_week["redzone_touches"] = player_week["redzone_targets"] + player_week["redzone_carries"]
    player_week["goal_line_touches"] = player_week["goal_line_targets"] + player_week["goal_line_carries"]
    player_week["redzone_tds"] = player_week["receiving_tds"] + player_week["rushing_tds"]
    player_week = player_week.drop(
        columns=["goal_line_targets", "goal_line_carries", "receiving_tds", "rushing_tds"]
    )

    team_week_targets = (
        targets.groupby(["posteam", "season", "week"])["redzone_targets"]
        .sum()
        .reset_index(name="team_redzone_targets")
    )
    player_week = player_week.merge(team_week_targets, on=["posteam", "season", "week"], how="left")
    player_week["redzone_target_share"] = (
        player_week["redzone_targets"] / player_week["team_redzone_targets"]
    ).fillna(0)

    print(f"Rows out: {len(player_week)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    player_week.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(player_week)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
