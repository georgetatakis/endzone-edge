"""Split-half check: does weeks 1-8 red zone usage predict weeks 9+ touchdown production?

redzone_target_share is averaged (not summed) across a player's weeks 1-8 -
it's a rate, not a count, and this matches how src/ranking.py aggregates it
elsewhere in the pipeline. Back-half touchdown totals are matched by player
name only (not posteam), so a mid-season trade doesn't drop a player from
the weeks 9+ count.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ground_truth import load_full_game_tds  # noqa: E402

PLAYER_WEEK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_week.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "split_half_2025.csv"

SEASON = 2025
FIRST_HALF_WEEKS = range(1, 9)
TOP_N = 20
RANDOM_SEED = 42

DISPLAY_COLS = [
    "rank",
    "player",
    "posteam",
    "redzone_touches",
    "redzone_target_share",
    "second_half_tds",
    "scored_back_half",
]


def build_first_half(player_week: pd.DataFrame) -> pd.DataFrame:
    first_half = player_week[
        (player_week["season"] == SEASON) & (player_week["week"].isin(FIRST_HALF_WEEKS))
    ]
    return (
        first_half.groupby(["player", "posteam"])
        .agg(
            redzone_targets=("redzone_targets", "sum"),
            redzone_carries=("redzone_carries", "sum"),
            redzone_touches=("redzone_touches", "sum"),
            redzone_target_share=("redzone_target_share", "mean"),
        )
        .reset_index()
    )


def build_second_half_tds(full_game_tds: pd.DataFrame, max_week: int) -> pd.DataFrame:
    second_half = full_game_tds[
        (full_game_tds["season"] == SEASON) & (full_game_tds["week"] >= 9) & (full_game_tds["week"] <= max_week)
    ]
    return second_half.groupby("player")["full_game_tds"].sum().rename("second_half_tds").reset_index()


def rank_and_attach(first_half: pd.DataFrame, second_half_tds: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    ranked = first_half.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranked = ranked.merge(second_half_tds, on="player", how="left")
    ranked["second_half_tds"] = ranked["second_half_tds"].fillna(0).astype(int)
    ranked["scored_back_half"] = ranked["second_half_tds"] > 0
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def summarize(label: str, top20: pd.DataFrame, control: pd.DataFrame) -> None:
    top20_hit_rate = top20["scored_back_half"].mean() * 100
    top20_avg_tds = top20["second_half_tds"].mean()
    control_hit_rate = control["scored_back_half"].mean() * 100
    control_avg_tds = control["second_half_tds"].mean()

    print(f"\n[{label}]")
    print(f"  Top {TOP_N}:    {top20_hit_rate:.1f}% scored >=1 back-half TD, avg {top20_avg_tds:.2f} TDs")
    print(f"  Control (n={len(control)}): {control_hit_rate:.1f}% scored >=1 back-half TD, avg {control_avg_tds:.2f} TDs")


def main() -> None:
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    full_game_tds = load_full_game_tds()

    max_week = int(player_week.loc[player_week["season"] == SEASON, "week"].max())
    print(f"2025 season: first half = weeks 1-8, back half = weeks 9-{max_week}")

    first_half = build_first_half(player_week)
    second_half_tds = build_second_half_tds(full_game_tds, max_week)

    all_rows = []
    for view, sort_col in [("touches", "redzone_touches"), ("share", "redzone_target_share")]:
        ranked = rank_and_attach(first_half, second_half_tds, sort_col)

        top20 = ranked.head(TOP_N).copy()
        rest = ranked.iloc[TOP_N:]
        control = rest.sample(n=min(TOP_N, len(rest)), random_state=RANDOM_SEED).copy()

        print(f"\n=== Ranked by {sort_col} (top {TOP_N}) ===")
        print(top20[DISPLAY_COLS].to_string(index=False))

        summarize(f"view={view} (ranked by {sort_col})", top20, control)

        top20["view"] = view
        top20["group"] = "top20"
        control["view"] = view
        control["group"] = "control"
        all_rows.append(top20)
        all_rows.append(control)

    combined = pd.concat(all_rows, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
