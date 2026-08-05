"""Backtest build_ranking(): for each week, do the top-ranked players actually score?

"Scored a touchdown" is measured against full-game touchdown production
(any receiving or rushing TD that week), not redzone_tds. redzone_tds only
counts touchdowns on plays that *started* inside the 20-yard line - a player
who catches a screen at the 30 and takes it to the house scores a real
touchdown that redzone_tds misses entirely, because the play itself began
outside the red zone. Grading the ranking against that narrower column would
understate real hits and unfairly penalize players who scored on long plays,
so ground truth here comes from aggregating touchdown plays across the full
play-by-play data (data/raw/pbp/), regardless of yardline_100. The old
redzone_tds-based numbers are kept alongside the new ones purely so the two
can be compared directly.

If scripts/baseline.py is written later, it should source ground truth from
`load_full_game_tds()` below rather than redzone_tds too, for the same reason.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ranking import build_ranking  # noqa: E402

RAW_PBP_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pbp"
PLAYER_WEEK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_week.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "backtest.csv"

SEASONS = [2022, 2023, 2024]
TOP_N = 10
WINDOW = 4


def load_full_game_tds() -> pd.DataFrame:
    """Per player/team/season/week count of receiving + rushing touchdowns, any yardline."""
    columns = ["season", "week", "posteam", "receiver_player_name", "rusher_player_name", "touchdown"]
    pbp = pd.read_parquet(RAW_PBP_DIR, columns=columns)
    pbp["season"] = pbp["season"].astype(int)
    scores = pbp[pbp["touchdown"] == 1]

    receiving = scores[scores["receiver_player_name"].notna()].rename(
        columns={"receiver_player_name": "player"}
    )
    rushing = scores[scores["rusher_player_name"].notna()].rename(columns={"rusher_player_name": "player"})
    combined = pd.concat([receiving[["season", "week", "posteam", "player"]],
                           rushing[["season", "week", "posteam", "player"]]])

    return (
        combined.groupby(["season", "week", "posteam", "player"])
        .size()
        .rename("full_game_tds")
        .reset_index()
    )


def backtest_week(player_week: pd.DataFrame, full_game_tds: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    ranking = build_ranking(player_week, as_of_week=week, season=season, window=WINDOW)
    if ranking.empty:
        return ranking

    top = ranking.head(TOP_N).copy()
    top["rank"] = range(1, len(top) + 1)

    redzone_actual = player_week[(player_week["season"] == season) & (player_week["week"] == week)][
        ["player", "posteam", "redzone_tds"]
    ]
    top = top.merge(redzone_actual, on=["player", "posteam"], how="left")
    top["redzone_tds"] = top["redzone_tds"].fillna(0).astype(int)
    top["scored_td_redzone"] = top["redzone_tds"] > 0

    full_game_actual = full_game_tds[(full_game_tds["season"] == season) & (full_game_tds["week"] == week)][
        ["player", "posteam", "full_game_tds"]
    ]
    top = top.merge(full_game_actual, on=["player", "posteam"], how="left")
    top["full_game_tds"] = top["full_game_tds"].fillna(0).astype(int)
    top["scored_td"] = top["full_game_tds"] > 0

    top = top.rename(columns={"as_of_week": "week"})
    return top[["season", "week", "rank", "player", "posteam", "score", "scored_td", "scored_td_redzone"]]


def print_metrics(label: str, backtest: pd.DataFrame, outcome_col: str) -> None:
    hit_rate_top5 = backtest.loc[backtest["rank"] <= 5, outcome_col].mean()
    hit_rate_top10 = backtest.loc[backtest["rank"] <= 10, outcome_col].mean()
    rank_corr = backtest["rank"].corr(backtest[outcome_col].astype(int))

    print(f"[{label}]")
    print(f"  Hit rate @ top-5:  {hit_rate_top5:.3f}")
    print(f"  Hit rate @ top-10: {hit_rate_top10:.3f}")
    print(f"  Correlation(rank position, scored_td): {rank_corr:.3f}")


def main() -> None:
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    full_game_tds = load_full_game_tds()

    results = []
    for season in SEASONS:
        max_week = int(player_week.loc[player_week["season"] == season, "week"].max())
        for week in range(2, max_week + 1):
            week_result = backtest_week(player_week, full_game_tds, season, week)
            if not week_result.empty:
                results.append(week_result)

    backtest = pd.concat(results, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(backtest)} rows to {OUTPUT_PATH}")
    print()

    print_metrics("NEW - full-game touchdowns", backtest, "scored_td")
    print()
    print_metrics("OLD - redzone_tds only", backtest, "scored_td_redzone")


if __name__ == "__main__":
    main()
