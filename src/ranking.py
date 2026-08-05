"""Rank players by red zone opportunity using a trailing window of games."""

import pandas as pd

# Composite score weights. redzone_target_share and redzone_touches carry the
# most weight because volume/opportunity metrics are "sticky" week to week -
# a player who's seeing 30% of his team's redzone targets this month is
# likely to keep seeing a similar share next month. goal_line_touches and
# team_redzone_trip_rate matter but are noisier. td_conversion_rate gets a
# near-zero weight on purpose: over a 3-5 game window a player might have,
# say, 6 redzone touches and 2 TDs (33%) or 0 TDs (0%) purely from how plays
# were called or a bounce at the goal line, not from a repeatable skill.
# Small-sample TD rates regress hard to the mean, so leaning on them to rank
# players would mostly be ranking on noise. It's still surfaced in the output
# for context, just not allowed to move the score.
WEIGHTS = {
    "redzone_target_share": 0.35,
    "redzone_touches_per_game": 0.30,
    "goal_line_touches_per_game": 0.20,
    "team_redzone_trip_rate": 0.15,
    "td_conversion_rate": 0.0,
}


def _min_max_normalize(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def build_ranking(
    player_week_df: pd.DataFrame,
    as_of_week: int,
    season: int,
    window: int = 4,
) -> pd.DataFrame:
    """Rank players on red zone opportunity using the `window` games before `as_of_week`.

    Only weeks strictly before `as_of_week` are used, so the ranking never
    looks ahead into the week being predicted.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    start_week = max(as_of_week - window, 1)
    eligible = player_week_df[
        (player_week_df["season"] == season)
        & (player_week_df["week"] >= start_week)
        & (player_week_df["week"] < as_of_week)
    ]

    grouped = eligible.groupby(["player", "posteam"]).agg(
        games=("week", "nunique"),
        redzone_target_share=("redzone_target_share", "mean"),
        redzone_touches_per_game=("redzone_touches", "mean"),
        goal_line_touches_per_game=("goal_line_touches", "mean"),
        team_redzone_trip_rate=("redzone_trips_per_game", "mean"),
        redzone_touches_total=("redzone_touches", "sum"),
        redzone_tds_total=("redzone_tds", "sum"),
    )

    grouped["td_conversion_rate"] = (
        grouped["redzone_tds_total"] / grouped["redzone_touches_total"]
    ).fillna(0.0)
    grouped = grouped.drop(columns=["redzone_touches_total", "redzone_tds_total"])

    score = pd.Series(0.0, index=grouped.index)
    for column, weight in WEIGHTS.items():
        if weight == 0.0:
            continue
        score += weight * _min_max_normalize(grouped[column])
    grouped["score"] = score

    ranking = grouped.reset_index()
    ranking.insert(0, "as_of_week", as_of_week)
    ranking.insert(1, "season", season)

    return ranking.sort_values("score", ascending=False).reset_index(drop=True)
