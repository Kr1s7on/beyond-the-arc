"""Train the location-first model used by the Streamlit app.

The full model (``models/curry_shot_model.pkl``) uses ``action_type`` and
``shot_zone_basic``, which the app cannot observe.  Deriving those labels from
distance with hard-coded thresholds made the app's probability jump at 4/16/24
ft and stay flat in between (boosted trees had no distance splits along those
paths).

This script trains a separate, app-only model on the features the UI really
knows: continuous shot distance/angle, clutch/clock, and form/context values.
The distance effect is a single linear term in ``log_dist`` with a negative
coefficient, so the predicted probability is *strictly decreasing* with shot
distance -- the slider always moves the number, and the curve has no cliffs.

Run with the project's Python 3.9.6 environment:

    python notebooks/train_app_model.py
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "curry_2009to25.csv"
MODEL_PATH = ROOT / "models" / "curry_location_model.pkl"

CAT_FEATURES = ["period", "opp"]
# ``is_3pt`` is deliberately excluded: distance already encodes it, and its
# positive coefficient put a bump back in at the 3-point line.
NUM_FEATURES = [
    "time_frac", "is_clutch", "is_overtime", "is_home", "attempt_idx",
    "attempt_frac", "game_fg_sofar", "game_3fg_sofar", "days_rest", "roll5_3fg",
]
FEATURES = ["log_dist", "abs_angle"] + NUM_FEATURES + CAT_FEATURES
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the leak-safe feature set from the modelling notebook."""
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw.columns = raw.columns.str.lower()

    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"].astype(str))
    df = df.sort_values(["game_date", "game_id", "game_event_id"]).reset_index(drop=True)
    y = df["shot_made_flag"]

    # home/away + opponent
    df["is_home"] = (df["htm"] == "GSW").astype(int)
    df["opp"] = np.where(df["htm"] == "GSW", df["vtm"], df["htm"])

    # geometry
    df["log_dist"] = np.log1p(df["shot_distance"].astype(float))
    df["abs_angle"] = np.abs(np.degrees(np.arctan2(df["loc_x"], df["loc_y"])))

    # clock / situation
    df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
    df["time_frac"] = df["time_remaining"] / 720.0
    df["is_clutch"] = (df["time_remaining"] < 10).astype(int)
    df["is_overtime"] = (df["period"] > 4).astype(int)
    df["is_3pt"] = (df["shot_distance"] >= 23).astype(int)

    # within-game attempt context
    df["attempt_idx"] = df.groupby("game_id").cumcount()
    df["attempt_tot"] = df.groupby("game_id")["game_id"].transform("size")
    df["attempt_frac"] = df["attempt_idx"] / df["attempt_tot"]

    # shooting form so far in this game (shift(1) keeps it leak-safe)
    df["game_fg_sofar"] = (
        df.groupby("game_id")["shot_made_flag"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(y.mean())
    )
    made3 = (
        (df["shot_made_flag"] * df["is_3pt"])
        .groupby(df["game_id"])
        .transform(lambda s: s.shift(1).expanding().sum())
    )
    att3 = (
        df["is_3pt"]
        .groupby(df["game_id"])
        .transform(lambda s: s.shift(1).expanding().sum())
        .replace(0, np.nan)
    )
    df["game_3fg_sofar"] = (made3 / att3).fillna(
        df.loc[df["is_3pt"] == 1, "shot_made_flag"].mean()
    )

    # rest days between games
    game_dates = df.groupby("game_id")["game_date"].first().sort_values()
    rest_days = game_dates.diff().dt.days
    df["days_rest"] = df["game_id"].map(rest_days).fillna(7).clip(0, 7)

    # rolling last-5-games 3P% before this game
    game_3fg = (
        df.groupby("game_id")
        .apply(lambda g: (g["is_3pt"] * g["shot_made_flag"]).sum() / max(g["is_3pt"].sum(), 1))
        .reindex(game_dates.index)
    )
    df["roll5_3fg"] = df["game_id"].map(
        game_3fg.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(df.loc[df["is_3pt"] == 1, "shot_made_flag"].mean())

    return df, y


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CAT_FEATURES),
        # one linear term -> logit is strictly decreasing in distance
        ("dist", StandardScaler(), ["log_dist"]),
        ("angle", SplineTransformer(n_knots=6, degree=3, include_bias=False),
         ["abs_angle"]),
    ])
    return Pipeline([
        ("pre", pre),
        ("m", LogisticRegression(C=0.5, max_iter=3000)),
    ])


def app_prediction(model: Pipeline, dist: float, angle: float = 0.0,
                   clutch: bool = False) -> float:
    row = {
        "log_dist": np.log1p(dist),
        "abs_angle": abs(angle),
        "time_frac": 5.0 / 720.0 if clutch else 1.0,
        "is_clutch": int(clutch),
        "is_overtime": 0,
        "is_home": 0,
        "attempt_idx": 9,
        "attempt_frac": 0.47,
        "game_fg_sofar": 0.474,
        "game_3fg_sofar": 0.422,
        "days_rest": 2,
        "roll5_3fg": 0.409,
        "period": 3,
        "opp": "PHX",
    }
    return float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])


def main() -> None:
    df, y = build_dataset()
    print(f"rows: {len(df)}  overall FG%: {y.mean():.4f}")

    oof = cross_val_predict(
        make_pipeline(), df[FEATURES], y, cv=CV,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    auc = roc_auc_score(y, oof)
    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = float(thr[f1.argmax()])
    print(f"OOF AUC        : {auc:.4f}")
    print(f"best threshold : {best_thr:.3f}  (F1 = {f1.max():.4f})")

    final = make_pipeline().fit(df[FEATURES], y)

    curve = [app_prediction(final, d) * 100 for d in range(31)]
    steps = np.diff(curve)
    print("distance curve @ 0 deg:")
    print("  " + " ".join(f"{d}:{p:.1f}" for d, p in enumerate(curve)))
    print(f"  strictly-decreasing steps: {int((steps < -1e-9).sum())}/{len(steps)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final,
            "features": FEATURES,
            "threshold": best_thr,
        }, f)
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
