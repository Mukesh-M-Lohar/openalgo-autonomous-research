"""
Feature Importance Analysis for Supertrend Touch Scanner
=========================================================
Analyzes which features best predict a successful bounce after a
supertrend band touch using 6 complementary methods:

  1. Mutual Information (captures nonlinear relationships)
  2. XGBoost feature importance (gradient + gain)
  3. SHAP values (model-agnostic explanations)
  4. Random Forest feature importance (impurity + permutation)
  5. Information Coefficient (IC) for each feature
  6. Correlation by market regime (trending vs ranging)

Target: `bounced` (True/False) — did price reach the bounce threshold
before the supertrend trend reversed?

Usage:
  python3 feature_importance.py [csv_path]
  python3 feature_importance.py supertrend_touch_output/supertrend_touches_ALL_5m.csv
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ────────────────────────────────────────────────────────

# Columns to EXCLUDE from features (identifiers, targets, leaky outcomes)
EXCLUDE_COLS = {
    "timestamp",
    "symbol",
    "signal_type",
    # Target and outcome columns (would leak future info)
    "bounced",
    "bounce_pct",
    "peak_bounce_points",
    "peak_bounce_timestamp",
    "points_at_reversal",
    "reversal_timestamp",
    "bars_to_reversal",
    "bounce_threshold_points",
    "bounce_threshold_hit",
    "bars_to_threshold_hit",
    # Internal tracking
    "trend_start_idx",
}

# Columns that are categorical (will be label-encoded)
CATEGORICAL_COLS = {"candle_pattern", "candle_signal"}

TARGET = "bounced"


# ── Data Preparation ─────────────────────────────────────────────────


def prepare_data(csv_path: str) -> tuple:
    """Load CSV, clean, encode categoricals, return (X, y, feature_names)."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(int)
    print(
        f"  Target distribution: bounced={y.sum()} ({y.mean() * 100:.1f}%), "
        f"not_bounced={len(y) - y.sum()} ({(1 - y.mean()) * 100:.1f}%)"
    )

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in feature_cols:
            df[col] = df[col].astype("category").cat.codes  # -1 for NaN

    # Drop non-numeric columns that slipped through
    X = df[feature_cols].copy()
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric: {non_numeric}")
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    # Fill NaN with median (tree models handle this, but MI/IC need it)
    X = X.fillna(X.median())

    # Drop constant columns
    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if const_cols:
        print(f"  Dropping constant: {const_cols}")
        X = X.drop(columns=const_cols)

    print(f"  Final features: {len(X.columns)}")
    return X, y, df


# ── 1. Mutual Information ────────────────────────────────────────────


def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual Information (nonlinear dependency measure)."""
    from sklearn.feature_selection import mutual_info_classif

    print("\n[1/6] Computing Mutual Information ...")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X.columns, name="mutual_info").sort_values(ascending=False)
    print("  Top 10:")
    for feat, val in mi_series.head(10).items():
        print(f"    {feat:30s}  MI = {val:.4f}")
    return mi_series


# ── 2. XGBoost Feature Importance ────────────────────────────────────


def compute_xgboost_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """XGBoost gain-based and weight-based importance."""
    from xgboost import XGBClassifier

    print("\n[2/6] Training XGBoost ...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)

    # Gain importance
    gain = pd.Series(
        model.get_booster().get_score(importance_type="gain"),
        name="xgb_gain",
    ).reindex(X.columns, fill_value=0)

    # Weight (frequency) importance
    weight = pd.Series(
        model.get_booster().get_score(importance_type="weight"),
        name="xgb_weight",
    ).reindex(X.columns, fill_value=0)

    # Normalize
    gain = gain / gain.sum() if gain.sum() > 0 else gain
    weight = weight / weight.sum() if weight.sum() > 0 else weight

    result = pd.DataFrame({"xgb_gain": gain, "xgb_weight": weight})
    result = result.sort_values("xgb_gain", ascending=False)

    print("  Top 10 by gain:")
    for feat, row in result.head(10).iterrows():
        print(f"    {feat:30s}  gain={row['xgb_gain']:.4f}  weight={row['xgb_weight']:.4f}")

    return result, model


# ── 3. SHAP Values ──────────────────────────────────────────────────


def compute_shap_values(model, X: pd.DataFrame) -> pd.Series:
    """SHAP mean absolute values from the XGBoost model."""
    import shap

    print("\n[3/6] Computing SHAP values ...")
    # Use a sample for speed if dataset is large
    if len(X) > 5000:
        X_sample = X.sample(5000, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_mean = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
        name="shap_mean_abs",
    ).sort_values(ascending=False)

    print("  Top 10:")
    for feat, val in shap_mean.head(10).items():
        print(f"    {feat:30s}  SHAP = {val:.4f}")
    return shap_mean


# ── 4. Random Forest Feature Importance ──────────────────────────────


def compute_rf_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Random Forest impurity-based + permutation importance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    print("\n[4/6] Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Impurity-based importance
    impurity = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="rf_impurity",
    )

    # Permutation importance (on a held-out subset)
    print("  Computing permutation importance ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(
        perm.importances_mean,
        index=X.columns,
        name="rf_permutation",
    )

    result = pd.DataFrame({"rf_impurity": impurity, "rf_permutation": perm_imp})
    result = result.sort_values("rf_impurity", ascending=False)

    print("  Top 10 by impurity:")
    for feat, row in result.head(10).iterrows():
        print(
            f"    {feat:30s}  impurity={row['rf_impurity']:.4f}  perm={row['rf_permutation']:.4f}"
        )

    return result


# ── 5. Information Coefficient (IC) ──────────────────────────────────


def compute_information_coefficient(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Spearman rank IC and Pearson IC for each feature vs target."""
    print("\n[5/6] Computing Information Coefficient (IC) ...")

    results = []
    for col in X.columns:
        x = X[col].values
        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(x, y)
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        # Point-biserial (same as Pearson for binary target)
        results.append(
            {
                "feature": col,
                "ic_spearman": spearman_r,
                "ic_spearman_pval": spearman_p,
                "ic_pearson": pearson_r,
                "ic_pearson_pval": pearson_p,
                "ic_abs_spearman": abs(spearman_r),
            }
        )

    ic_df = (
        pd.DataFrame(results).set_index("feature").sort_values("ic_abs_spearman", ascending=False)
    )

    print("  Top 10 by |Spearman IC|:")
    for feat, row in ic_df.head(10).iterrows():
        sig = (
            "***"
            if row["ic_spearman_pval"] < 0.001
            else "**"
            if row["ic_spearman_pval"] < 0.01
            else "*"
            if row["ic_spearman_pval"] < 0.05
            else ""
        )
        print(
            f"    {feat:30s}  IC={row['ic_spearman']:+.4f} {sig}   Pearson={row['ic_pearson']:+.4f}"
        )

    return ic_df


# ── 6. Correlation by Market Regime ──────────────────────────────────


def compute_regime_correlations(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Split data into trending vs ranging regimes and compare feature correlations."""
    print("\n[6/6] Computing Regime-Specific Correlations ...")

    # Define regime using ADX: trending (ADX > 25) vs ranging (ADX <= 25)
    if "adx" in X.columns:
        adx = X["adx"]
        trending_mask = adx > 25
        ranging_mask = adx <= 25
        regime_col = "adx"
    else:
        # Fallback: use ATR percentile
        if "atr" in X.columns:
            atr_median = X["atr"].median()
            trending_mask = X["atr"] > atr_median
            ranging_mask = X["atr"] <= atr_median
            regime_col = "atr"
        else:
            print("  No ADX or ATR column found, skipping regime analysis")
            return pd.DataFrame()

    print(
        f"  Regime split ({regime_col}): trending={trending_mask.sum()}, ranging={ranging_mask.sum()}"
    )

    results = []
    for col in X.columns:
        # Trending regime
        if trending_mask.sum() > 30:
            tr_corr, tr_p = stats.spearmanr(X.loc[trending_mask, col], y[trending_mask])
        else:
            tr_corr, tr_p = np.nan, np.nan

        # Ranging regime
        if ranging_mask.sum() > 30:
            rg_corr, rg_p = stats.spearmanr(X.loc[ranging_mask, col], y[ranging_mask])
        else:
            rg_corr, rg_p = np.nan, np.nan

        results.append(
            {
                "feature": col,
                "trending_ic": tr_corr,
                "trending_pval": tr_p,
                "ranging_ic": rg_corr,
                "ranging_pval": rg_p,
                "regime_diff": abs(tr_corr - rg_corr)
                if not (np.isnan(tr_corr) or np.isnan(rg_corr))
                else np.nan,
            }
        )

    regime_df = (
        pd.DataFrame(results).set_index("feature").sort_values("regime_diff", ascending=False)
    )

    print("  Top 10 features with LARGEST regime difference:")
    for feat, row in regime_df.head(10).iterrows():
        print(
            f"    {feat:30s}  trending={row['trending_ic']:+.4f}  ranging={row['ranging_ic']:+.4f}  diff={row['regime_diff']:.4f}"
        )

    # Also show features that work well in BOTH regimes
    both_strong = regime_df[
        (regime_df["trending_ic"].abs() > 0.05) & (regime_df["ranging_ic"].abs() > 0.05)
    ].sort_values("regime_diff")
    if len(both_strong) > 0:
        print("\n  Features strong in BOTH regimes (smallest diff, |IC|>0.05):")
        for feat, row in both_strong.head(10).iterrows():
            print(
                f"    {feat:30s}  trending={row['trending_ic']:+.4f}  ranging={row['ranging_ic']:+.4f}"
            )

    return regime_df


# ── Combined Rankings ────────────────────────────────────────────────


def build_combined_ranking(
    mi: pd.Series,
    xgb: pd.DataFrame,
    shap_vals: pd.Series,
    rf: pd.DataFrame,
    ic: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all methods into a single ranking table."""
    print("\n" + "=" * 70)
    print("COMBINED FEATURE RANKING")
    print("=" * 70)

    all_features = mi.index.tolist()

    combined = pd.DataFrame(index=all_features)
    combined["mutual_info"] = mi
    combined["xgb_gain"] = xgb["xgb_gain"]
    combined["shap_mean_abs"] = shap_vals
    combined["rf_impurity"] = rf["rf_impurity"]
    combined["ic_abs_spearman"] = ic["ic_abs_spearman"]

    # Rank each method (1 = best)
    for col in combined.columns:
        combined[f"{col}_rank"] = combined[col].rank(ascending=False, method="min")

    # Average rank across all methods
    rank_cols = [c for c in combined.columns if c.endswith("_rank")]
    combined["avg_rank"] = combined[rank_cols].mean(axis=1)
    combined = combined.sort_values("avg_rank")

    # Normalized scores (0-1 scale for each method)
    score_cols = ["mutual_info", "xgb_gain", "shap_mean_abs", "rf_impurity", "ic_abs_spearman"]
    for col in score_cols:
        col_max = combined[col].max()
        if col_max > 0:
            combined[f"{col}_norm"] = combined[col] / col_max
        else:
            combined[f"{col}_norm"] = 0

    norm_cols = [c for c in combined.columns if c.endswith("_norm")]
    combined["composite_score"] = combined[norm_cols].mean(axis=1)
    combined = combined.sort_values("composite_score", ascending=False)

    print("\nTop 20 features by composite score:")
    print(
        f"{'Rank':>4}  {'Feature':30s}  {'Composite':>9}  {'MI':>6}  {'XGB':>6}  {'SHAP':>6}  {'RF':>6}  {'IC':>6}"
    )
    print("-" * 105)
    for rank, (feat, row) in enumerate(combined.head(20).iterrows(), 1):
        print(
            f"{rank:4d}  {feat:30s}  {row['composite_score']:9.4f}  "
            f"{row['mutual_info_norm']:6.3f}  {row['xgb_gain_norm']:6.3f}  "
            f"{row['shap_mean_abs_norm']:6.3f}  {row['rf_impurity_norm']:6.3f}  "
            f"{row['ic_abs_spearman_norm']:6.3f}"
        )

    return combined


# ── Export Results ────────────────────────────────────────────────────


def export_results(
    combined: pd.DataFrame,
    mi: pd.Series,
    xgb: pd.DataFrame,
    shap_vals: pd.Series,
    rf: pd.DataFrame,
    ic: pd.DataFrame,
    regime: pd.DataFrame,
    output_dir: str,
):
    """Save all results to CSV files."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Combined ranking
    out = combined.sort_values("composite_score", ascending=False)
    out.to_csv(f"{output_dir}/combined_ranking.csv")
    print(f"\nSaved: {output_dir}/combined_ranking.csv")

    # Individual method results
    mi.to_csv(f"{output_dir}/mutual_information.csv")
    xgb.to_csv(f"{output_dir}/xgboost_importance.csv")
    shap_vals.to_csv(f"{output_dir}/shap_values.csv")
    rf.to_csv(f"{output_dir}/random_forest_importance.csv")
    ic.to_csv(f"{output_dir}/information_coefficient.csv")
    if len(regime) > 0:
        regime.to_csv(f"{output_dir}/regime_correlations.csv")

    print(f"All results saved to {output_dir}/")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "supertrend_touch_output/supertrend_touches_ALL_5m.csv"
    )
    output_dir = "supertrend_touch_output/feature_importance"

    X, y, df = prepare_data(csv_path)

    # 1. Mutual Information
    mi = compute_mutual_information(X, y)

    # 2. XGBoost
    xgb_imp, xgb_model = compute_xgboost_importance(X, y)

    # 3. SHAP (uses the XGBoost model)
    shap_vals = compute_shap_values(xgb_model, X)

    # 4. Random Forest
    rf_imp = compute_rf_importance(X, y)

    # 5. Information Coefficient
    ic = compute_information_coefficient(X, y)

    # 6. Regime Correlations
    regime = compute_regime_correlations(X, y, df)

    # Combined ranking
    combined = build_combined_ranking(mi, xgb_imp, shap_vals, rf_imp, ic, regime)

    # Export
    export_results(combined, mi, xgb_imp, shap_vals, rf_imp, ic, regime, output_dir)


if __name__ == "__main__":
    main()
