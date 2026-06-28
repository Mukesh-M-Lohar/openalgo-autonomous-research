"""Explainability Engine — calculates SHAP, permutation, and native feature importances."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Computes model interpretability metrics and exports reports & charts."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize ExplainabilityEngine with a configuration.

        Args:
            config: Explainability configuration flags.
        """
        self.config = config or {}
        self.shap_enabled = self.config.get("shap", True)
        self.permutation_enabled = self.config.get("permutation_importance", True)
        self.feature_importance_enabled = self.config.get("feature_importance", True)

    def generate_explainability_report(
        self,
        adapter: Any,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str,
    ) -> dict[str, Any]:
        """Compute enabled explainability metrics, draw charts, and export reports.

        Args:
            adapter: Trained model adapter instance.
            X: Validation/testing features.
            y: Validation/testing target labels.
            output_dir: Directory where plots and reports will be saved.

        Returns:
            Dictionary containing feature importance summaries.
        """
        os.makedirs(output_dir, exist_ok=True)
        report: dict[str, Any] = {}
        feature_names = adapter.feature_names or list(X.columns)

        # 1. Native Feature Importance
        if self.feature_importance_enabled and hasattr(adapter.model, "feature_importances_"):
            try:
                importances = adapter.model.feature_importances_
                native_imp = pd.Series(importances, index=feature_names).sort_values(
                    ascending=False
                )
                report["native_importance"] = native_imp.to_dict()
                self._save_importance_table(native_imp, output_dir, "native_importance.csv")
                self._plot_importance(
                    native_imp, "Native Feature Importance", output_dir, "native_importance.png"
                )
            except Exception as e:
                logger.warning(f"Failed to generate native feature importance: {e}", exc_info=True)

        # 2. Permutation Importance
        if self.permutation_enabled:
            try:
                from sklearn.inspection import permutation_importance

                result = permutation_importance(adapter.model, X, y, n_repeats=5, random_state=42)
                perm_imp = pd.Series(result.importances_mean, index=feature_names).sort_values(
                    ascending=False
                )
                report["permutation_importance"] = perm_imp.to_dict()
                self._save_importance_table(perm_imp, output_dir, "permutation_importance.csv")
                self._plot_importance(
                    perm_imp,
                    "Permutation Feature Importance",
                    output_dir,
                    "permutation_importance.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate permutation importance: {e}", exc_info=True)

        # 3. SHAP values
        if self.shap_enabled:
            try:
                import matplotlib
                import shap

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Sample background data if dataset is too large
                X_sample = X
                if len(X) > 100:
                    X_sample = X.sample(100, random_state=42)

                # Explainer dispatch
                explainer = shap.Explainer(adapter.model, X_sample)
                shap_values = explainer(X_sample)

                # Save SHAP Summary Plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                shap_plot_path = os.path.join(output_dir, "shap_summary.png")
                plt.title("SHAP Feature Summary")
                plt.tight_layout()
                plt.savefig(shap_plot_path, dpi=150)
                plt.close()

                # Get average absolute SHAP values per feature
                if hasattr(shap_values, "values"):
                    vals = shap_values.values
                else:
                    vals = shap_values

                mean_shap = np.abs(vals).mean(axis=0)
                # Handle multiclass dimension output if needed
                if len(mean_shap.shape) > 1:
                    mean_shap = mean_shap.mean(axis=1)

                shap_imp = pd.Series(mean_shap, index=X_sample.columns).sort_values(ascending=False)
                report["shap_importance"] = shap_imp.to_dict()
                self._save_importance_table(shap_imp, output_dir, "shap_importance.csv")
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanation: {e}", exc_info=True)

        # 4. Generate Combined Top Features summary
        self._generate_top_features_summary(report, output_dir)

        return report

    def _save_importance_table(self, series: pd.Series, output_dir: str, filename: str) -> None:
        path = os.path.join(output_dir, filename)
        df_imp = pd.DataFrame({"Feature": series.index, "Importance": series.values})
        df_imp.to_csv(path, index=False)

    def _plot_importance(
        self, series: pd.Series, title: str, output_dir: str, filename: str
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Plot top 15 features
            top_series = series.head(15).sort_values(ascending=True)

            plt.figure(figsize=(10, 6))
            top_series.plot(kind="barh", color="#1f77b4")
            plt.title(title)
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot feature importance chart: {e}")

    def _generate_top_features_summary(self, report: dict[str, Any], output_dir: str) -> None:
        """Create a Markdown and CSV summary table ranking features combined across methods."""
        ranks: dict[str, list[float]] = {}

        for method, values in report.items():
            sorted_feats = sorted(values.keys(), key=lambda k: values[k], reverse=True)
            for rank, feat in enumerate(sorted_feats, 1):
                if feat not in ranks:
                    ranks[feat] = []
                ranks[feat].append(rank)

        # Average rank calculation (lower is better / more important)
        summary_rows = []
        for feat, rank_list in ranks.items():
            avg_rank = np.mean(rank_list)
            summary_rows.append(
                {"Feature": feat, "AvgRank": avg_rank, "Appearances": len(rank_list)}
            )

        df_summary = pd.DataFrame(summary_rows).sort_values("AvgRank")

        # Save CSV
        df_summary.to_csv(os.path.join(output_dir, "top_features_summary.csv"), index=False)

        # Save Markdown Table
        md_path = os.path.join(output_dir, "top_features_summary.md")
        with open(md_path, "w") as f:
            f.write("# Model Feature Importance Summary\n\n")
            f.write(
                "Ranked by average relative importance rank across active explainability engines:\n\n"
            )
            f.write("| Rank | Feature | Avg Rank | Appearances |\n")
            f.write("|------|---------|----------|-------------|\n")
            for i, row in enumerate(df_summary.head(20).itertuples(), 1):
                f.write(f"| {i} | `{row.Feature}` | {row.AvgRank:.2f} | {row.Appearances} |\n")
