"""
Dashboard generator module - compiles results and outputs interactive dashboard.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def clean_dict_str(s):
    if not isinstance(s, str) or pd.isna(s):
        return {}
    # Replace np.float64(xxx) with xxx
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    s = re.sub(r"np\.int64\(([^)]+)\)", r"\1", s)
    s = re.sub(r"np\.nan", "None", s)
    s = re.sub(r"\bnan\b", "None", s)
    try:
        return ast.literal_eval(s)
    except Exception as e:
        logger.warning(f"Error parsing dict string: {s[:100]}... Error: {e}")
        return {}


def collect_run_data(runs_dir: Path) -> list:
    run_list = []

    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue

        config_file = run_path / "config.yaml"
        if not config_file.exists():
            continue

        run_id = run_path.name

        # Load config
        with open(config_file) as f:
            try:
                config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config for {run_id}: {e}")
                continue

        # Read CSV files and count rows
        def count_rows(filename):
            path = run_path / filename
            if not path.exists():
                return 0
            try:
                df = pd.read_csv(path)
                return len(df)
            except Exception:
                return 0

        gen_count = count_rows("generated.csv")
        backtest_count = count_rows("backtested.csv")
        winner_count = count_rows("winners.csv")
        survivor_count = count_rows("survivors.csv")

        # Rejection metrics
        rejections = []
        rejection_path = run_path / "rejected.csv"
        if rejection_path.exists():
            try:
                rejections = pd.read_csv(rejection_path).to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Could not load rejections for {run_id}: {e}")

        rejection_stages = {}
        rejection_reasons = {}
        for r in rejections:
            stage = r.get("stage", "unknown")
            reason = r.get("rejection_reason", "unknown")
            rejection_stages[stage] = rejection_stages.get(stage, 0) + 1
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        # Parse winners
        winners_list = []
        winners_path = run_path / "winners.csv"
        if winners_path.exists():
            try:
                winners_df = pd.read_csv(winners_path)
                strat_map = {}
                for _, row in winners_df.iterrows():
                    sid = row["strategy_id"]
                    backtest_data = clean_dict_str(row["backtest"])
                    validation_data = clean_dict_str(row["validation"])
                    signal_logic_data = clean_dict_str(row.get("signal_logic"))
                    genome_data = clean_dict_str(row.get("genome"))
                    winner_cat = str(row.get("winner_category", "unknown"))

                    if sid not in strat_map:
                        strat_map[sid] = {
                            "strategy_id": sid,
                            "rank": int(row.get("rank", 999)),
                            "category": str(row.get("category", "unknown")),
                            "composite_score": float(row.get("composite_score", 0.0)),
                            "pareto_front": int(row.get("pareto_front", 0)),
                            "winner_categories": [winner_cat],
                            "backtest": backtest_data,
                            "validation": validation_data,
                            "signal_logic": signal_logic_data,
                            "genome": genome_data,
                        }
                    else:
                        if winner_cat not in strat_map[sid]["winner_categories"]:
                            strat_map[sid]["winner_categories"].append(winner_cat)

                winners_list = list(strat_map.values())
                winners_list.sort(key=lambda x: x["rank"])
            except Exception as e:
                logger.error(f"Error loading/parsing winners for {run_id}: {e}")

        # Populate run entry
        run_data = {
            "run_id": run_id,
            "name": config.get("name", "Unnamed Run"),
            "description": config.get("description", ""),
            "symbols": [s.get("symbol", "") for s in config.get("data", {}).get("symbols", [])],
            "timeframes": config.get("data", {}).get("timeframes", []),
            "start_date": config.get("data", {}).get("start_date", ""),
            "end_date": config.get("data", {}).get("end_date", ""),
            "config": config,
            "stats": {
                "generated": gen_count,
                "backtested": backtest_count,
                "rejected": len(rejections),
                "winners": len(winners_list),
                "survivors": survivor_count,
                "fast_rejections": rejection_stages.get("fast_reject", 0),
                "backtest_rejections": rejection_stages.get("backtest_filter", 0),
                "validation_rejections": rejection_stages.get("validation", 0),
            },
            "rejections": {"stages": rejection_stages, "reasons": rejection_reasons},
            "winners": winners_list,
        }
        run_list.append(run_data)

    return run_list


def generate_html(run_data: list, output_path: Path):
    # Prepare JSON database to inject into HTML
    json_data = json.dumps(run_data, default=str)

    # Read the template from scripts/generate_dashboard.py, or keep it self-contained
    # We will write the full self-contained HTML generation code here:
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenAlgo Quant Research Engine - Dashboard</title>
    <!-- Inter Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {{
            --bg-base: #080b10;
            --bg-surface: rgba(17, 24, 39, 0.7);
            --bg-surface-opaque: #0f1622;
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.15);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;

            --accent-primary: #06b6d4;      /* Cyan */
            --accent-glow: rgba(6, 182, 212, 0.15);
            --accent-primary-gradient: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);

            --success: #10b981;             /* Emerald */
            --warning: #f59e0b;             /* Amber */
            --danger: #ef4444;              /* Rose */
            --purple: #8b5cf6;              /* Violet */

            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            overflow: hidden;
        }}

        /* App Layout */
        .sidebar {{
            width: 320px;
            background-color: #0c0f16;
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            height: 100vh;
            flex-shrink: 0;
        }}

        .main-content {{
            flex-grow: 1;
            height: 100vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.05), transparent 600px),
                        radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.03), transparent 600px);
        }}

        /* Sidebar Styling */
        .sidebar-header {{
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
        }}

        .brand-title {{
            font-size: 1.1rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            background: linear-gradient(135deg, #fff 30%, var(--accent-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .brand-subtitle {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }}

        .run-list-container {{
            flex-grow: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .run-item {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .run-item:hover {{
            background-color: rgba(255, 255, 255, 0.04);
            border-color: var(--border-hover);
            transform: translateY(-2px);
        }}

        .run-item.active {{
            background: rgba(6, 182, 212, 0.06);
            border-color: var(--accent-primary);
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.1);
        }}

        .run-item-name {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .run-item-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .run-item-id {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }}

        .badge {{
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-winners {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .badge-zero {{
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }}

        .sidebar-footer {{
            padding: 16px;
            border-top: 1px solid var(--border-color);
            font-size: 0.75rem;
            color: var(--text-muted);
            text-align: center;
        }}

        /* Main Content Header */
        .content-header {{
            padding: 32px 40px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}

        .header-title-area h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #fff;
        }}

        .header-title-area p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 6px;
            max-width: 600px;
        }}

        .header-meta {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            flex-wrap: wrap;
        }}

        .meta-tag {{
            display: flex;
            align-items: center;
            gap: 6px;
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .meta-tag strong {{
            color: #fff;
        }}

        /* Tab Navigation */
        .tab-nav {{
            display: flex;
            gap: 24px;
            padding: 0 40px;
            border-bottom: 1px solid var(--border-color);
            background-color: rgba(12, 15, 22, 0.3);
        }}

        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 16px 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            position: relative;
            transition: color 0.2s;
        }}

        .tab-btn:hover {{
            color: #fff;
        }}

        .tab-btn.active {{
            color: var(--accent-primary);
        }}

        .tab-btn.active::after {{
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: var(--accent-primary);
            box-shadow: 0 0 8px var(--accent-primary);
        }}

        /* Content Sections */
        .tab-content {{
            display: none;
            padding: 40px;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Grid System */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: var(--bg-surface);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--card-color, var(--accent-primary-gradient));
        }}

        .stat-label {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 800;
            color: #fff;
            line-height: 1.1;
        }}

        .stat-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        /* Section Layouts */
        .section-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 32px;
            margin-bottom: 42px;
        }}

        @media (max-width: 1024px) {{
            .section-row {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--bg-surface);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 32px;
            box-shadow: var(--card-shadow);
        }}

        .panel-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 24px;
            color: #fff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Funnel Diagram */
        .funnel-container {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 10px 0;
        }}

        .funnel-step {{
            display: flex;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            position: relative;
        }}

        .funnel-step::after {{
            content: '↓';
            position: absolute;
            bottom: -22px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 1.2rem;
            color: var(--text-muted);
            font-weight: bold;
        }}

        .funnel-step:last-child::after {{
            display: none;
        }}

        .funnel-step-info {{
            flex-grow: 1;
        }}

        .funnel-step-name {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .funnel-step-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .funnel-step-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
            text-align: right;
        }}

        .funnel-step-pct {{
            font-size: 0.75rem;
            font-weight: 600;
            background-color: rgba(255, 255, 255, 0.05);
            padding: 4px 8px;
            border-radius: 6px;
            color: var(--accent-primary);
            margin-left: 12px;
        }}

        /* Chart Canvas wrapper */
        .chart-wrapper {{
            height: 280px;
            position: relative;
        }}

        /* Table Styling */
        .table-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .search-input-wrapper {{
            position: relative;
            flex-grow: 1;
            max-width: 400px;
        }}

        .search-input {{
            width: 100%;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 16px;
            color: #fff;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s;
        }}

        .search-input:focus {{
            border-color: var(--accent-primary);
            background-color: rgba(255, 255, 255, 0.05);
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.1);
        }}

        .filter-select {{
            background-color: var(--bg-surface-opaque);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 16px;
            color: #fff;
            font-size: 0.9rem;
            outline: none;
            cursor: pointer;
        }}

        .filter-select:focus {{
            border-color: var(--accent-primary);
        }}

        .table-responsive {{
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        .dashboard-table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.88rem;
        }}

        .dashboard-table th {{
            background-color: rgba(12, 15, 22, 0.6);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            user-select: none;
        }}

        .dashboard-table th:hover {{
            color: #fff;
            background-color: rgba(255, 255, 255, 0.02);
        }}

        .dashboard-table th.sort-asc::after {{
            content: ' ▲';
            font-size: 0.7rem;
            color: var(--accent-primary);
        }}

        .dashboard-table th.sort-desc::after {{
            content: ' ▼';
            font-size: 0.7rem;
            color: var(--accent-primary);
        }}

        .dashboard-table td {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        .dashboard-table tbody tr {{
            cursor: pointer;
            transition: background-color 0.2s;
        }}

        .dashboard-table tbody tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        .dashboard-table tbody tr.selected {{
            background-color: rgba(6, 182, 212, 0.05);
            border-left: 2px solid var(--accent-primary);
        }}

        .strategy-id-cell {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--accent-primary);
        }}

        .winner-cat-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            background-color: rgba(139, 92, 246, 0.1);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.2);
            margin-right: 4px;
            margin-bottom: 4px;
        }}

        .metric-up {{
            color: var(--success);
            font-weight: 500;
        }}

        .metric-down {{
            color: var(--danger);
            font-weight: 500;
        }}

        /* Empty State */
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 40px;
            text-align: center;
            color: var(--text-muted);
        }}

        .empty-state-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
        }}

        .empty-state-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}

        /* Strategy Details View */
        .detail-header-panel {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}

        .detail-title-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-primary);
        }}

        .metric-grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 32px;
        }}

        @media (max-width: 768px) {{
            .metric-grid-3 {{
                grid-template-columns: 1fr;
            }}
        }}

        .metric-item-box {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
        }}

        .metric-item-title {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}

        .metric-item-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
        }}

        .metric-item-sub {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .detail-sub-section {{
            margin-bottom: 32px;
        }}

        .detail-sub-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}

        .metrics-list-table {{
            width: 100%;
            font-size: 0.85rem;
            border-collapse: collapse;
        }}

        .metrics-list-table tr:nth-child(even) {{
            background-color: rgba(255, 255, 255, 0.01);
        }}

        .metrics-list-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}

        .metrics-list-table td.m-label {{
            color: var(--text-secondary);
            font-weight: 500;
            width: 60%;
        }}

        .metrics-list-table td.m-value {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #fff;
        }}

        /* Configuration Details tab */
        .config-code-block {{
            background-color: #05070a;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #38bdf8;
        }}

        /* Rejections View */
        .rejections-summary-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            margin-bottom: 32px;
        }}

        @media (max-width: 768px) {{
            .rejections-summary-layout {{
                grid-template-columns: 1fr;
            }}
        }}

        .rejection-progress-bar-container {{
            margin-top: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}

        .rejection-progress-item {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .rejection-progress-lbl {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .rejection-progress-lbl span.name {{
            color: var(--text-primary);
        }}

        .rejection-progress-lbl span.val {{
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}

        .rejection-track-bar {{
            height: 8px;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }}

        .rejection-fill-bar {{
            height: 100%;
            background-color: var(--danger);
            border-radius: 4px;
        }}
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <div class="brand-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="margin-top:-2px"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                OpenAlgo Quant
            </div>
            <div class="brand-subtitle">Autonomous Research Engine</div>
        </div>

        <div class="run-list-container" id="run-list">
            <!-- Run items will be injected here -->
        </div>

        <div class="sidebar-footer">
            Generated Local Time: 2026-06-27
        </div>
    </div>

    <!-- Main Content Area -->
    <div class="main-content">
        <!-- Content Header -->
        <div class="content-header">
            <div class="header-title-area">
                <h1 id="active-run-name">Loading Run...</h1>
                <p id="active-run-desc">Please select a research run from the sidebar.</p>
                <div class="header-meta" id="active-run-meta">
                    <!-- Meta tags will be injected here -->
                </div>
            </div>
        </div>

        <!-- Navigation -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
            <button class="tab-btn" onclick="switchTab('winners')">Winning Strategies</button>
            <button class="tab-btn" onclick="switchTab('rejections')">Rejection Analysis</button>
            <button class="tab-btn" onclick="switchTab('config')">Run Configuration</button>
        </div>

        <!-- Tab contents -->
        <!-- OVERVIEW TAB -->
        <div id="tab-overview" class="tab-content active">
            <!-- Stats Row -->
            <div class="stats-grid" id="overview-stats-grid">
                <!-- Injected Stats -->
            </div>

            <!-- Funnel & Rejection Panel -->
            <div class="section-row">
                <div class="panel">
                    <div class="panel-title">System Discovery Funnel</div>
                    <div class="funnel-container" id="discovery-funnel">
                        <!-- Funnel steps -->
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title">Rejection Stage Breakdown</div>
                    <div class="chart-wrapper">
                        <canvas id="rejectionStageChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- WINNERS TAB -->
        <div id="tab-winners" class="tab-content">
            <div class="section-row" style="grid-template-columns: 3fr 2fr;">
                <!-- Left: Table -->
                <div class="panel" style="padding: 24px;">
                    <div class="panel-title">Ranked Winners</div>

                    <div class="table-controls">
                        <div class="search-input-wrapper">
                            <input type="text" id="strategy-search" class="search-input" placeholder="Search by ID or style..." oninput="filterWinnersTable()">
                        </div>
                        <select id="style-filter" class="filter-select" onchange="filterWinnersTable()">
                            <option value="all">All Trading Styles</option>
                            <option value="swing">Swing</option>
                            <option value="intraday">Intraday</option>
                            <option value="positional">Positional</option>
                            <option value="btst">BTST</option>
                        </select>
                    </div>

                    <div class="table-responsive">
                        <table class="dashboard-table" id="winners-table">
                            <thead>
                                <tr>
                                    <th onclick="sortTable('rank')">Rank</th>
                                    <th onclick="sortTable('strategy_id')">Strategy ID</th>
                                    <th onclick="sortTable('category')">Style</th>
                                    <th onclick="sortTable('cagr')">CAGR</th>
                                    <th onclick="sortTable('sharpe')">Sharpe</th>
                                    <th onclick="sortTable('max_drawdown_pct')">Max DD %</th>
                                    <th onclick="sortTable('win_rate')">Win Rate</th>
                                    <th onclick="sortTable('composite_score')">Score</th>
                                </tr>
                            </thead>
                            <tbody id="winners-table-body">
                                <!-- Injected rows -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Right: Selection Detail -->
                <div class="panel" id="strategy-detail-panel">
                    <div class="empty-state" id="detail-empty-state">
                        <div class="empty-state-icon">🔍</div>
                        <div class="empty-state-title">No Strategy Selected</div>
                        <p>Click on any row in the Winners table to view comprehensive metrics, walkforward scores, and parameter stability stats.</p>
                    </div>

                    <div id="detail-content" style="display: none;">
                        <!-- Detail contents injected here -->
                    </div>
                </div>
            </div>
        </div>

        <!-- REJECTIONS TAB -->
        <div id="tab-rejections" class="tab-content">
            <div class="panel" style="margin-bottom: 32px;">
                <div class="panel-title">Rejection Breakdown</div>
                <div class="rejections-summary-layout">
                    <div>
                        <h4 style="color:#fff; margin-bottom: 12px; font-size:0.95rem;">Killed by Stage</h4>
                        <div class="rejection-progress-bar-container" id="rejection-stage-progress">
                            <!-- Injected -->
                        </div>
                    </div>
                    <div>
                        <h4 style="color:#fff; margin-bottom: 12px; font-size:0.95rem;">Top Rejection Reasons</h4>
                        <div class="rejection-progress-bar-container" id="rejection-reason-progress">
                            <!-- Injected -->
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Fast Rejection Diagnostics</div>
                <p style="color: var(--text-secondary); font-size:0.88rem; margin-bottom: 20px; line-height:1.5;">
                    The OpenAlgo Quant engine applies a highly efficient multi-stage pipeline. The first stage is <strong>Fast Reject</strong> which eliminates mathematically contradictory or duplicate indicator conditions (e.g., trying to buy if RSI &gt; 70 and RSI &lt; 30 simultaneously) in less than a millisecond, before running the expensive multi-core backtester. This saves massive CPU resources.
                </p>
                <div class="chart-wrapper" style="height: 320px;">
                    <canvas id="rejectionReasonsChart"></canvas>
                </div>
            </div>
        </div>

        <!-- CONFIG TAB -->
        <div id="tab-config" class="tab-content">
            <div class="panel">
                <div class="panel-title">Run Configuration Details</div>
                <pre class="config-code-block" id="config-code-block">
                    <!-- Config yaml injected here -->
                </pre>
            </div>
        </div>
    </div>

    <!-- Data Injection -->
    <script id="runs-database" type="application/json">
        {json_data}
    </script>

    <!-- Main Logic Script -->
    <script>
        // Load database
        const runs = JSON.parse(document.getElementById('runs-database').textContent);
        let activeRunIndex = 0;
        let activeTab = 'overview';
        let currentSortColumn = 'rank';
        let currentSortAsc = true;

        let rejectionChartInstance = null;
        let reasonsChartInstance = null;

        // Init App
        window.addEventListener('DOMContentLoaded', () => {{
            renderRunList();
            if (runs.length > 0) {{
                selectRun(0);
            }} else {{
                showNoRunsState();
            }}
        }});

        function renderRunList() {{
            const listEl = document.getElementById('run-list');
            listEl.innerHTML = '';

            runs.forEach((run, index) => {{
                const item = document.createElement('div');
                item.className = `run-item ${{index === activeRunIndex ? 'active' : ''}}`;
                item.onclick = () => selectRun(index);

                const hasWinners = run.stats.winners > 0;
                const badgeText = hasWinners ? `${{run.stats.winners}} Winners` : '0 Winners';
                const badgeClass = hasWinners ? 'badge-winners' : 'badge-zero';

                item.innerHTML = `
                    <div class="run-item-name">${{run.name}}</div>
                    <div class="run-item-meta">
                        <span class="run-item-id">${{run.run_id}}</span>
                        <span class="badge ${{badgeClass}}">${{badgeText}}</span>
                    </div>
                `;
                listEl.appendChild(item);
            }});
        }}

        function selectRun(index) {{
            // Update active state in sidebar
            const items = document.querySelectorAll('.run-item');
            items.forEach((item, idx) => {{
                if (idx === index) item.classList.add('active');
                else item.classList.remove('active');
            }});

            activeRunIndex = index;
            const run = runs[index];

            // Render active run info
            document.getElementById('active-run-name').textContent = run.name;
            document.getElementById('active-run-desc').textContent = run.description || 'No description provided.';

            // Header meta tags
            const metaEl = document.getElementById('active-run-meta');
            metaEl.innerHTML = `
                <div class="meta-tag">Run ID: <strong>${{run.run_id}}</strong></div>
                <div class="meta-tag">Symbols: <strong>${{run.symbols.join(', ') || 'N/A'}}</strong></div>
                <div class="meta-tag">Timeframes: <strong>${{run.timeframes.join(', ') || 'N/A'}}</strong></div>
                <div class="meta-tag">Period: <strong>${{run.start_date}} to ${{run.end_date}}</strong></div>
            `;

            // Reset selected strategy in detail view
            document.getElementById('detail-empty-state').style.display = 'flex';
            document.getElementById('detail-content').style.display = 'none';

            // Refresh currently visible tab content
            refreshActiveTab();
        }}

        function switchTab(tabId) {{
            activeTab = tabId;

            // Toggle active button
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => {{
                if (btn.textContent.toLowerCase().includes(tabId)) btn.classList.add('active');
                else btn.classList.remove('active');
            }});

            // Toggle active content
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {{
                if (content.id === `tab-${{tabId}}`) content.classList.add('active');
                else content.classList.remove('active');
            }});

            refreshActiveTab();
        }}

        function refreshActiveTab() {{
            const run = runs[activeRunIndex];
            if (!run) return;

            if (activeTab === 'overview') {{
                renderOverviewTab(run);
            }} else if (activeTab === 'winners') {{
                renderWinnersTab(run);
            }} else if (activeTab === 'rejections') {{
                renderRejectionsTab(run);
            }} else if (activeTab === 'config') {{
                renderConfigTab(run);
            }}
        }}

        function renderOverviewTab(run) {{
            // 1. Stats Grid
            const statsGrid = document.getElementById('overview-stats-grid');

            const fastRejectPct = run.stats.generated > 0
                ? ((run.stats.fast_rejections / run.stats.generated) * 100).toFixed(1)
                : '0.0';

            const conversionPct = run.stats.generated > 0
                ? ((run.stats.winners / run.stats.generated) * 100).toFixed(2)
                : '0.00';

            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-label">Total Generated</div>
                    <div class="stat-value">${{run.stats.generated.toLocaleString()}}</div>
                    <div class="stat-sub">Grammar-based candidates</div>
                </div>
                <div class="stat-card" style="--card-color: var(--danger)">
                    <div class="stat-label">Fast Rejected</div>
                    <div class="stat-value">${{run.stats.fast_rejections.toLocaleString()}}</div>
                    <div class="stat-sub">${{fastRejectPct}}% filters failed early</div>
                </div>
                <div class="stat-card" style="--card-color: var(--warning)">
                    <div class="stat-label">Backtested</div>
                    <div class="stat-value">${{run.stats.backtested.toLocaleString()}}</div>
                    <div class="stat-sub">${{run.stats.backtest_rejections.toLocaleString()}} rejected by metrics</div>
                </div>
                <div class="stat-card" style="--card-color: var(--success)">
                    <div class="stat-label">Ultimate Winners</div>
                    <div class="stat-value">${{run.stats.winners}}</div>
                    <div class="stat-sub">${{conversionPct}}% success rate</div>
                </div>
            `;

            // 2. Funnel
            const funnel = document.getElementById('discovery-funnel');
            const total = run.stats.generated || 1;
            const fastPassed = total - run.stats.fast_rejections;
            const btPassed = run.stats.backtested - run.stats.backtest_rejections;

            funnel.innerHTML = `
                <div class="funnel-step">
                    <div class="funnel-step-info">
                        <div class="funnel-step-name">1. Generation Stage</div>
                        <div class="funnel-step-sub">Synthesized candidate strategy space</div>
                    </div>
                    <div class="funnel-step-value">${{run.stats.generated.toLocaleString()}}</div>
                    <div class="funnel-step-pct">100%</div>
                </div>
                <div class="funnel-step">
                    <div class="funnel-step-info">
                        <div class="funnel-step-name">2. Fast Rejection Pipeline</div>
                        <div class="funnel-step-sub">Eliminated structural and rule contradictions</div>
                    </div>
                    <div class="funnel-step-value">${{fastPassed.toLocaleString()}}</div>
                    <div class="funnel-step-pct">${{((fastPassed/total)*100).toFixed(1)}}%</div>
                </div>
                <div class="funnel-step">
                    <div class="funnel-step-info">
                        <div class="funnel-step-name">3. Vectorized Backtesting</div>
                        <div class="funnel-step-sub">Simulated trading performance on training data</div>
                    </div>
                    <div class="funnel-step-value">${{run.stats.backtested.toLocaleString()}}</div>
                    <div class="funnel-step-pct">${{((run.stats.backtested/total)*100).toFixed(1)}}%</div>
                </div>
                <div class="funnel-step">
                    <div class="funnel-step-info">
                        <div class="funnel-step-name">4. Multi-Objective Filtering</div>
                        <div class="funnel-step-sub">Checked minimum Sharpe, Trades, and Drawdown thresholds</div>
                    </div>
                    <div class="funnel-step-value">${{btPassed.toLocaleString()}}</div>
                    <div class="funnel-step-pct">${{((btPassed/total)*100).toFixed(1)}}%</div>
                </div>
                <div class="funnel-step">
                    <div class="funnel-step-info">
                        <div class="funnel-step-name">5. Walkforward & Robustness Validation</div>
                        <div class="funnel-step-sub">Passed out-of-sample, Monte Carlo, and parameter stability</div>
                    </div>
                    <div class="funnel-step-value">${{run.stats.winners}}</div>
                    <div class="funnel-step-pct">${{conversionPct}}%</div>
                </div>
            `;

            // 3. Rejection Stage Chart
            setTimeout(() => {{
                const ctx = document.getElementById('rejectionStageChart').getContext('2d');
                if (rejectionChartInstance) rejectionChartInstance.destroy();

                rejectionChartInstance = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Survived', 'Fast Rejected', 'Backtest Filtered', 'Validation Rejected'],
                        datasets: [{{
                            data: [
                                run.stats.winners,
                                run.stats.fast_rejections,
                                run.stats.backtest_rejections,
                                run.stats.validation_rejections
                            ],
                            backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{
                                    color: '#9ca3af',
                                    font: {{ family: 'Inter', size: 11 }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}, 50);
        }}

        function renderWinnersTab(run) {{
            renderWinnersTable(run.winners);
        }}

        function renderWinnersTable(winners) {{
            const tableBody = document.getElementById('winners-table-body');
            tableBody.innerHTML = '';

            if (winners.length === 0) {{
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="8" style="text-align: center; padding: 40px; color: var(--text-muted);">
                            No winning strategies survived the validation pipeline for this run.
                        </td>
                    </tr>
                `;
                return;
            }}

            winners.forEach(w => {{
                const tr = document.createElement('tr');
                tr.id = `strat-row-${{w.strategy_id}}`;
                tr.onclick = () => selectStrategy(w.strategy_id);

                const cagrVal = w.backtest.cagr;
                const sharpeVal = w.backtest.sharpe;
                const ddVal = w.backtest.max_drawdown_pct;
                const winRateVal = w.backtest.win_rate;

                tr.innerHTML = `
                    <td><strong>${{w.rank}}</strong></td>
                    <td class="strategy-id-cell">${{w.strategy_id}}</td>
                    <td><span class="badge" style="background-color: rgba(255,255,255,0.05); color:#fff; border:1px solid rgba(255,255,255,0.1);">${{w.category.toUpperCase()}}</span></td>
                    <td class="${{cagrVal >= 0 ? 'metric-up' : 'metric-down'}}">${{cagrVal ? cagrVal.toFixed(2) + '%' : '0.00%'}}</td>
                    <td class="${{sharpeVal >= 0.8 ? 'metric-up' : ''}}">${{sharpeVal ? sharpeVal.toFixed(2) : '0.00'}}</td>
                    <td class="metric-down">${{ddVal ? ddVal.toFixed(2) + '%' : '0.00%'}}</td>
                    <td>${{winRateVal ? (winRateVal * 100).toFixed(1) + '%' : '0.0%'}}</td>
                    <td style="font-weight:600; color:#fff;">${{w.composite_score.toFixed(4)}}</td>
                `;
                tableBody.appendChild(tr);
            }});
        }}

        function filterWinnersTable() {{
            const run = runs[activeRunIndex];
            if (!run) return;

            const searchQuery = document.getElementById('strategy-search').value.toLowerCase();
            const styleFilter = document.getElementById('style-filter').value;

            let filtered = run.winners.filter(w => {{
                const matchesSearch = w.strategy_id.toLowerCase().includes(searchQuery) || w.category.toLowerCase().includes(searchQuery);
                const matchesStyle = styleFilter === 'all' || w.category.toLowerCase() === styleFilter.toLowerCase();
                return matchesSearch && matchesStyle;
            }});

            renderWinnersTable(filtered);
        }}

        function sortTable(column) {{
            const run = runs[activeRunIndex];
            if (!run || run.winners.length === 0) return;

            if (currentSortColumn === column) {{
                currentSortAsc = !currentSortAsc;
            }} else {{
                currentSortColumn = column;
                currentSortAsc = true;
            }}

            // Update headers class
            const ths = document.querySelectorAll('#winners-table th');
            const colMap = {{
                'rank': 0,
                'strategy_id': 1,
                'category': 2,
                'cagr': 3,
                'sharpe': 4,
                'max_drawdown_pct': 5,
                'win_rate': 6,
                'composite_score': 7
            }};

            ths.forEach((th, index) => {{
                th.className = '';
                if (index === colMap[column]) {{
                    th.className = currentSortAsc ? 'sort-asc' : 'sort-desc';
                }}
            }});

            const sorted = [...run.winners].sort((a, b) => {{
                let valA, valB;

                if (column === 'rank' || column === 'strategy_id' || column === 'category' || column === 'composite_score') {{
                    valA = a[column];
                    valB = b[column];
                }} else {{
                    valA = a.backtest[column];
                    valB = b.backtest[column];
                }}

                if (typeof valA === 'string') {{
                    return currentSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
                }} else {{
                    return currentSortAsc ? valA - valB : valB - valA;
                }}
            }});

            renderWinnersTable(sorted);
        }}

        function selectStrategy(strategyId) {{
            const run = runs[activeRunIndex];
            const w = run.winners.find(item => item.strategy_id === strategyId);
            if (!w) return;

            // Highlight selected row
            const rows = document.querySelectorAll('#winners-table-body tr');
            rows.forEach(r => r.classList.remove('selected'));
            const selRow = document.getElementById(`strat-row-${{strategyId}}`);
            if (selRow) selRow.classList.add('selected');

            document.getElementById('detail-empty-state').style.display = 'none';
            const detailContent = document.getElementById('detail-content');
            detailContent.style.display = 'block';

            // Generate Winner badges html
            const catBadgesHtml = w.winner_categories.map(cat => `<span class="winner-cat-badge">${{cat.toUpperCase().replace('_', ' ')}}</span>`).join('');

            const bt = w.backtest;
            const val = w.validation;

            detailContent.innerHTML = `
                <div class="detail-header-panel">
                    <div>
                        <div class="detail-title-id">${{w.strategy_id}}</div>
                        <div style="margin-top: 6px; display:flex; flex-wrap:wrap;">
                            ${{catBadgesHtml}}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Composite Score</div>
                        <div style="font-size:1.6rem; font-weight:800; color:var(--accent-primary);">${{w.composite_score.toFixed(4)}}</div>
                    </div>
                </div>

                <!-- KPI Grid -->
                <div class="metric-grid-3">
                    <div class="metric-item-box">
                        <div class="metric-item-title">Annualized Return</div>
                        <div class="metric-item-value ${{bt.cagr >= 0 ? 'metric-up' : 'metric-down'}}">${{bt.cagr ? bt.cagr.toFixed(1) + '%' : '0.0%'}}</div>
                        <div class="metric-item-sub">CAGR</div>
                    </div>
                    <div class="metric-item-box">
                        <div class="metric-item-title">Sharpe Ratio</div>
                        <div class="metric-item-value">${{bt.sharpe ? bt.sharpe.toFixed(2) : '0.00'}}</div>
                        <div class="metric-item-sub">Risk-Adjusted return</div>
                    </div>
                    <div class="metric-item-box">
                        <div class="metric-item-title">Max Drawdown</div>
                        <div class="metric-item-value metric-down">${{bt.max_drawdown_pct ? bt.max_drawdown_pct.toFixed(1) + '%' : '0.0%'}}</div>
                        <div class="metric-item-sub">Peak-to-trough drop</div>
                    </div>
                </div>

                <!-- Section: Backtest Summary -->
                <div class="detail-sub-section">
                    <div class="detail-sub-title">Simulation Metrics</div>
                    <table class="metrics-list-table">
                        <tr>
                            <td class="m-label">Net Profit</td>
                            <td class="m-value ${{bt.net_profit_pct >= 0 ? 'metric-up' : 'metric-down'}}">${{bt.net_profit_pct ? bt.net_profit_pct.toFixed(2) + '%' : '0.00%'}} (${{bt.net_profit ? bt.net_profit.toLocaleString(undefined, {{maximumFractionDigits: 0}}) : '0'}})</td>
                        </tr>
                        <tr>
                            <td class="m-label">Profit Factor</td>
                            <td class="m-value">${{bt.profit_factor ? bt.profit_factor.toFixed(2) : '0.00'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Win Rate</td>
                            <td class="m-value">${{bt.win_rate ? (bt.win_rate * 100).toFixed(1) + '%' : '0.0%'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Total Trades</td>
                            <td class="m-value">${{bt.total_trades}} (W: ${{bt.winning_trades}} / L: ${{bt.losing_trades}})</td>
                        </tr>
                        <tr>
                            <td class="m-label">Average Trade PnL</td>
                            <td class="m-value">${{bt.avg_trade_pct ? bt.avg_trade_pct.toFixed(2) + '%' : '0.00%'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Recovery Factor</td>
                            <td class="m-value">${{bt.recovery_factor ? bt.recovery_factor.toFixed(2) : '0.00'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Expectancy</td>
                            <td class="m-value">${{bt.expectancy ? bt.expectancy.toFixed(3) : '0.000'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Ulcer Index (DD Depth/Duration)</td>
                            <td class="m-value">${{bt.ulcer_index ? bt.ulcer_index.toFixed(2) : '0.00'}}</td>
                        </tr>
                    </table>
                </div>

                <!-- Section: Validation & Robustness -->
                <div class="detail-sub-section">
                    <div class="detail-sub-title">Robustness & Stress Tests</div>
                    <table class="metrics-list-table">
                        <tr>
                            <td class="m-label">Overall Robustness Score</td>
                            <td class="m-value" style="color: var(--accent-primary); font-weight:700;">${{val.robustness_score ? val.robustness_score.toFixed(1) : '0.0'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Parameter Stability Score</td>
                            <td class="m-value">${{val.param_stability_score ? val.param_stability_score.toFixed(1) : '0.0'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Stress Test Score (Noise Resistance)</td>
                            <td class="m-value">${{val.stress_test_score ? val.stress_test_score.toFixed(1) : '0.0'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Out-of-Sample (OOS) Sharpe</td>
                            <td class="m-value">${{val.oos_sharpe ? val.oos_sharpe.toFixed(2) : '0.00'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">OOS Sharpe Decay Ratio</td>
                            <td class="m-value">${{val.oos_sharpe_decay ? (val.oos_sharpe_decay * 100).toFixed(1) + '%' : '0.0%'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Walk Forward Score / Consistency</td>
                            <td class="m-value">${{val.walk_forward_score ? val.walk_forward_score.toFixed(1) : '0.0'}} / ${{val.walk_forward_consistency ? (val.walk_forward_consistency * 100).toFixed(0) + '%' : '0%'}}</td>
                        </tr>
                        <tr>
                            <td class="m-label">Monte Carlo Score</td>
                            <td class="m-value">${{val.monte_carlo_score ? val.monte_carlo_score.toFixed(1) : '0.0'}}</td>
                        </tr>
                    </table>
                </div>

                <!-- Section: Strategy Logic & Rules -->
                <div class="detail-sub-section">
                    <div class="detail-sub-title">Strategy Rules & Logic</div>
                    <div style="background: var(--bg-surface-opaque); border: 1px solid var(--accent-glow); border-radius: 6px; padding: 12px; margin-top: 8px;">
                        <div style="margin-bottom: 12px;">
                            <span style="font-weight: 700; color: var(--accent-primary); font-size: 0.85rem; text-transform: uppercase;">Entry Long Conditions:</span>
                            <div style="font-family: monospace; font-size: 0.85rem; padding: 8px; background-color: var(--bg-base); border-radius: 4px; margin-top: 4px; word-break: break-all; color: var(--text-primary); border-left: 3px solid var(--accent-primary);">
                                ${{w.signal_logic && w.signal_logic.entry_long ? w.signal_logic.entry_long : 'No entry conditions recorded.'}}
                            </div>
                        </div>
                        <div>
                            <span style="font-weight: 700; color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase;">Exit & Risk Parameters:</span>
                            <div style="font-family: monospace; font-size: 0.85rem; padding: 8px; background-color: var(--bg-base); border-radius: 4px; margin-top: 4px; color: var(--text-primary); border-left: 3px solid var(--text-muted);">
                                ${{w.signal_logic && w.signal_logic.exit_long ?
                                    `Stop Loss: ${{w.signal_logic.exit_long.stop_loss_pct ? w.signal_logic.exit_long.stop_loss_pct + '%' : 'None'}} | ` +
                                    `Take Profit: ${{w.signal_logic.exit_long.take_profit_pct ? w.signal_logic.exit_long.take_profit_pct + '%' : 'None'}} | ` +
                                    `Trailing Stop: ${{w.signal_logic.exit_long.trailing_stop_pct ? w.signal_logic.exit_long.trailing_stop_pct + '%' : 'None'}} | ` +
                                    `Max Hold Bars: ${{w.signal_logic.exit_long.max_hold_bars || 'None'}}` : 'No exit parameters recorded.'}}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }}

        function renderRejectionsTab(run) {{
            // 1. Stage progress bar lists
            const stageProgress = document.getElementById('rejection-stage-progress');
            stageProgress.innerHTML = '';

            const totalRejections = run.stats.rejected || 1;

            const stages = [
                {{ name: 'Fast Rejection (Indicator checks)', count: run.stats.fast_rejections }},
                {{ name: 'Backtest Filter (Metrics thresholds)', count: run.stats.backtest_rejections }},
                {{ name: 'Validation Failures (OOS, Stability, WFA)', count: run.stats.validation_rejections }}
            ];

            stages.forEach(st => {{
                const pct = ((st.count / totalRejections) * 100).toFixed(1);

                const item = document.createElement('div');
                item.className = 'rejection-progress-item';
                item.innerHTML = `
                    <div class="rejection-progress-lbl">
                        <span class="name">${{st.name}}</span>
                        <span class="val">${{st.count.toLocaleString()}} (${{pct}}%)</span>
                    </div>
                    <div class="rejection-track-bar">
                        <div class="rejection-fill-bar" style="width: ${{pct}}%; background-color:${{st.name.includes('Fast') ? '#ef4444' : st.name.includes('Backtest') ? '#f59e0b' : '#8b5cf6'}}"></div>
                    </div>
                `;
                stageProgress.appendChild(item);
            }});

            // 2. Reason progress list
            const reasonProgress = document.getElementById('rejection-reason-progress');
            reasonProgress.innerHTML = '';

            const sortedReasons = Object.entries(run.rejections.reasons)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5);

            if (sortedReasons.length === 0) {{
                reasonProgress.innerHTML = `<div style="color:var(--text-muted); font-size:0.85rem; padding: 20px 0;">No rejection reasons logged for this run.</div>`;
            }} else {{
                sortedReasons.forEach(([reason, count]) => {{
                    const pct = ((count / totalRejections) * 100).toFixed(1);
                    const readableReason = reason.replace(/_/g, ' ');

                    const item = document.createElement('div');
                    item.className = 'rejection-progress-item';
                    item.innerHTML = `
                        <div class="rejection-progress-lbl">
                            <span class="name" style="text-transform: capitalize;">${{readableReason}}</span>
                            <span class="val">${{count.toLocaleString()}} (${{pct}}%)</span>
                        </div>
                        <div class="rejection-track-bar">
                            <div class="rejection-fill-bar" style="width: ${{pct}}%"></div>
                        </div>
                    `;
                    reasonProgress.appendChild(item);
                }});
            }}

            // 3. Reasons Chart
            setTimeout(() => {{
                const ctx = document.getElementById('rejectionReasonsChart').getContext('2d');
                if (reasonsChartInstance) reasonsChartInstance.destroy();

                const labels = Object.keys(run.rejections.reasons).map(lbl => lbl.replace(/_/g, ' '));
                const data = Object.values(run.rejections.reasons);

                reasonsChartInstance = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Killed Count',
                            data: data,
                            backgroundColor: 'rgba(239, 68, 68, 0.65)',
                            borderColor: '#ef4444',
                            borderWidth: 1,
                            borderRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            x: {{
                                grid: {{ display: false }},
                                ticks: {{ color: '#9ca3af', font: {{ size: 10 }} }}
                            }},
                            y: {{
                                grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                                ticks: {{ color: '#9ca3af' }}
                            }}
                        }}
                    }}
                }});
            }}, 50);
        }}

        function renderConfigTab(run) {{
            const codeBlock = document.getElementById('config-code-block');

            // Format config to clean yaml-like text manually
            let yamlStr = '';

            function printObj(obj, indent = 0) {{
                const pad = ' '.repeat(indent);
                for (let key in obj) {{
                    const val = obj[key];
                    if (val && typeof val === 'object' && !Array.isArray(val)) {{
                        yamlStr += `${{pad}}${{key}}:\\n`;
                        printObj(val, indent + 2);
                    }} else if (Array.isArray(val)) {{
                        yamlStr += `${{pad}}${{key}}:\\n`;
                        val.forEach(item => {{
                            if (typeof item === 'object') {{
                                yamlStr += `${{pad}}  - \\n`;
                                printObj(item, indent + 4);
                            }} else {{
                                yamlStr += `${{pad}}  - ${{item}}\\n`;
                            }}
                        }});
                    }} else {{
                        yamlStr += `${{pad}}${{key}}: ${{val}}\\n`;
                    }}
                }}
            }}

            printObj(run.config);
            codeBlock.textContent = yamlStr || JSON.stringify(run.config, null, 2);
        }}

        function showNoRunsState() {{
            document.body.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100vw; height:100vh; background-color:#080b10; color:var(--text-secondary); text-align:center; padding:40px;">
                    <div style="font-size:4rem; margin-bottom:20px;">📊</div>
                    <h2 style="color:#fff; font-size:1.5rem; margin-bottom:12px;">No Research Runs Found</h2>
                    <p style="max-width:400px; line-height:1.6; font-size:0.95rem; color:var(--text-muted)">
                        We couldn't locate any completed research runs in <code>./data/runs/</code>. Please execute the research pipeline first using:
                        <br><br>
                        <code style="display:block; background-color:#0f1622; padding:12px; border-radius:8px; color:var(--accent-primary); font-family:monospace; text-align:left; font-size:0.85rem;">
                            python -m quant_engine run config/default_research.yaml
                        </code>
                    </p>
                </div>
            `;
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info(f"Dashboard HTML generated successfully: {output_path}")


def generate_all_dashboards(runs_base_dir: Path | str):
    runs_dir = Path(runs_base_dir)
    workspace_dir = runs_dir.parent.parent

    logger.info(f"Generating dashboards. Scanning runs in: {runs_dir}")
    run_list = collect_run_data(runs_dir)

    # Save workspace root dashboard
    root_dash = workspace_dir / "dashboard.html"
    generate_html(run_list, root_dash)
    logger.info(f"Generated root dashboard at: {root_dash}")

    # Save docs dashboard
    docs_dash = workspace_dir / "docs" / "dashboard.html"
    generate_html(run_list, docs_dash)
    logger.info(f"Generated docs dashboard at: {docs_dash}")

    # Also save run-specific dashboards inside each run folder!
    for run in run_list:
        run_id = run["run_id"]
        run_dash_path = runs_dir / run_id / "dashboard.html"
        generate_html([run], run_dash_path)
        logger.info(f"Generated run-specific dashboard at: {run_dash_path}")
