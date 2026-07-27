import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path("/root/openalgo-autonomous-research")
sys.path.append(str(BASE_DIR / "backtesting" / "ma_ribbon_stochastic"))
from backtest import load_data, run_backtest

OUTPUT_DIR = BASE_DIR / "backtesting" / "ma_ribbon_stochastic"
# ruff: noqa: E402

def plot_portfolio_html(res, filepath):
    pf = res["pf_bi"]
    df = res["df"]
    close = res["close"]
    value = pf.value()
    drawdown = pf.drawdown() * 100

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"Price & Indicators - {res['symbol']} ({res['timeframe']})",
            "Portfolio Value (INR)",
            "Drawdown (%)",
        ),
    )

    # 1. Price & EMAs
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=close,
            mode="lines",
            name="Close Price",
            line=dict(color="#2962FF", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=res["ema9"],
            mode="lines",
            name="EMA 9",
            line=dict(color="#f6c309", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=res["ema30"],
            mode="lines",
            name="EMA 30",
            line=dict(color="#fb9800", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=res["ema100"],
            mode="lines",
            name="EMA 100",
            line=dict(color="#f60c0c", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Long Entries / Exits
    long_entries = df.index[res["long_entries"]]
    long_exits = df.index[res["long_exits"]]
    if len(long_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_entries,
                y=close.loc[long_entries],
                mode="markers",
                name="Buy (Long)",
                marker=dict(symbol="triangle-up", size=10, color="green"),
            ),
            row=1,
            col=1,
        )
    if len(long_exits) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_exits,
                y=close.loc[long_exits],
                mode="markers",
                name="Exit (Long)",
                marker=dict(symbol="x", size=8, color="orange"),
            ),
            row=1,
            col=1,
        )

    # Short Entries / Exits
    short_entries = df.index[res["short_entries"]]
    short_exits = df.index[res["short_exits"]]
    if len(short_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_entries,
                y=close.loc[short_entries],
                mode="markers",
                name="Sell (Short)",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            ),
            row=1,
            col=1,
        )
    if len(short_exits) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_exits,
                y=close.loc[short_exits],
                mode="markers",
                name="Exit (Short)",
                marker=dict(symbol="x", size=8, color="magenta"),
            ),
            row=1,
            col=1,
        )

    # 2. Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=value,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00E676", width=1.5),
        ),
        row=2,
        col=1,
    )

    # 3. Drawdown
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=drawdown,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#FF5252", width=1),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=900,
        title_text=f"MA Ribbon + Stochastic Strategy - {res['symbol']} ({res['timeframe']})",
        showlegend=True,
    )

    fig.write_html(str(filepath))


def main():
    df_bse = load_data("BSE", "NSE", "15m")
    if df_bse is not None:
        res_bse = run_backtest(df_bse, symbol="BSE", timeframe="15m")
        plot_portfolio_html(res_bse, OUTPUT_DIR / "bse_15m_report.html")
        print("Generated bse_15m_report.html")

    df_cdsl = load_data("CDSL", "NSE", "15m")
    if df_cdsl is not None:
        res_cdsl = run_backtest(df_cdsl, symbol="CDSL", timeframe="15m")
        plot_portfolio_html(res_cdsl, OUTPUT_DIR / "cdsl_15m_report.html")
        print("Generated cdsl_15m_report.html")

    df_sbin = load_data("SBIN", "NSE", "D")
    if df_sbin is not None:
        res_sbin = run_backtest(df_sbin, symbol="SBIN", timeframe="D")
        plot_portfolio_html(res_sbin, OUTPUT_DIR / "sbin_daily_report.html")
        print("Generated sbin_daily_report.html")


if __name__ == "__main__":
    main()
