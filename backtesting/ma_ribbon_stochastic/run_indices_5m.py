import sys
from pathlib import Path

BASE_DIR = Path("/root/openalgo-autonomous-research")
sys.path.append(str(BASE_DIR / "backtesting" / "ma_ribbon_stochastic"))
from backtest import OUTPUT_DIR, generate_summary, load_data, run_backtest
from generate_charts import plot_portfolio_html


def main():
    print("=========================================================================")
    print(" 5-MINUTE INTRADAY BACKTEST: NIFTY 50 & BANKNIFTY ")
    print(" MA RIBBON (EMA 9/30/100) + STOCHASTIC (140, 10, 30) ")
    print("=========================================================================\n")

    indices = [("NIFTY", "NSE_INDEX", "NIFTY 50"), ("BANKNIFTY", "NSE_INDEX", "BANKNIFTY")]

    for sym, ex, name in indices:
        df_5m = load_data(sym, ex, "5m")
        if df_5m is not None and not df_5m.empty:
            print(f"==================== {name} (5-Min Intraday Timeframe) ====================")
            res_5m = run_backtest(df_5m, symbol=name, timeframe="5m", stoch_variant="state")
            if res_5m is not None:
                summary_5m = generate_summary(res_5m, None)
                print(summary_5m.to_string(index=False))
                print("\n")

                # Export trade log
                if res_5m["pf_bi"].trades.count() > 0:
                    res_5m["pf_bi"].trades.records_readable.to_csv(
                        OUTPUT_DIR / f"{sym}_5m_trades.csv", index=False
                    )
                    print(f"Saved {sym}_5m_trades.csv")

                # Export HTML report
                html_path = OUTPUT_DIR / f"{sym.lower()}_5m_report.html"
                plot_portfolio_html(res_5m, html_path)
                print(f"Saved {html_path.name}\n")
        else:
            print(f"No 5m data found for {name}")


if __name__ == "__main__":
    main()
