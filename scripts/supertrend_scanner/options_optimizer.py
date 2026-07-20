import itertools
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# pyrefly: ignore [missing-import]
import options_backtester

# pyrefly: ignore [missing-import]
from data_cache import CachedDataFetcher

# pyrefly: ignore [missing-import]
from download_all_options import get_client

warnings.filterwarnings("ignore")


def run_combination(args):
    offset, tp, sl, st_mult, df_underlying, opts_master = args

    # Override Backtester globals in this isolated process
    options_backtester.OPTION_STRIKE_OFFSET = offset
    options_backtester.TAKE_PROFIT_PCT = tp
    options_backtester.STOP_LOSS_PCT = sl
    options_backtester.UNDERLYING_ST_MULT = st_mult
    options_backtester.OPTION_ST_MULT = st_mult

    # Suppress stdout to prevent flooded console
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    try:
        res_df = options_backtester.main(
            print_logs=False,
            df_underlying_prefetched=df_underlying,
            opts_master_prefetched=opts_master,
        )
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    if res_df is None or res_df.empty:
        return None

    win_rate = (len(res_df[res_df["Return %"] > 0]) / len(res_df) * 100) if len(res_df) else 0
    avg_return = res_df["Return %"].mean() if len(res_df) else 0

    ce = res_df[res_df["Option Type"] == "CE"]
    ce_win_rate = (len(ce[ce["Return %"] > 0]) / len(ce) * 100) if len(ce) else 0

    pe = res_df[res_df["Option Type"] == "PE"]
    pe_win_rate = (len(pe[pe["Return %"] > 0]) / len(pe) * 100) if len(pe) else 0

    avg_drawdown = res_df["Max Drawdown %"].mean()
    best_trade = res_df["Return %"].max()
    worst_trade = res_df["Return %"].min()
    total_return_sum = res_df["Return %"].sum() if len(res_df) else 0
    total_return_comp = ((np.prod(1 + res_df["Return %"] / 100) - 1) * 100) if len(res_df) else 0

    return {
        "Strike Offset": offset,
        "Take Profit %": tp,
        "Stop Loss %": sl,
        "ST Mult": st_mult,
        "Total Trades": len(res_df),
        "Win Rate %": round(win_rate, 2),
        "Avg Return %": round(avg_return, 2),
        "Total Return % (Sum)": round(total_return_sum, 2),
        "Total Return % (Compounded)": round(total_return_comp, 2),
        "Avg Drawdown %": round(avg_drawdown, 2),
        "Best Trade %": round(best_trade, 2),
        "Worst Trade %": round(worst_trade, 2),
        "CE Win Rate %": round(ce_win_rate, 2),
        "PE Win Rate %": round(pe_win_rate, 2),
    }


def run_optimization():
    print("==================================================")
    print("STARTING MULTI-CORE GRID SEARCH OPTIMIZATION")
    print("==================================================")

    client = get_client()
    fetcher = CachedDataFetcher(client)
    print("Calculating Buy & Hold baseline for BANKNIFTY...")
    df_bh = fetcher.fetch_history("BANKNIFTY", "NSE_INDEX", "D", "2026-07-01", "2026-08-01")
    buy_hold_return = 0
    if not df_bh.empty:
        start_price = df_bh["close"].iloc[0]
        end_price = df_bh["close"].iloc[-1]
        buy_hold_return = ((end_price - start_price) / start_price) * 100
        print(f"BANKNIFTY Buy & Hold Return: {buy_hold_return:.2f}%")

    print("\nFetching Instruments Master and Underlying data for workers...")

    master_cache_file = (
        "/root/openalgo-autonomous-research/supertrend_touch_output/.cache/opts_master.csv"
    )
    if os.path.exists(master_cache_file):
        print("Loading Instruments Master from local cache...")
        opts_master = pd.read_csv(master_cache_file)
        opts_master["expiry_date"] = pd.to_datetime(opts_master["expiry_date"])
    else:
        for attempt in range(3):
            instruments = client.instruments(exchange="NFO")
            if isinstance(instruments, list):
                break
            print(f"API timeout/error on attempt {attempt + 1}, retrying in 2 seconds...")
            time.sleep(2)

        if (
            isinstance(instruments, dict)
            and "status" in instruments
            and instruments["status"] == "error"
        ):
            print(f"Failed to fetch instruments after 3 attempts: {instruments}")
            sys.exit(1)

        opts_master = pd.DataFrame(instruments)
        opts_master = opts_master[
            (opts_master["name"] == options_backtester.UNDERLYING_SYMBOL)
            & (opts_master["instrumenttype"].isin(["CE", "PE"]))
        ].copy()
        opts_master["expiry_date"] = pd.to_datetime(opts_master["expiry"])

        # Save to cache
        os.makedirs(os.path.dirname(master_cache_file), exist_ok=True)
        opts_master.to_csv(master_cache_file, index=False)
        print("Instruments Master downloaded and cached locally.")

    df_underlying = fetcher.fetch_history(
        options_backtester.UNDERLYING_SYMBOL,
        options_backtester.UNDERLYING_EXCHANGE,
        options_backtester.UNDERLYING_TIMEFRAME,
        options_backtester.START_DATE,
        options_backtester.END_DATE,
    )

    print("Starting Grid Search Optimization...\n")

    # Grid definition
    strike_offsets = [0, 500]
    take_profits = [10.0, 20.0, 30.0, 40.0]
    stop_losses = [10.0, 15.0, 20.0]
    st_mults = [2.0, 3.0]

    # Build combinations
    combinations = []
    for offset, tp, sl, st in itertools.product(
        strike_offsets, take_profits, stop_losses, st_mults
    ):
        combinations.append((offset, tp, sl, st, df_underlying, opts_master))

    # Add "No TP / No SL" combinations
    for offset in strike_offsets:
        for st in st_mults:
            combinations.append((offset, None, 10, st, df_underlying, opts_master))

    total_runs = len(combinations)
    print(f"Total Combinations to Test: {total_runs}")

    results_list = []

    # Use half of the available CPU cores
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"Launching {max_workers} worker processes...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(run_combination, args): args for args in combinations}

        completed = 0
        for future in as_completed(future_to_args):
            completed += 1
            args = future_to_args[future]
            offset, tp, sl, st = args[0], args[1], args[2], args[3]
            try:
                res = future.result()
                if res:
                    results_list.append(res)
                    print(
                        f"[{completed}/{total_runs}] SUCCESS: Offset {offset} | TP/SL: {tp}/{sl} | ST Mult: {st} | Avg Return: {res['Avg Return %']}%"
                    )
                else:
                    print(
                        f"[{completed}/{total_runs}] NO TRADES: Offset {offset} | TP/SL: {tp}/{sl} | ST Mult: {st}"
                    )
            except Exception as exc:
                print(
                    f"[{completed}/{total_runs}] FAILED: Offset {offset} | TP/SL: {tp}/{sl} | ST Mult: {st} -> {exc}"
                )

    # Compile results
    if results_list:
        opt_df = pd.DataFrame(results_list)
        # Sort by Average Return descending
        opt_df = opt_df.sort_values(by="Avg Return %", ascending=False).reset_index(drop=True)

        out_path = (
            "/root/openalgo-autonomous-research/supertrend_touch_output/optimization_results.csv"
        )
        opt_df.to_csv(out_path, index=False)

        print("\n==================================================")
        print("TOP 5 PARAMETER COMBINATIONS (By Avg Return %)")
        print("==================================================")
        print(opt_df.head(5).to_string())

        print(f"\nFull optimization results saved to: {out_path}")

        # Send WhatsApp Notification
        try:
            print("\nSending WhatsApp notification to 919790856795...")
            top_res = opt_df.iloc[0]
            msg = (
                f"✅ Optimization Finished!\n"
                f"Total Combinations: {total_runs}\n\n"
                f"🏆 Best Performance:\n"
                f"Avg Return: {top_res['Avg Return %']}%\n"
                f"Params: Offset={top_res['Strike Offset']}, TP={top_res['Take Profit %']}, "
                f"SL={top_res['Stop Loss %']}, ST={top_res['ST Mult']}\n"
                f"Win Rate: {top_res['Win Rate %']}%\n"
                f"Drawdown: {top_res['Avg Drawdown %']}%\n\n"
                f"Full CSV report is attached below."
            )
            client.whatsapp(message=msg, to="919790856795", document=out_path)
            print("WhatsApp message sent successfully!")
        except Exception as e:
            print(f"Error sending WhatsApp: {e}")
    else:
        print("No valid results found during optimization.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    run_optimization()
