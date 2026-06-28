import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_engine.config import load_config
from quant_engine.data.cache import DataCache


def main():
    parser = argparse.ArgumentParser(
        description="Cache helper to convert local CSV data to Parquet format based on a research config."
    )
    parser.add_argument("config", help="Path to the research config YAML file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    cache = DataCache(Path("./data/cache"))

    start = config.data.start_date
    end = config.data.end_date
    timeframes = config.data.timeframes
    symbols = config.data.symbols

    for sym_cfg in symbols:
        sym = sym_cfg.symbol
        exch = sym_cfg.exchange
        for tf in timeframes:
            tf_map = {"D": "D", "1d": "D", "15m": "15m", "5m": "5m", "1h": "1h", "60m": "1h"}
            tf_key = tf_map.get(tf, tf)
            csv_path = Path(f"./data/cache/{sym}_{exch}_{tf_key}.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                df = df.set_index("timestamp").sort_index()
                df = df.loc[start:end]
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                cache.put(sym, exch, tf, start, end, df)
                print(f"Cached {sym}/{exch}/{tf} from {start} to {end} successfully")
            else:
                print(f"CSV file not found for {sym}/{exch} ({tf}): {csv_path}")


if __name__ == "__main__":
    main()
