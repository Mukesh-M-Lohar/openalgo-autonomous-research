import os

import pandas as pd


def _ensure_ist(df):
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        # If min hour is < 7, it's UTC -> Localize to UTC then convert to Asia/Kolkata
        if len(df) > 0 and df.index.min().hour < 7:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            # Already IST naive -> Localize to Asia/Kolkata
            df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df


class CachedDataFetcher:
    def __init__(self, client, cache_dir="/root/openalgo-autonomous-research/cache"):
        self.client = client
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_history(self, symbol, exchange, interval, start_date, end_date):
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{interval}.csv")

        # Parse requested dates with Asia/Kolkata timezone
        req_start = (
            pd.to_datetime(start_date).tz_localize("Asia/Kolkata")
            if pd.to_datetime(start_date).tz is None
            else pd.to_datetime(start_date).tz_convert("Asia/Kolkata")
        )
        req_end = (
            pd.to_datetime(end_date).tz_localize("Asia/Kolkata")
            if pd.to_datetime(end_date).tz is None
            else pd.to_datetime(end_date).tz_convert("Asia/Kolkata")
        )
        req_end = req_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df = pd.DataFrame()
        cache_loaded = False

        # Load cache if exists
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                ts_col = next(
                    (c for c in ["timestamp", "datetime", "date", "time"] if c in df.columns), None
                )
                if ts_col:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                    df = _ensure_ist(df)
                    df = df.sort_index()
                    cache_loaded = True
            except Exception as e:
                print(f"Error reading cache for {symbol}: {e}")
                df = pd.DataFrame()

        needs_fetch = False
        fetch_start = req_start
        fetch_end = req_end

        # Fallback to 1m cache if the requested cache file doesn't exist
        if not cache_loaded and interval != "1m":
            cache_file_1m = os.path.join(self.cache_dir, f"{symbol}_1m.csv")
            if os.path.exists(cache_file_1m):
                try:
                    df_1m = pd.read_csv(cache_file_1m)
                    ts_col = next(
                        (
                            c
                            for c in ["timestamp", "datetime", "date", "time"]
                            if c in df_1m.columns
                        ),
                        None,
                    )
                    if ts_col:
                        df_1m[ts_col] = pd.to_datetime(df_1m[ts_col])
                        df_1m = df_1m.set_index(ts_col)
                        df_1m.index = df_1m.index.tz_localize(None)
                        df_1m = df_1m.sort_index()

                        rule_map = {
                            "1m": "1min",
                            "3m": "3min",
                            "5m": "5min",
                            "15m": "15min",
                            "1d": "1D",
                        }
                        rule = rule_map.get(interval, interval.replace("m", "min"))

                        # Resample
                        df = df_1m.resample(rule).agg(
                            {"open": "first", "high": "max", "low": "min", "close": "last"}
                        )
                        if "volume" in df_1m.columns:
                            df["volume"] = df_1m["volume"].resample(rule).sum()

                        df = df.dropna()
                        cache_loaded = True

                        # Save the resampled cache so we don't have to resample next time
                        save_df = df.copy()
                        save_df.index.name = "timestamp"
                        save_df.reset_index().to_csv(cache_file, index=False)
                except Exception as e:
                    print(f"Error resampling 1m cache for {symbol}: {e}")
                    df = pd.DataFrame()

        if cache_loaded and not df.empty:
            cache_min = df.index.min()
            cache_max = df.index.max()

            # If the requested date range is completely within the cache date range, return slice
            # We compare .date() because req_start might be midnight but market data starts at 09:15
            # We add a 4-day buffer to cache_max to account for weekends and holidays!
            if (
                req_start.date() >= cache_min.date()
                and req_end.date() <= (cache_max + pd.Timedelta(days=4)).date()
            ):
                return df.loc[req_start:req_end].copy()
            else:
                # We need to fetch data. To keep it simple, just fetch the missing chunks or the whole requested range.
                # It's easiest to just fetch the whole requested range, merge it, and save it.
                needs_fetch = True
        else:
            needs_fetch = True

        if needs_fetch:
            print(f"[API Call] Fetching {symbol} from {start_date} to {end_date}...")
            new_df = self._api_fetch(symbol, exchange, interval, start_date, end_date)

            if new_df.empty:
                # If API fails but we have cache, return what we can from cache
                if cache_loaded and not df.empty:
                    return df.loc[req_start:req_end].copy()
                return pd.DataFrame()

            # Merge with existing cache
            if not df.empty:
                combined = pd.concat([df, new_df])
                # Drop duplicates based on index
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_df

            # Save back to cache
            # We want to save with the index as a column named 'timestamp'
            try:
                save_df = combined.copy()
                save_df.index.name = "timestamp"
                save_df.reset_index().to_csv(cache_file, index=False)
            except Exception as e:
                print(f"Error saving cache for {symbol}: {e}")

            return combined.loc[req_start:req_end].copy()

    def _api_fetch(self, symbol, exchange, interval, start_date, end_date):
        try:
            df = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )
            if isinstance(df, dict):
                print(f"API Error fetching {symbol}: {df}")
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            ts_col = next(
                (c for c in ["timestamp", "datetime", "date", "time"] if c in df.columns), None
            )
            if ts_col is not None:
                df[ts_col] = pd.to_datetime(df[ts_col])
                df = df.set_index(ts_col)

            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)
                except Exception:
                    pass

            df = df.sort_index()
            df = _ensure_ist(df)
            return df
        except Exception as e:
            print(f"Exception fetching {symbol}: {e}")
            return pd.DataFrame()
