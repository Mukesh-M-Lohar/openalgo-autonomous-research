from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate a bot script from the template.")
    parser.add_argument("symbol", type=str, help="Stock or index symbol")
    parser.add_argument(
        "--exchange",
        type=str,
        default="NSE",
        help="Exchange code (e.g. NSE, NSE_INDEX)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Timeframe interval (e.g. 1m, 5m, 15m, D)",
    )
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name")
    parser.add_argument(
        "--whatsapp",
        type=str,
        default=None,
        help="WhatsApp broadcast numbers (comma-separated)",
    )

    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    exchange = args.exchange.strip().upper()
    interval = args.interval.strip()
    strategy = args.strategy
    whatsapp = args.whatsapp

    if not strategy:
        strategy = f"{symbol}_Strategy_v1"
    else:
        strategy = strategy.strip()

    if not symbol.isalnum() and "_" not in symbol:
        print(
            f"Error: Invalid stock symbol '{symbol}'. Must contain alphanumeric characters or underscores."
        )
        sys.exit(1)

    template_path = Path("/root/openalgo-autonomous-research/scripts/bot/strategy_template.py")
    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)

    # Output file path
    output_filename = f"{symbol.lower()}_bot.py"
    output_path = Path("/root/openalgo-autonomous-research/scripts/bot") / output_filename

    try:
        content = template_path.read_text(encoding="utf-8")

        # Define replacements for configs
        replacements = {
            'SYMBOL = os.getenv("SYMBOL", "NIFTY")': f'SYMBOL = os.getenv("SYMBOL", "{symbol}")',
            'EXCHANGE = os.getenv("EXCHANGE", "NSE_INDEX")': f'EXCHANGE = os.getenv("EXCHANGE", "{exchange}")',
            'CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "15m")': f'CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "{interval}")',
            'STRATEGY_NAME = os.getenv("STRATEGY_NAME", "MyStrategy_v1")': f'STRATEGY_NAME = os.getenv("STRATEGY_NAME", "{strategy}")',
        }

        if whatsapp:
            replacements[
                'for n in os.getenv("WHATSAPP_NUMBERS", "919566029048,919790856795").split(",")'
            ] = f'for n in os.getenv("WHATSAPP_NUMBERS", "{whatsapp.strip()}").split(",")'

        # Apply replacements
        for old, new in replacements.items():
            if old not in content:
                print(
                    f"Warning: Configuration line '{old}' not found in template. Skipping replacement."
                )
            content = content.replace(old, new)

        # Write to the new bot file
        output_path.write_text(content, encoding="utf-8")
        print(f"🎉 Success! Generated bot for {symbol} at:")
        print(f"   {output_path}")
        print(
            f"   Configs: symbol={symbol}, exchange={exchange}, interval={interval}, "
            f"strategy={strategy}"
        )
        if whatsapp:
            print(f"   WhatsApp alerts: {whatsapp.strip()}")
        print("\nTo run the generated bot:")
        print(f"   .venv/bin/python scripts/bot/{output_filename}")

    except Exception as e:
        print(f"Error generating bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
