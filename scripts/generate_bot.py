from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Error: No stock symbol specified.")
        print("Usage: python scripts/generate_bot.py <STOCK_SYMBOL>")
        sys.exit(1)

    symbol = sys.argv[1].strip().upper()
    if not symbol.isalnum():
        print(
            f"Error: Invalid stock symbol '{symbol}'. Must contain alphanumeric characters only."
        )
        sys.exit(1)

    template_path = Path(
        "/root/openalgo-autonomous-research/scripts/bot/strategy_template.py"
    )
    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)

    # Output file path
    output_filename = f"{symbol.lower()}_bot.py"
    output_path = (
        Path("/root/openalgo-autonomous-research/scripts/bot") / output_filename
    )

    try:
        content = template_path.read_text(encoding="utf-8")

        # Replace default instrument configuration
        old_symbol_line = 'SYMBOL = os.getenv("SYMBOL", "NIFTY")'
        new_symbol_line = f'SYMBOL = os.getenv("SYMBOL", "{symbol}")'

        old_exchange_line = 'EXCHANGE = os.getenv("EXCHANGE", "NSE_INDEX")'
        new_exchange_line = 'EXCHANGE = os.getenv("EXCHANGE", "NSE")'

        if old_symbol_line not in content or old_exchange_line not in content:
            print(
                "Warning: Template structure has changed. Attempting simple text replacements..."
            )

        content = content.replace(old_symbol_line, new_symbol_line)
        content = content.replace(old_exchange_line, new_exchange_line)

        # Write to the new bot file
        output_path.write_text(content, encoding="utf-8")
        print(f"🎉 Success! Generated bot for {symbol} at:")
        print(f"   {output_path}")
        print("\nTo run the generated bot:")
        print(f"   .venv/bin/python scripts/bot/{output_filename}")

    except Exception as e:
        print(f"Error generating bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
