#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
X_PATH = ROOT / 'data' / 'x' / 'latest.json'
A_PATH = ROOT / 'data' / 'alpha_vantage' / 'latest.json'
OUTDIR = ROOT / 'reports'
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    x = load_json(X_PATH) or {'accounts': []}
    a = load_json(A_PATH) or {'quotes': []}
    lines = []
    lines.append(f"Hermes X + Alpha Vantage Brief")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append('')
    lines.append('Top X Accounts Monitored:')
    for acct in x.get('accounts', [])[:10]:
        latest = acct.get('recent_tweets', [])[:1]
        if latest:
            txt = latest[0].get('text','').replace('\n', ' ').strip()
            txt = (txt[:180] + '...') if len(txt) > 180 else txt
            lines.append(f"- @{acct['handle']}: {txt}")
        else:
            lines.append(f"- @{acct['handle']}: no recent tweet captured")
    lines.append('')
    lines.append('Market Watchlist Quotes:')
    for q in a.get('quotes', []):
        lines.append(f"- {q['symbol']}: {q['price']} ({q['change_percent']}) day={q['latest_trading_day']}")
    out = '\n'.join(lines) + '\n'
    outpath = OUTDIR / 'daily_brief.txt'
    outpath.write_text(out)
    print(str(outpath))


if __name__ == '__main__':
    main()
