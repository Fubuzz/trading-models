#!/usr/bin/env python3
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'data' / 'alpha_vantage'
OUTDIR.mkdir(parents=True, exist_ok=True)
WATCHLIST = ['SPY','QQQ','XLE','SMH','NVDA','AMD','TSM','CCJ','DNN','NXE','GLD','COPX']
MIN_SECONDS_BETWEEN_CALLS = 12.5  # stay under Alpha Vantage free-tier limits


def load_key():
    env_path = Path.home() / '.hermes' / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('ALPHA_VANTAGE_API_KEY='):
                return line.split('=',1)[1].strip()
    return os.getenv('ALPHA_VANTAGE_API_KEY')


def fetch_quote(symbol, key, retries=3):
    params = urllib.parse.urlencode({
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': key,
    })
    url = 'https://www.alphavantage.co/query?' + params
    last_obj = {}
    for attempt in range(1, retries + 1):
        with urllib.request.urlopen(url, timeout=60) as r:
            obj = json.loads(r.read().decode())
        last_obj = obj
        if obj.get('Global Quote'):
            return obj['Global Quote']
        if obj.get('Note') or obj.get('Information'):
            if attempt < retries:
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)
                continue
        break
    return {'error': last_obj.get('Note') or last_obj.get('Information') or 'No quote returned'}


def main():
    key = load_key()
    if not key:
        raise SystemExit('Missing ALPHA_VANTAGE_API_KEY')
    snapshot = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'quotes': []
    }
    for i, sym in enumerate(WATCHLIST):
        if i:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS)
        q = fetch_quote(sym, key)
        snapshot['quotes'].append({
            'symbol': q.get('01. symbol', sym),
            'price': q.get('05. price'),
            'change_percent': q.get('10. change percent'),
            'latest_trading_day': q.get('07. latest trading day'),
            'volume': q.get('06. volume'),
            'error': q.get('error')
        })
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    outpath = OUTDIR / f'quotes_{ts}.json'
    outpath.write_text(json.dumps(snapshot, indent=2))
    latest = OUTDIR / 'latest.json'
    latest.write_text(json.dumps(snapshot, indent=2))
    print(str(outpath))


if __name__ == '__main__':
    main()
