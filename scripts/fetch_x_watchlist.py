#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'accounts.json'
OUTDIR = ROOT / 'data' / 'x'
OUTDIR.mkdir(parents=True, exist_ok=True)


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=120)


def ensure_xcli_env():
    hermes_env = Path.home() / '.hermes' / '.env'
    x_env_dir = Path.home() / '.config' / 'x-cli'
    x_env_dir.mkdir(parents=True, exist_ok=True)
    x_env = x_env_dir / '.env'
    if hermes_env.exists() and (not x_env.exists() or x_env.resolve() != hermes_env.resolve()):
        if x_env.exists() or x_env.is_symlink():
            x_env.unlink()
        x_env.symlink_to(hermes_env)


def main():
    ensure_xcli_env()
    cfg = json.loads(CONFIG.read_text())
    snapshot = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'accounts': []
    }
    for acct in cfg['accounts']:
        handle = acct['handle']
        entry = dict(acct)
        try:
            out = run(['x-cli', '-j', 'user', 'timeline', handle, '--max', '5'])
            tweets = json.loads(out)
            cleaned = []
            if isinstance(tweets, list):
                for t in tweets[:5]:
                    cleaned.append({
                        'id': t.get('id'),
                        'created_at': t.get('created_at'),
                        'text': t.get('text'),
                        'public_metrics': t.get('public_metrics', {}),
                        'conversation_id': t.get('conversation_id')
                    })
            entry['recent_tweets'] = cleaned
            entry['status'] = 'ok'
        except Exception as e:
            entry['status'] = 'error'
            entry['error'] = str(e)
        snapshot['accounts'].append(entry)

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    outpath = OUTDIR / f'watchlist_{ts}.json'
    outpath.write_text(json.dumps(snapshot, indent=2))
    latest = OUTDIR / 'latest.json'
    latest.write_text(json.dumps(snapshot, indent=2))
    print(str(outpath))


if __name__ == '__main__':
    main()
