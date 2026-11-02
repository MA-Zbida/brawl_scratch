"""Small run-artifact and operator-prompt helpers; safe before CUDA imports."""
from __future__ import annotations

import hashlib
import json
import time


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prompt(args, message):
    print(message, flush=True)
    if not args.no_prompt:
        input('Press Enter when configured, then switch to Brawlhalla. ')
    print(f'Starting in {args.delay:g}s...', flush=True)
    time.sleep(args.delay)
