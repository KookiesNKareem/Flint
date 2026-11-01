"""Display a remote_run's real receipts in the editor terminal (no model calls)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # A writer may still be appending this line.
    return result


def watch(root, replay=False):
    seen = set()
    print(('Recorded run' if replay else 'Live run') + ': ' + str(root), flush=True)
    while True:
        streams = [root / 'training.log', *sorted(root.glob('reference_*/run/events.jsonl'))]
        pending = []
        for path in streams:
            for index, event in enumerate(records(path)):
                key = (str(path), index)
                if key not in seen:
                    pending.append((path, index, event))
        # Generation receipts precede the training handoff on a completed replay.
        pending.sort(key=lambda item: 0 if item[2].get('phase') == 'captured' else
                     1 if item[0].name == 'events.jsonl' else 2)
        complete = False
        for path, index, event in pending:
            seen.add((str(path), index))
            kind = event.get('event', event.get('phase'))
            if kind == 'captured':
                print('Captured source: ' + event['hash'], flush=True)
            elif kind == 'llm_start':
                print('Generating ' + event['tag'] + ' ...', flush=True)
            elif kind == 'llm_done':
                print(f"  {Path(event['path']).name} returned in {event['seconds']:.1f}s", flush=True)
            elif kind == 'eval':
                counts = dict(Counter(case['outcome'] for case in event['results']))
                mode = 'forward + gradients' if event.get('grads') else 'forward'
                print(f"  {Path(event['path']).name} / {mode}: {event['outcome']} {counts}", flush=True)
                failed = next((case for case in event['results'] if case['outcome'] not in ('accept', 'unsupported', 'skip')), None)
                if failed:
                    reason = ' '.join(str(failed.get('reason', '')).split())
                    print('  First failure: ' + failed['name'] + ' — ' + reason[:180], flush=True)
            elif 'step' in event and (event['step'] % 5 == 0 or event['step'] == 29):
                print(f"Train {event['step']+1:2d}/30  loss {event['loss']:.4f}  "
                      f"{event['step_ms']:.1f} ms  calls={event['calls']}", flush=True)
            elif kind == 'complete':
                for file in root.glob('reference_*/holdout_results.json'):
                    result = json.loads(file.read_text())
                    coverage = result.get('coverage', {})
                    print(f"Fresh holdout: {coverage.get('accept', 0)}/{coverage.get('total', 0)} accepted", flush=True)
                status = event['status']
                print('Kernel SHA256: ' + status['kernel_sha256'], flush=True)
                print(f"Complete: {status['state']}  calls={status['calls']}  model estimate=${status.get('usd', 0):.5f}", flush=True)
                complete = True
        if replay or complete:
            return
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    watch(args.run, args.replay)
