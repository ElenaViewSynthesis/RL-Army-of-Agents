"""Read the ADR's canonical key classification without importing any SDK."""
import fnmatch
from pathlib import Path

ADR = Path(__file__).resolve().parent.parent / 'Architecture-decision/0002-wandb-tracking-contract.md'
CATEGORIES = {'config', 'metric', 'summary', 'table', 'artifact', 'weave', 'drop'}


def rules():
    result = []
    for line in ADR.read_text().splitlines():
        if line.startswith('| `'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            source, key = cells[0].strip('`').split(':', 1)
            result.append({'source': source, 'pattern': key, 'category': cells[1],
                           'namespace': cells[2], 'type': cells[3], 'unit': cells[4],
                           'null': cells[5], 'role': cells[6], 'owner': cells[7]})
    return result


def classify(source, key):
    matches = [r for r in rules() if r['source'] == source and fnmatch.fnmatchcase(key, r['pattern'])]
    if len(matches) != 1:
        raise ValueError(f'{source}:{key} has {len(matches)} contract classifications')
    return matches[0]['category']
