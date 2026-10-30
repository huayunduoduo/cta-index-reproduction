from pathlib import Path
import hashlib, sys
root = Path(__file__).parent
bad = []
for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
    h, p = line.split('  ', 1)
    f = root / p
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
        bad.append(p)
print('Mismatches:', bad)
sys.exit(bool(bad))
