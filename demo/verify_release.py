"""Read-only deployment preflight: provisional or stale thresholds fail."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import inference as inf


def main():
    thresholds = inf.load_thresholds()
    for name in inf.MODELS:
        if thresholds[name]['status'] != 'validation':
            raise RuntimeError(f'{name} has a provisional threshold; release blocked')
        inf.load_model(name)
    print('Release artifacts verified. Local startup/upload checks must also pass before deployment.')


if __name__ == '__main__':
    main()
