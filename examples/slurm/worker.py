import argparse
import json
import signal
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--resume-from")
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    state = {"count": 0}
    if args.resume_from is not None:
        resumed = read_json(Path(args.resume_from))
        if resumed is not None:
            state = resumed

    def save_checkpoint():
        write_json(checkpoint_path, state)

    def handle_term(signum, frame):
        del signum, frame
        save_checkpoint()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_term)

    while state["count"] < args.target_count:
        time.sleep(args.sleep_seconds)
        state["count"] = min(args.target_count, state["count"] + args.step)
        save_checkpoint()

    write_json(output_path, state)


if __name__ == "__main__":
    main()
