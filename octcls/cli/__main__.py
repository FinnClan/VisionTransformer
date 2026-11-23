from __future__ import annotations

import sys


def _usage() -> str:
    return (
        "Usage: python -m octcls.cli <command> [options]\n\n"
        "Commands:\n"
        "  train               Train a classifier (with optional YOLO ROI hints)\n"
        "  infer               Batch inference to CSV\n"
        "  gen-cam-labels      Generate YOLO labels from classifier CAMs\n"
        "  prepare-yolo-split  Create YOLO split + YAML (symlinks/copies)\n"
        "  train-yolo          Train YOLO (Ultralytics)\n"
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    # Map subcommands to existing CLI modules
    if cmd == "train":
        from . import train_classifier as mod
    elif cmd == "infer":
        from . import infer_classifier as mod
    elif cmd == "gen-cam-labels":
        from . import gen_cam_labels as mod
    elif cmd == "prepare-yolo-split":
        from . import prepare_yolo_split as mod
    elif cmd == "train-yolo":
        from . import train_yolo as mod
    else:
        print(f"Unknown command: {cmd}\n")
        print(_usage())
        return 1

    # Delegate parsing/execution to the selected module
    prog = f"{sys.argv[0]} {cmd}"
    sys.argv = [prog] + rest
    return mod.main()


if __name__ == "__main__":
    raise SystemExit(main())

