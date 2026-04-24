import sys

from teleoperation import main as teleop_main


if __name__ == "__main__":
    # Keep user-provided args, but enforce pickplace recording defaults.
    defaults = [
        "--mode",
        "task",
        "--task",
        "pickplace",
        "--record",
        "--dataset-dir",
        "dataset_raw_pickplace",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    teleop_main()
