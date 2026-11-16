# Runs count.py while holding one of a few machine-wide slots, so many jobs do not start the slow,
# memory-heavy pipeline at once. It takes the same arguments as count.py. The slot is an OS file
# lock, so it is released automatically if the process dies.

import msvcrt
import os
import subprocess
import sys
import time

SLOTS = int(os.environ.get("COUNT_SLOTS", "3"))
HERE = os.path.dirname(os.path.abspath(__file__))
LOCKDIR = os.path.join(HERE, "..", "results", "_locks")
os.makedirs(LOCKDIR, exist_ok=True)


def acquire():
    waited = 0
    while True:
        for i in range(SLOTS):
            f = open(os.path.join(LOCKDIR, f"slot{i}.lock"), "a+")
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return f
            except OSError:
                f.close()
        if waited % 60 == 0:
            print(f"[run_locked] waiting for a free slot ({waited}s)", flush=True)
        time.sleep(5)
        waited += 5


if __name__ == "__main__":
    lk = acquire()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "count.py")] + sys.argv[1:])
    finally:
        lk.seek(0)
        msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)
        lk.close()
    sys.exit(r.returncode)
