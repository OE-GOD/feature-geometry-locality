"""
Local validation / unit test for the locality metric.

Runs the metric on three ground-truth worlds and asserts it recovers the
known answer. This is the "validate before you burn pod compute" gate:
if this fails, the metric is broken and the real-model run is meaningless.

Run:  python3 validate.py
"""

import sys
from locality import locality_metric, format_result
from worlds import world_local, world_global, world_null

EXPECT = {
    "LOCAL": ("world_local", world_local, "LOCAL"),
    "GLOBAL": ("world_global", world_global, "GLOBAL"),
    "NULL": ("world_null", world_null, "NULL"),
}


def main():
    print("Validating locality metric against ground-truth worlds")
    print("=" * 70)
    ok = True
    # multiple seeds so we aren't fooled by one lucky draw
    for seed in range(3):
        print(f"\nseed={seed}")
        for _, (name, gen, expected) in EXPECT.items():
            D, F = gen(seed=seed)
            res = locality_metric(D, F)
            got = res["verdict"]
            mark = "ok " if got == expected else "XX "
            if got != expected:
                ok = False
            print(f"  {mark}{format_result(name, res)}  expected={expected}")

    print("\n" + "=" * 70)
    if ok:
        print("PASS: metric recovers local / global / null on all seeds.")
        return 0
    print("FAIL: metric mislabeled at least one world. Do NOT run on pod.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
