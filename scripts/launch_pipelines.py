import sys, os, time, random

print(
    f"Usage: python {sys.argv[0]} <datasets> seed1-seedN <attrib> attr1-attrM <partition> [--execute]"
)

# Proc input
if "," in sys.argv[1]:
    NAMEDATASETS = sys.argv[1].split(",")
else:
    NAMEDATASETS = [sys.argv[1]]
if "-" in sys.argv[2]:
    lo, hi = sys.argv[2].split("-")
    SEEDS = list(range(int(lo), int(hi) + 1))
else:
    SEEDS = [int(sys.argv[2])]
NAMEATTRIB = sys.argv[3]
if "-" in sys.argv[4]:
    lo, hi = sys.argv[4].split("-")
    NUMS = list(range(int(lo), int(hi) + 1))
else:
    NUMS = [int(sys.argv[4])]
PARTITION = sys.argv[5]
EXECUTE = len(sys.argv) > 6 and sys.argv[6] == "--execute"
SLEEPTIME = 2.0

# Create call
CALL = "python pipeline.py --what=<pre>a,tpost,gpost --job=;;; --seed=??? --attrib=<nameattrib>&&& --sbatch"
if NAMEATTRIB == "random":
    CALL = CALL.replace("<pre>", "tpre,gpre,")
else:
    CALL = CALL.replace("<pre>", "")
CALL = CALL.replace("<nameattrib>", NAMEATTRIB)
if PARTITION == "sfxfm":
    CALL += " slurm:account=sfxfm slurm:partition=sfxfm"
elif PARTITION == "ds_l40s":
    CALL += " slurm:partition=ds_l40s"
elif PARTITION == "sfxfm_l40s":
    CALL += " slurm:account=sfxfm slurm:partition=sfxfm_l40s"

# Info and run
print()
print("CALL =", CALL)
print("SEEDS =", SEEDS)
print("NUMS =", NUMS)
print("EXECUTE =", EXECUTE)
print()
input(f'You should have all "{NAMEATTRIB}&&&.py" ready. Ready? ')

i = 1
for d in NAMEDATASETS:
    for n in NUMS:
        for s in SEEDS:
            c = CALL.replace(";;;", d)
            c = c.replace("???", str(s))
            if n == 0:
                c = c.replace("&&&", "")
            else:
                c = c.replace("&&&", str(n))
            print("-" * 80)
            print(i, "\t", c)
            print("-" * 80)
            if EXECUTE:
                os.system(c)
            time.sleep(SLEEPTIME + random.uniform(0, SLEEPTIME / 2))
            i += 1

print("Done!")
