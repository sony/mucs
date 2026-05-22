import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--jobname", type=str, required=True)
parser.add_argument("--path", type=str, default="logs/current/", required=True)
parser.add_argument("--nlines", type=int, default=15, required=False)
parser.add_argument("--err", action="store_true")
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
print(args)
print()

if args.path[-1] != "/":
    args.path += "/"

if args.full:
    cmd = "cat"
else:
    cmd = "tail"
    cmd += " -" + str(args.nlines)
outerr = "err" if args.err else "out"
cmd += " " + args.path + args.jobname + "/slurm-train-" + outerr + ".txt"

os.system(cmd)
print()
