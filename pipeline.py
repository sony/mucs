import sys, os, argparse

EXECUTE_CALLS = True
NEED_TRAK_ENV = ("dtrak", "das")

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--what", type=str, default=None, required=True)
parser.add_argument("--job", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=None, required=True)
parser.add_argument("--attrib", type=str, default=None, required=True)
parser.add_argument("--subsample", type=int, default=0, required=False)
parser.add_argument("--num_attrib", type=int, default=20, required=False)
parser.add_argument("--control_factor", type=float, default=1.0, required=False)
parser.add_argument("--topk_factor", type=float, default=2.0, required=False)
parser.add_argument("--logpath", type=str, default="pointer_to/logs/", required=False)
parser.add_argument("--use_last_ckpt", action="store_true")
parser.add_argument("--no_exec", action="store_true")
parser.add_argument("--sbatch", action="store_true")
parser.add_argument("--ngpus", type=int, default=1, required=False)
args, args_extra = parser.parse_known_args()

# Check what to do
args.what = args.what.split(",")
for i, aux in enumerate(args.what):
    if aux not in ("tpre", "gpre", "a", "tpost", "gpost"):
        print(f'ERROR: Unknown instruction "{aux}"')
        sys.exit()

# Check extra arguments
for ae in args_extra:
    if not ae.startswith("pipe:") and not ae.startswith("slurm:"):
        print(
            'ERROR: Cannot pass extra arguments other than "slurm:xxx=yyy" and "pipe:xxx=yyy"'
        )
        sys.exit()

# Set num_removal and num_gen depending on data set
if "cifar10" in args.job:
    args.num_removal = int(500 * args.topk_factor)
elif "artbench10" in args.job:
    args.num_removal = int(499 * args.topk_factor)
elif "celeba" in args.job:
    args.num_removal = int(1628 * args.topk_factor)
elif "wikiart" in args.job:
    args.num_removal = int(736 * args.topk_factor)
elif "coco" in args.job:
    args.num_removal = int(1183 * args.topk_factor)
else:
    raise NotImplementedError
args.num_gen = args.num_attrib + int(args.num_attrib * args.control_factor)

# Define filenames
confname = args.job
jobname_base = f"{args.job}_s{args.seed}_k{args.topk_factor}_na{args.num_attrib}_sb{args.subsample}"
jobname = f"{args.attrib}_{jobname_base}"
logpath_pre = os.path.join(args.logpath, f"{jobname_base}--pre")
logpath_post = os.path.join(args.logpath, f"{jobname_base}--post_{args.attrib}")
fn_samples_pre = f"{logpath_pre}/samples.pt"
fn_samples_post = f"{logpath_post}/samples.pt"
fn_attributed = f"{logpath_post}/attrib-ids.pt"
fn_sh = os.path.join(args.logpath, "tmp", f"{jobname}.sh")
os.makedirs(os.path.split(fn_sh)[0], exist_ok=True)


###############################################################################
# Helper functions
###############################################################################


def finalize(call, preorpost, in_trak_env=False):
    if preorpost == "pre":
        logpath = logpath_pre
    else:
        logpath = logpath_post
    call += f" slurm.jobname={jobname} slurm.logpath={logpath} slurm.nnodes=1 slurm.ngpus={args.ngpus}"
    call += f" seed={args.seed}"
    for ae in args_extra:
        if ae.startswith("pipe:"):
            nameval = ae.split(":")[-1]
            if not nameval.startswith("seed="):
                call += f" {nameval}"
    if in_trak_env:
        return "/bin/bash scripts/run_in_trak_env.sh " + call
    return call


def get_arg_checkpoint(preorpost):
    if preorpost == "pre":
        logpath = logpath_pre
    elif preorpost == "post":
        logpath = logpath_post
    else:
        raise NotImplementedError
    if args.use_last_ckpt:
        return f" --checkpoint={logpath}/checkpoint_last.ckpt"
    else:
        return f" --checkpoint={logpath}/checkpoint_best.ckpt"


###############################################################################
# Construct .sh lines
###############################################################################

lines = ["#! /bin/bash"]
for instruct in args.what:

    # Train 1st model
    if instruct == "tpre":
        # lines.append("rm -rf " + logpath_pre)
        # lines.append("mkdir " + logpath_pre)
        lines.append("python scripts/prepare_folder.py " + logpath_pre + " pre")
        call = f"python train.py --conf=configs/{confname}.yaml"
        if args.subsample > 0:
            call += f" data.subsample_train={args.subsample}"
        call = finalize(call, "pre")
        lines.append(call)

    # 1st generation
    if instruct == "gpre":
        call = f"python generate.py" + get_arg_checkpoint("pre")
        call += f" --fn_out={fn_samples_pre}"
        call += f" --num_gen={args.num_gen}"
        call = finalize(call, "pre")
        lines.append(call)

    # Attribution
    if instruct == "a":
        # lines.append("rm -rf " + logpath_post)
        # lines.append("mkdir " + logpath_post)
        lines.append("python scripts/prepare_folder.py " + logpath_post + " post")
        call = f"python attrib.py --method={args.attrib}" + get_arg_checkpoint("pre")
        call += f" --fn_samples={fn_samples_pre}"
        call += f" --fn_out={fn_attributed}"
        call += f" --num_attrib={args.num_attrib} --top_k_attrib={args.num_removal}"
        call = finalize(call, "pre", in_trak_env=args.attrib in NEED_TRAK_ENV)
        lines.append(call)

    # Train 2nd model
    if instruct == "tpost":
        call = f"python train.py --conf=configs/{confname}.yaml"
        call += f" data.remove_selected_datapoints={fn_attributed}"
        if args.subsample > 0:
            call += f" data.subsample_train={args.subsample}"
        call = finalize(call, "post")
        lines.append(call)

    # 2nd generation
    if instruct == "gpost":
        call = f"python generate.py" + get_arg_checkpoint("post")
        call += f" --pre_seeds={fn_samples_pre}"
        call += f" --fn_out={fn_samples_post}"
        call += f" --num_gen={args.num_gen}"
        call = finalize(call, "post")
        lines.append(call)


###############################################################################
# Finish
###############################################################################

# Create slurm arguments
arguments = []
arguments += ["--export ALL,OMP_NUM_THREADS=1,CUBLAS_WORKSPACE_CONFIG=:4096:8"]
if not args.sbatch:
    arguments += ["--unbuffered"]
arguments += [f"--job-name {jobname}"]
overrided_names = []
for ae in args_extra:
    if ae.startswith("slurm:"):
        name, val = ae.split(":")[-1].split("=")
        arguments += [f"--{name} {val}"]
        overrided_names.append(name)
if "partition" not in overrided_names:
    arguments += ["--partition ds"]
if "account" not in overrided_names:
    arguments += ["--account ds"]
arguments += ["--nodes 1"]
if "gres" not in overrided_names:
    arguments += [f"--gres gpu:{args.ngpus}"]
if "ntasks-per-node" not in overrided_names:
    arguments += [f"--ntasks-per-node {args.ngpus}"]

# Prepare sh file
if args.sbatch:
    header = ["#!/bin/bash"]
    for a in arguments:
        header += ["#SBATCH " + a]
    header += [f"#SBATCH --output {fn_sh}.out"]
    header += [f"#SBATCH --error {fn_sh}.err"]
    lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = "srun " + lines[i]
    lines = header + lines

# Write sh
with open(fn_sh, "w") as f:
    for line in lines:
        f.write(line + "\n")
os.system("chmod +x " + fn_sh)

# Prepare command
if args.sbatch:
    cmd = "sbatch " + fn_sh
else:
    cmd = "srun " + " ".join(arguments) + " " + fn_sh

# Print all & execute
print()
print("--> This is the content of " + fn_sh + ":")
print()
os.system("cat " + fn_sh)
print()
print("--> This is the proposed slurm line:")
print()
print(cmd)
print()
if not args.no_exec:
    print("--> Execute")
    print()
    os.system(cmd)
    print()
