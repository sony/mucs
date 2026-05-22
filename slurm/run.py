import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--jobname", type=str, required=True)
parser.add_argument("--pyfile", type=str, required=True)
parser.add_argument("--logpath", type=str, default="pointer_to/logs/")
parser.add_argument("--partition", type=str, default="ds")
parser.add_argument("--account", type=str, default="ds")
parser.add_argument("--nnodes", type=int, default=1)
parser.add_argument("--nodelist", type=str, default="")
parser.add_argument("--ngpus", type=int, default=1)  # per node
parser.add_argument("--requeue", action="store_true")
parser.add_argument("--sbatch", action="store_true")
parser.add_argument("--time", type=int, default=-1)  # in minutes
parser.add_argument("--no_code_backup", action="store_true")
args, args_extra = parser.parse_known_args()

# Check logpath
user_supplied = False
for tmp in sys.argv:
    if tmp.startswith("--logpath"):
        user_supplied = True
        break
if not user_supplied:
    args.logpath = os.path.join(args.logpath, args.jobname)
if os.path.exists(args.logpath):
    rem = input("Remove contents in " + args.logpath + "? [y/N] ")
    if rem.lower() == "y":
        print("Erasing " + args.logpath + "...")
        os.system("rm -rf " + args.logpath)
os.makedirs(args.logpath, exist_ok=True)

# Code backup?
if not args.no_code_backup:
    print("Code backup...")
    os.system(
        'rsync -qr --exclude=".*" --no-links . '
        + os.path.join(args.logpath, "code_backup")
    )

# Define filenames
fn_slurm_launch = os.path.join(args.logpath, "slurm-launch.sh")
fn_slurm_out = os.path.join(args.logpath, "slurm-out.txt")
fn_slurm_err = os.path.join(args.logpath, "slurm-err.txt")

###################################################################################################
# Run
###################################################################################################

# Compose arguments
arguments = []
arguments += [f"--export ALL,OMP_NUM_THREADS=1"]
if not args.sbatch:
    arguments += [f"--unbuffered"]
arguments += [f"--job-name {args.jobname}"]
arguments += [f"--partition {args.partition}"]
if args.nodelist != "":
    arguments += [f"--nodelist {args.nodelist}"]
arguments += [f"--account {args.account}"]
arguments += [f"--nodes {args.nnodes}"]
arguments += [f"--gres gpu:{args.ngpus}"]
arguments += [f"--ntasks-per-node {max(1,args.ngpus)}"]
if args.time > 0:
    arguments += [f"--time {args.time}"]
if args.requeue:
    arguments += [f"--requeue"]

# Compose command
command = "python " + args.pyfile
command += " " + " ".join(args_extra)
command += f" slurm.jobname={args.jobname}"
command += f" slurm.logpath={args.logpath}"
command += f" slurm.nnodes={args.nnodes}"
command += f" slurm.ngpus={args.ngpus}"

if not args.sbatch:

    # Compose command
    print("Compose command...")
    cmd = "srun " + " ".join(arguments) + " " + command
    with open(fn_slurm_launch, "w") as f:
        f.write(cmd + "\n")

    # Execute srun
    print("Launch...")
    print("\n" + cmd + "\n")
    os.system(cmd)

else:

    # Write sbatch file
    print("Write sbatch file...")
    lines = []
    lines += ["#!/bin/bash"]
    lines += [""]
    for a in arguments:
        lines += ["#SBATCH " + a]
    lines += [f"#SBATCH --output {fn_slurm_out}"]
    lines += [f"#SBATCH --error {fn_slurm_err}"]
    lines += [""]
    lines += ['echo "Launch..."']
    lines += [f"srun {command}"]
    with open(fn_slurm_launch, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.system("chmod +x " + fn_slurm_launch)

    # Execute sbatch
    print(lines[-1])
    os.system("sbatch " + fn_slurm_launch)

    # Print log file
    print("Output file = " + fn_slurm_out)

###################################################################################################
