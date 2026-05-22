import sys, os, time

FOLDER = sys.argv[1]
PRE = sys.argv[2].lower() == "pre"
IS_GLOBAL_ZERO = int(os.environ["SLURM_PROCID"]) == 0

if IS_GLOBAL_ZERO:
    print("Preparing folder " + FOLDER + " ...")
    if PRE:
        os.system("rm -rf " + FOLDER)
        os.system("mkdir " + FOLDER)
    else:
        if os.path.exists(FOLDER):
            os.system("rm -f " + os.path.join(FOLDER, "attrib-id*.*"))
            os.system("rm -f " + os.path.join(FOLDER, "*.yaml"))
            os.system("rm -f " + os.path.join(FOLDER, "*.ckpt"))
            os.system("rm -f " + os.path.join(FOLDER, "event*.*"))
            os.system("rm -f " + os.path.join(FOLDER, "sample*.*"))
        else:
            os.system("mkdir " + FOLDER)
    print("Ok to proceed.")
else:
    time.sleep(2)
