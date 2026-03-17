import subprocess

subprocess.run("which plumed", shell=True)
subprocess.run("plumed config module opes", shell=True)
subprocess.run("echo $PLUMED_KERNEL", shell=True)
subprocess.run("which gmx_mpi", shell=True)