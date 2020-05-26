import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

# Experiment 1
reg_w = [20, 40, 60, 80, 100]
wrls = [1e-2, 1e-3, 1e-4, 1e-5]
eps = [0.1, 0.01, 0.001]
lrs = [1e-3, 1e-4, 1e-5]


os.system('touch summary/adult7.out')



for reg_wasserstein, wlr, epsilon, lr in itertools.product(reg_w, wrls, eps, lrs):
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=adult.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=5\n')
        fh.writelines('#SBATCH --mem-per-cpu=2gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 experiment.py {lr} {wlr} {reg_wasserstein} {epsilon}")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
