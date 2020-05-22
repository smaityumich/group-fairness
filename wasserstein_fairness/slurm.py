import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

# Experiment 1
alpha_vec = [0, 0.5]
beta_vec = [1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100]
lr_vec = [1e-4, 1e-3, 1e-2, 1e-1]


os.system('touch summary/adult-wfm3.out')



for alpha, beta, lr in itertools.product(alpha_vec, beta_vec, lr_vec):
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=adult.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=20\n')
        fh.writelines('#SBATCH --mem-per-cpu=1gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 adult_expt.py {alpha} {beta} {lr}")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
