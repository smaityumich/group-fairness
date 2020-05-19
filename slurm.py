import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

# Experiment 1
reg_w = [10, 20, 30, 40, 50, 60]
wrls = [1e-4, 1e-5, 5e-6, 1e-6]
eps = [0.1, 0.05, 0.01, 0.005, 0.001]


os.system('touch summary/adult3.out')



for reg_wasserstein, wlr, epsilon in itertools.product(reg_w, wrls, eps):
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=adult.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=2gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 adult_expt_gender_race.py {reg_wasserstein} {wlr} {epsilon} >> summary/adult3.out")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
