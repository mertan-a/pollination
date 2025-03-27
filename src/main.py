'''
this script 
- uses argparse to determine the jobs to run 
    a either runs the job 
    b1 writes the .sh file for each job
    b2 submits the jobs
    b3 deletes the .sh files
'''
import os
import time
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser(description='run jobs')

# sim
parser.add_argument('--sim_sif', '-ss', type=str, default='evogym_numba.sif', 
                    choices=['evogym_numba.sif'], help='simulation sif')
# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['Walker-v0'], default='Walker-v0')
# experiment related arguments
parser.add_argument('-ros', '--run_or_slurm', type=str,
                    help='run the job directly or submit it to slurm', choices=['slurm', 'run'])
parser.add_argument('-sq', '--slurm_queue', type=str,
                    help='choose the job queue for the slurm submissions', choices=['short', 'week', 'bluemoon'])
parser.add_argument('-cpu', '--cpu', type=int, 
                    help='number of cpu cores requested for slurm job')
parser.add_argument('-sp', '--saving_path', help='path to save the experiment')
parser.add_argument('-n', '--n_jobs', type=int, default=1,
                    help='number of jobs to submit')
parser.add_argument('-jt', '--job_type', type=str, 
                    help='job type', choices=['qd'])
parser.add_argument('-nm', '--no_migration', 
                    action='store_true', help="don't allow migrations to occur in map elites")
parser.add_argument('-nmsg', '--no_migration_start_gen', type=int,
                    help='generation to start border control')
parser.add_argument('-ab', '--archive_bins', nargs='+', type=int,
                    help='archive is assumed to be a 2D grid, specify the number of bins for each dimension. first dimension is num active voxels, second dimension is num voxels')
parser.add_argument('-p', '--pollination', action='store_true',
                    help='do pollination')
parser.add_argument('-pf', '--pollination_frequency', type=int,
                    help='every how many generations pollination should occur')
parser.add_argument('-prs', '--pollination_replacement_strategy', type=str,
                    help='pollination replacement strategy', choices=['RANDOM', 'BEST', 'WORST', 'LOCAL'])
parser.add_argument('-plr', '--pollination_local_radius', type=int, 
                    help='local radius (moore neighborhood) for pollination')
parser.add_argument('-pr', '--pollination_ratio', type=float,
                    help='ratio of the population to pollinate')
parser.add_argument('--re_eval_pollinated', action='store_true',
                    help='re evaluate the fitness of the pollinated individuals')
parser.add_argument('--replace_only_if_similar', action='store_true',
                    help='replace the controller only if the distilled controller is similar in fitness i.e. within 10%')
parser.add_argument('--num_distillation_epochs', type=int, 
                    help='number of epochs for distillation')
# evolutionary algorithm related arguments
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')
# softrobot related arguments
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int,
                    help='Bounding box dimensions (x,y). e.g.IND_SIZE=(6, 6)->workspace is a rectangle of 6x6 voxels') # trying to get rid of this
# controller
parser.add_argument('--controller', '-ctrl', help='specify the controller',
                    choices=['CENTRALIZED', 'CENTRALIZED_BIG'], default='CENTRALIZED')
#parser.add_argument('--observation_range', '-or', type=int,
#                     help='observation range')
parser.add_argument('--observe_structure', '-ostr', action='store_true')
parser.add_argument('--observe_voxel_volume', '-ovol', action='store_true')
parser.add_argument('--observe_voxel_speed', '-ospd', action='store_true')
parser.add_argument('--sparse_acting', '-sa', action='store_true')
parser.add_argument('--act_every', '-ae', type=int)
# body
parser.add_argument('--body_representation', '-br', type=str,
                    help='specify the body representation', choices=['CPPN', 'DIRECT'])


parser.add_argument('-r', '--repetition', type=int,
                    help='repetition number, dont specify this if you want it to be determined automatically', nargs='+')
parser.add_argument('-id', '--id', type=str,
                    help='id of the job, dont specify this if you want no specific id for the jobs', nargs='+')

# testing
parser.add_argument('--path_to_ind', '-pti', help='path to the indivuduals pkl file')
parser.add_argument('--output_path', '-op', help='path to the gif file')
parser.add_argument('--gif_every', '-ge', type=int, default=50)

parser.add_argument('--verbose', '-v', action='store_true')

parser.add_argument('--do_post_job_processing', '-dpjp', action='store_true')
parser.add_argument('--local', '-l', action='store_true', help='set this to true to if you are running the job locally (as opposed to on slurm)')

args = parser.parse_args()

def run(args):
    import random
    import numpy as np
    import multiprocessing
    import time
    import traceback

    from utils import prepare_rundir
    from population import ARCHIVE
    from algorithms import MAP_ELITES
    from make_gif import MAKEGIF
    import settings
    import torch
    from post_job import post_job_processing

    multiprocessing.set_start_method('spawn')

    # save the start time
    settings.START_TIME = time.time()

    # set verbosity
    settings.VERBOSE = args.verbose

    # run the job directly
    if args.repetition is None:
        args.repetition = [1]
    rundir = prepare_rundir(args)
    args.rundir = rundir
    print('rundir', rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING'):
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.repetition[0]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setting up the optimization algorithm and runnning
    map_elites = MAP_ELITES(args=args, archive=ARCHIVE(args=args))
    if args.local:
        try:
            map_elites.optimize()
        except Exception as e:
            print(traceback.format_exc())
            # delete running file
            if os.path.isfile(args.rundir + '/RUNNING'):
                os.remove(args.rundir + '/RUNNING')
                print('running file deleted')
            exit()
    else:
        map_elites.optimize()

    # delete running file in any case
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # if the job is finished, create a finished file
    if not settings.STOP:
        if args.do_post_job_processing == False and args.local == False:
            print('job finished successfully')
            args.do_post_job_processing = True
            # resubmitting it for one last time
            submit_slurm(args, resubmit=True)
        else:
            print('post job processing')
            post_job_processing(args)
            # write a file to indicate that the job finished successfully
            with open(args.rundir + '/FINISHED', 'w') as f:
                pass
    else:
        # if the job is not finished, resubmit it
        print('resubmitting job')
        submit_slurm(args, resubmit=True)

def submit_slurm(args, resubmit=False):
    # submit the job to slurm
    base_string = '#!/bin/sh\n\n'
    base_string += '#SBATCH --partition=' + args.slurm_queue + ' # Specify a partition \n\n'
    base_string += '#SBATCH --nodes=1  # Request nodes \n\n'
    if args.cpu is None:
        if args.nr_parents < 50:
            base_string += '#SBATCH --ntasks=' + str(args.nr_parents*2) + '  # Request some processor cores \n\n'
        else:
            base_string += '#SBATCH --ntasks=100 # Request some processor cores \n\n'
    else:
        base_string += '#SBATCH --ntasks=' + str(args.cpu) + '  # Request some processor cores \n\n'
    base_string += '#SBATCH --job-name=evogym  # Name job \n\n'
    base_string += '#SBATCH --signal=B:USR1@600  # signal the bash \n\n'
    base_string += '#SBATCH --output=outs/%x_%j.out  # Name output file \n\n'
    base_string += '#SBATCH --mail-user=alican.mertan@uvm.edu  # Set email address (for user with email "usr1234@uvm.edu") \n\n'
    base_string += '#SBATCH --mail-type=FAIL   # Request email to be sent at begin and end, and if fails \n\n'
    base_string += '#SBATCH --mem-per-cpu=10GB  # Request 16 GB of memory per core \n\n'
    if args.slurm_queue == 'short':
        base_string += '#SBATCH --time=0-02:30:00  # Request 2 hours and 30 minutes of runtime \n\n'
    elif args.slurm_queue == 'week':
        base_string += '#SBATCH --time=3-00:00:00  # Request 3 days of runtime \n\n'
    elif args.slurm_queue == 'bluemoon':
        base_string += '#SBATCH --time=0-27:00:00  # Request 16 hours of runtime \n\n'#TODO: this isn't the correct time

    base_string += 'cd /users/a/m/amertan/workspace/EVOGYM/evogym-qd-investigation/ \n'
    base_string += 'spack load singularity@3.7.1\n'
    base_string += 'trap \'kill -INT "${PID}"; wait "${PID}"; handler\' USR1 SIGINT SIGTERM \n'
    base_string += 'singularity exec --bind ../../../scratch/evogym_experiments:/scratch_evogym_experiments ' + args.sim_sif + ' xvfb-run -a python3 '

    # for each job
    for i in range(args.n_jobs):
        # create a string to write to the .sh file
        string_to_write = base_string
        string_to_write += 'main.py '
        # iterate over all of the arguments
        dict_args = deepcopy(vars(args))
        # handle certain arguments differently
        if 'run_or_slurm' in dict_args:
            dict_args['run_or_slurm'] = 'run'
        if 'n_jobs' in dict_args:
            dict_args['n_jobs'] = 1
        if 'repetition' in dict_args:
            if dict_args['repetition'] is None:
                dict_args['repetition'] = i+1
            else:
                dict_args['repetition'] = dict_args['repetition'][i]
        if 'rundir' in dict_args: # rundir might be in, delete it
            del dict_args['rundir']
        # write the arguments
        for key in dict_args:
            # key can be None, skip it in that case
            if dict_args[key] is None:
                continue
            # if the key is a list, we need to iterate over the list
            if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
                string_to_write += '--' + key + ' '
                for item in dict_args[key]:
                    string_to_write += str(item) + ' '
            elif isinstance(dict_args[key], bool):
                if dict_args[key]:
                    string_to_write += '--' + key + ' '
            else:
                string_to_write += '--' + key + ' ' + str(dict_args[key]) + ' '
        # job submission or resubmission
        if resubmit == False: # this process can call sbatch since it is not in container
            # write to the file
            with open('job.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')
            # submit the job
            os.system('sbatch job.sh')
            # sleep for a second
            time.sleep(0.1)
            # remove the job file
            os.remove('job.sh')
            # sleep for a second
            time.sleep(0.1)
        else: # this process is in container, so it cannot call sbatch. there is shell script running that will check for sh files and submit them
            import random
            import string
            # write to the file
            with open('resubmit_'+''.join(random.choices(string.ascii_lowercase, k=5))+'.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')


if __name__ == '__main__':

    # sanity checks
    if args.archive_bins is None:
        raise ValueError('you need to specify the archive bins')
    if args.nr_generations is None:
        raise ValueError('you need to specify the number of generations')
    if args.nr_parents is None:
        raise ValueError('you need to specify the number of parents')
    if args.body_representation is None:
        raise ValueError('you need to specify the body representation')
    if args.bounding_box is None:
        raise ValueError('you need to specify the bounding box')
    if args.nr_random_individual is None:
        raise ValueError('you need to specify the number of random individuals')
    if args.no_migration:
        if args.no_migration_start_gen is None:
            raise ValueError('you need to specify the generation to start border control')
    if args.observe_structure is False and args.observe_voxel_volume is False and args.observe_voxel_speed is False:
        raise ValueError('you need to specify at least one observation type')
    if args.pollination:
        if args.pollination_frequency is None:
            raise ValueError('you need to specify the pollination frequency')
        if args.pollination_replacement_strategy is None:
            raise ValueError('you need to specify the pollination replacement strategy')
        if args.pollination_replacement_strategy == 'LOCAL' and args.pollination_local_radius is None:
            raise ValueError('you need to specify the pollination local radius for local replacement strategy')
        if args.pollination_ratio is None:
            raise ValueError('you need to specify the pollination ratio')
        if args.num_distillation_epochs is None:
            raise ValueError('you need to specify the number of distillation epochs')

    if args.run_or_slurm == 'run':
        # run the job
        run(args)
    elif args.run_or_slurm == 'slurm':
        # submit the job
        submit_slurm(args)
    else:
        raise ValueError('run_or_slurm must be either run or slurm')





