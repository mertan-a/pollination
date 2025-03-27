import numpy as np
import os
import shutil
from copy import deepcopy
import re

import settings

# filesystem related
def prepare_rundir(args):

    # decide where the experiment will be saved
    if args.saving_path is None:
        run_dir = "experiments/"
    else:
        # if saving path starts with '/', then it is an absolute path
        if args.saving_path[0] == '/':
            args.saving_path = os.path.relpath(args.saving_path)
        run_dir = args.saving_path
        # make sure there is a trailing slash
        if not run_dir.endswith("/"):
            run_dir += "/"
    dict_args = deepcopy(vars(args))

    # remove the arguments that are not relevant for the run_dir
    if 'run_or_slurm' in dict_args:
        dict_args.pop('run_or_slurm')
    if 'slurm_queue' in dict_args:
        dict_args.pop('slurm_queue')
    if 'cpu' in dict_args:
        dict_args.pop('cpu')
    if 'saving_path' in dict_args:
        dict_args.pop('saving_path')
    if 'n_jobs' in dict_args:
        dict_args.pop('n_jobs')
    if 'path_to_ind' in dict_args:
        dict_args.pop('path_to_ind')
    if 'output_path' in dict_args:
        dict_args.pop('output_path')
    if 'gif_every' in dict_args:
        dict_args.pop('gif_every')
    if 'do_post_job_processing' in dict_args:
        dict_args.pop('do_post_job_processing')
    if 'local' in dict_args:
        dict_args.pop('local')
    if 'verbose' in dict_args:
        dict_args.pop('verbose')

    # write the arguments
    for key_counter, key in enumerate(dict_args):
        # key can be None, skip it in that case
        if dict_args[key] is None:
            continue
        to_add = ''
        key_string = ''
        for k in key.split('_'):
            key_string += k[0]
        # if the key is a list, we need to iterate over the list
        if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
            to_add += '.' + key_string
            for i, item in enumerate(dict_args[key]):
                to_add += '-' + str(item)
        elif isinstance(dict_args[key], bool):
            if dict_args[key]:
                to_add += '.' + key_string + '-True'
        elif "/" in str(dict_args[key]):
            processed_string = str(dict_args[key])
            processed_string = processed_string.split('/')[-1]
            processed_string = processed_string.split('.')[0]
            to_add += '.' + key_string + '-' + processed_string
        else:
            to_add = '.' + key_string + '-' + str(dict_args[key])

        if run_dir.endswith('/') and to_add.startswith('.'):
            run_dir += to_add[2:]
        else:
            run_dir += to_add
            
    return run_dir

def get_immediate_subdirectories_of(directory):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all subdirectories
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def get_files_in(directory, extension=None):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all files
    if extension is None:
        return [name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name))]
    else:
        return [name for name in os.listdir(directory)
                if (os.path.isfile(os.path.join(directory, name)) and
                    os.path.splitext(name)[1] == extension)]

def create_folder_structure(rundir):
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    if os.path.exists(os.path.join(rundir, 'pickled_population')):
        shutil.rmtree(os.path.join(rundir, 'pickled_population'))
    os.makedirs(os.path.join(rundir, 'pickled_population'))
    if os.path.exists(os.path.join(rundir, 'evolution_summary')):
        shutil.rmtree(os.path.join(rundir, 'evolution_summary'))
    os.makedirs(os.path.join(rundir, 'evolution_summary'))
    if os.path.exists(os.path.join(rundir, 'best_so_far')):
        shutil.rmtree(os.path.join(rundir, 'best_so_far'))
    os.makedirs(os.path.join(rundir, 'best_so_far'))
    if os.path.exists(os.path.join(rundir, 'to_record')):
        shutil.rmtree(os.path.join(rundir, 'to_record'))
    os.makedirs(os.path.join(rundir, 'to_record'))
    if os.path.exists(os.path.join(rundir, 'sub_simulations')):
        shutil.rmtree(os.path.join(rundir, 'sub_simulations'))
    os.makedirs(os.path.join(rundir, 'sub_simulations'))

# pareto front related
def natural_sort(l, reverse):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)

def make_one_shape_only(output_state):
    """Find the largest continuous arrangement of True elements after applying boolean mask.

    Avoids multiple disconnected softbots in simulation counted as a single individual.

    Parameters
    ----------
    output_state : numpy.ndarray
        Network output

    Returns
    -------
    part_of_ind : bool
        True if component of individual

    """
    one_shape = np.zeros(output_state.shape, dtype=np.int32)

    not_yet_checked = []
    for x in range(output_state.shape[0]):
        for y in range(output_state.shape[1]):
                not_yet_checked.append((x, y))

    largest_shape = []
    queue_to_check = []
    while len(not_yet_checked) > len(largest_shape):
        queue_to_check.append(not_yet_checked.pop(0))
        this_shape = []
        if output_state[queue_to_check[0]]:
            this_shape.append(queue_to_check[0])

        while len(queue_to_check) > 0:
            this_voxel = queue_to_check.pop(0)
            x = this_voxel[0]
            y = this_voxel[1]
            for neighbor in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if neighbor in not_yet_checked:
                    not_yet_checked.remove(neighbor)
                    if output_state[neighbor]:
                        queue_to_check.append(neighbor)
                        this_shape.append(neighbor)

        if len(this_shape) > len(largest_shape):
            largest_shape = this_shape

    for loc in largest_shape:
        one_shape[loc] = 1

    return one_shape

def moore_neighbors(bin, radius, shape):
    """ return moore neighbors of a bin (tuple of x, y)"""
    neighbors = []
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            if x == 0 and y == 0:
                continue
            if bin[0]+x < 0 or bin[0]+x >= shape[0]:
                continue
            if bin[1]+y < 0 or bin[1]+y >= shape[1]:
                continue
            neighbors.append((bin[0]+x, bin[1]+y))
    return neighbors
            
def printv(string):
    if settings.VERBOSE:
        print(string)
            





