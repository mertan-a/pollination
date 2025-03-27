# for stopping the job before finish and resubmitting it to the slurm queue
STOP = False
START_TIME = 0

# dictionary for maximum allowed time for each job
MAX_TIME = { 'short': 3*60*60 - 60*60, 'week': 3*24*60*60 - 60*60, 'bluemoon': 25*60*60 }

# verbose
VERBOSE = False
