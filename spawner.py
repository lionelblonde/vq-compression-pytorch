import argparse

from copy import deepcopy
import os
import sys
import numpy as np
import subprocess
import yaml
from pathlib import Path

from helpers import logger
from helpers.argparser_util import boolean_flag
from helpers.experiment import uuid as create_uuid


MEMORY = 32
NUM_NODES = 1
NUM_WORKERS = 1
NUM_SWEEP_TRIALS = 10


def zipsame(*seqs):
    """Verify that all the sequences in `seqs` are the same length, then zip them together"""
    assert seqs, "empty input sequence"
    ref_len = len(seqs[0])
    assert all(len(seq) == ref_len for seq in seqs[1:])
    return zip(*seqs, strict=False)


class Spawner(object):

    def __init__(self, args):
        self.args = args

        # Retrieve config from filesystem
        self.config = yaml.safe_load(Path(self.args.config).open())

        # Assemble wandb project name
        self.wandb_project = '-'.join([
            self.config['wandb_project'].upper(),
            self.args.deployment.upper(),
        ])

        # Define spawn type
        self.type = 'sweep' if self.args.sweep else 'fixed'

        # Define the needed memory in GB
        self.memory = MEMORY

        # Write out the boolean arguments (using the 'boolean_flag' function)
        self.bool_args = [
            'cuda',
            'fp16',
            'pretrained_w_imagenet',
            'linear_probe',
            'fine_tuning',
            'lars',
            'sched',
        ]

        if self.args.deployment == 'slurm':
            # Translate intuitive 'caliber' into actual duration and partition on the Baobab cluster
            calibers = {
                'short': '0-06:00:00',
                'long': '0-12:00:00',
                'verylong': '1-00:00:00',
                'veryverylong': '2-00:00:00',
                'veryveryverylong': '4-00:00:00',
            }
            self.duration = calibers[self.args.caliber]  # intended KeyError trigger if invalid caliber
            if 'verylong' in self.args.caliber:
                if self.config['cuda']:
                    self.partition = 'private-cui-gpu'
                else:
                    self.partition = 'public-cpu,private-cui-cpu,public-longrun-cpu'
            else:
                if self.config['cuda']:
                    self.partition = 'shared-gpu,private-cui-gpu'
                else:
                    self.partition = 'shared-cpu,public-cpu,private-cui-cpu'

        # Create the data path
        match self.config['dataset_handle']:
            case 'bigearthnet':
                self.dataset = "BigEarthNet-v1.0"
            case _:
                raise ValueError("invalid dataset handle (strict folder naming rule!)")
        self.data_path = Path(os.environ['DATASET_DIR']) / self.dataset
        os.environ['DATASET_DIR'] = str(self.data_path)  # overwrite the environ variable

        # If fine-tuning or linear probing, add the path to the pretrained SSL model
        if 'load_checkpoint' in self.config:
            self.load_checkpoint = Path(os.environ['MODEL_DIR']) / self.config['load_checkpoint']

    def copy_and_add_seed(self, hpmap, seed):
        hpmap_ = deepcopy(hpmap)

        # Add the seed and edit the job uuid to only differ by the seed
        hpmap_.update({'seed': seed})

        # Enrich the uuid with extra information
        gitsha = ''
        try:
            out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            gitsha = "gitSHA_{}".format(out.strip().decode('ascii'))
        except OSError:
            pass

        uuid = f"{hpmap['uuid']}.{gitsha}.{hpmap['algo_handle']}_{NUM_WORKERS}"
        uuid += f".seed{str(seed).zfill(2)}"

        hpmap_.update({'uuid': uuid})

        return hpmap_

    def get_hps(self):
        """Return a list of maps of hyperparameters"""

        # Create a uuid to identify the current job
        uuid = create_uuid()

        # Assemble the hyperparameter map
        hpmap = {
            # meta
            # seed handled afterwards
            'uuid': uuid,  # created earlier just here

            # resources
            'cuda': self.config['cuda'],
            'fp16': self.config['fp16'],

            # logging
            'wandb_project': self.wandb_project,  # assembled earlier here

            # dataset
            'dataset_handle': self.config['dataset_handle'],
            'data_path': self.data_path,  # assembled earlier here
            'val_split': self.config['val_split'],
            'test_split': self.config['test_split'],

            # training
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'save_freq': self.config['save_freq'],

            # opt
            'lr': self.config['lr'],
            'wd': self.config['wd'],
            'clip_norm': self.config['clip_norm'],
            'acc_grad_steps': self.config['acc_grad_steps'],
            'lars': self.config['lars'],
            'sched': self.config['sched'],

            # algo
            'algo_handle': self.config['algo_handle'],
        }
        if 'truncate_at' in self.config:
            hpmap.update({'truncate_at': self.config['truncate_at']})

        algo_handle = hpmap['algo_handle']
        if algo_handle == 'classifier':
            hpmap.update({
                # model architecture
                'backbone': self.config['backbone'],
                'pretrained_w_imagenet': self.config['pretrained_w_imagenet'],
                'fc_hid_dim': self.config['fc_hid_dim'],
            })
        elif algo_handle == 'simclr':
            hpmap.update({
                # model architecture
                'backbone': self.config['backbone'],
                'pretrained_w_imagenet': self.config['pretrained_w_imagenet'],
                'fc_hid_dim': self.config['fc_hid_dim'],
                'fc_out_dim': self.config['fc_out_dim'],
                # algorithm
                'ntx_temp': self.config['ntx_temp'],
                # fine-tuning or linear probing
                'linear_probe': self.config['linear_probe'],
                'fine_tuning': self.config['fine_tuning'],
                'ftop_epochs': self.config['ftop_epochs'],
                'ftop_batch_size': self.config['ftop_batch_size'],
            })
            if 'load_checkpoint' in self.config:
                hpmap.update({'load_checkpoint': self.load_checkpoint})
        elif algo_handle == 'compressor':
            hpmap.update({
                # training
                'max_lr': self.config['max_lr'],
                # model architecture
                'in_channels': self.config['in_channels'],
                'z_channels': self.config['z_channels'],
                'ae_hidden': self.config['ae_hidden'],
                'ae_resblocks': self.config['ae_resblocks'],
                'ae_kernel': self.config['ae_kernel'],
                'dsf': self.config['dsf'],
                # loss
                'alpha': self.config['alpha'],
                'beta': self.config['beta'],
                # centers
                'c_num': self.config['c_num'],
                'c_min': self.config['c_min'],
                'c_max': self.config['c_max'],
            })
        else:
            raise ValueError(f"invalid algo handle: {algo_handle}!")

        if self.args.sweep:
            # Random search: replace some entries with random values
            rng = np.random.default_rng(seed=None)
            hpmap.update({
                'batch_size': int(rng.choice([64, 128, 256])),
                'lr': float(rng.choice([1e-4, 3e-4])),
            })

        # Duplicate for each seed
        hpmaps = [self.copy_and_add_seed(hpmap, seed)
                  for seed in range(self.args.num_seeds)]

        return hpmaps

    def unroll_options(self, hpmap):
        """Transform the dictionary of hyperparameters into a string of bash options"""
        indent = 4 * ' '  # choice: indents are defined as 4 spaces
        arguments = ""

        for k, v in hpmap.items():
            if k in self.bool_args:
                if v is False:
                    argument = f"no-{k}"
                else:
                    argument = f"{k}"
            else:
                argument = f"{k}={v}"

            arguments += f"{indent}--{argument} \\\n"

        return arguments

    def create_job_str(self, name, command):
        """Build the batch script that launches a job"""

        # Prepend python command with python binary path
        command = Path(os.environ['CONDA_PREFIX']) / "bin" / command

        if self.args.deployment == 'slurm':
            Path("./out").mkdir(exist_ok=True)
            # Set sbatch config
            bash_script_str = ('#!/usr/bin/env bash\n\n')
            bash_script_str += (f"#SBATCH --job-name={name}\n"
                                f"#SBATCH --partition={self.partition}\n"
                                f"#SBATCH --nodes={NUM_NODES}\n"
                                f"#SBATCH --ntasks={NUM_WORKERS}\n"
                                "#SBATCH --cpus-per-task=4\n"
                                f"#SBATCH --time={self.duration}\n"
                                f"#SBATCH --mem={self.memory}000\n"
                                "#SBATCH --output=./out/run_%j.out\n")

            # Sometimes versions are needed (some clusters)
            if self.config['cuda']:
                constraint = ""
                bash_script_str += ("#SBATCH --gpus=titan:1\n")
                if constraint != "":
                    bash_script_str += (f'#SBATCH --constraint="{constraint}"\n')
            bash_script_str += ('\n')

            # Load modules
            bash_script_str += ("module load GCC/9.3.0\n")
            bash_script_str += ("module load CUDA/11.5.0\n")
            bash_script_str += ('\n')

            if self.args.quick:
                # Launch command
                bash_script_str += (f"srun {command}")
            else:
                # Add launch of a script that copies the dataset on the node's SSD
                pre1 = "chmod u+x prolog.sh"
                pre2 = ". prolog.sh"
                # Launch command
                bash_script_str += (f"srun {pre1} && {pre2} && {command}")

        elif self.args.deployment == 'tmux':
            # Set header
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"# job name: {name}\n\n")
            # Launch command
            bash_script_str += (f"{command}")  # left in this format for easy edits

        else:
            raise NotImplementedError("cluster selected is not covered.")

        return bash_script_str[:-2]  # remove the last `\` and `\n` tokens


def run(args):
    """Spawn jobs"""

    tmux_dir = ''

    if args.wandb_upgrade:
        # Upgrade the wandb package
        logger.info(">>>>>>>>>>>>>>>>>>>> Upgrading wandb pip package")
        out = subprocess.check_output([sys.executable, '-m', 'pip', 'install', 'wandb', '--upgrade'])
        logger.info(out.decode("utf-8"))

    # Create a spawner object
    spawner = Spawner(args)

    # Create directory for spawned jobs
    root = Path(__file__).resolve().parent
    spawn_dir = Path(root) / 'spawn'
    Path(spawn_dir).mkdir(exist_ok=True)
    if args.deployment == 'tmux':
        tmux_dir = Path(root) / 'tmux'
        Path(tmux_dir).mkdir(exist_ok=True)

    # Get the hyperparameter set(s)
    if args.sweep:
        hpmaps_ = [spawner.get_hps()
                   for _ in range(NUM_SWEEP_TRIALS)]
        # Flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = spawner.get_hps()

    # Create associated task strings
    commands = ["python main.py \\\n{}".format(spawner.unroll_options(hpmap)) for hpmap in hpmaps]
    if not len(commands) == len(set(commands)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again (:")
    # Create the job maps
    names = [f"{spawner.type}.{hpmap['uuid']}" for _, hpmap in enumerate(hpmaps)]

    # Finally get all the required job strings
    jobs = [spawner.create_job_str(name, command)
            for name, command in zipsame(names, commands)]

    # Spawn the jobs
    for i, (name, job) in enumerate(zipsame(names, jobs)):
        logger.info(f"job#={i},name={name} -> ready to be deployed.")
        if args.debug:
            logger.info("config below.")
            logger.info(job + "\n")
        dirname = name.split('.')[1]
        full_dirname = Path(spawn_dir) / dirname
        Path(full_dirname).mkdir(exist_ok=True)
        job_name = Path(full_dirname) / f"{name}.sh"
        job_name.write_text(job)
        if args.deploy_now and not args.deployment == 'tmux':
            # Spawn the job!
            stdout = subprocess.run(["sbatch", job_name]).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"job#={i},name={name} -> deployed on slurm.")

    if args.deployment == 'tmux':
        dir_ = hpmaps[0]['uuid'].split('.')[0]  # arbitrarilly picked index 0
        session_name = f"{spawner.type}-{str(args.num_seeds).zfill(2)}seeds-{dir_}"
        yaml_content = {'session_name': session_name,
                        'windows': [],
                        'environment': {'DATASET_DIR': os.environ['DATASET_DIR']}}
        for i, name in enumerate(names):
            executable = f"{name}.sh"
            pane = {'shell_command': [f"source activate {args.conda_env}",
                                      f"chmod u+x spawn/{dir_}/{executable}",
                                      f"spawn/{dir_}/{executable}"]}
            window = {'window_name': f"job{str(i).zfill(2)}",
                      'focus': False,
                      'panes': [pane]}
            yaml_content['windows'].append(window)
            logger.info(f"job#={i},name={name} -> will run in tmux, session={session_name},window={i}.")
        # Dump the assembled tmux config into a yaml file
        job_config = Path(tmux_dir) / f"{session_name}.yaml"
        job_config.write_text(yaml.dump(yaml_content, default_flow_style=False))
        if args.deploy_now:
            # Spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", job_config]).stdout
            if args.debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"[{len(jobs)}] jobs are now running in tmux session '{session_name}'.")
    else:
        # Summarize the number of jobs spawned
        logger.info(f"[{len(jobs)}] jobs were spawned.")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--conda_env', type=str, default=None)
    parser.add_argument('--deployment', type=str, choices=['tmux', 'slurm'], default='tmux', help='deploy how?')
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--caliber', type=str, choices=['short', 'long', 'verylong', 'veryverylong'], default='short')
    boolean_flag(parser, 'deploy_now', default=True, help="deploy immediately?")
    boolean_flag(parser, 'sweep', default=False, help="hp search?")
    boolean_flag(parser, 'wandb_upgrade', default=True, help="upgrade wandb?")
    boolean_flag(parser, 'wandb_dryrun', default=True, help="toggle wandb offline mode")
    boolean_flag(parser, 'debug', default=False, help="toggle debug/verbose mode in spawner")
    parser.add_argument('--debug_lvl', type=int, default=0, help="set the debug level for the spawned runs")
    boolean_flag(parser, 'quick', default=False, help="make it quick (no scratch dataset copy)")
    args = parser.parse_args()

    if args.wandb_dryrun:
        # Run wandb in offline mode (does not sync with wandb servers in real time,
        # use `wandb sync` later on the local directory in `wandb/` to sync to the wandb cloud hosted app)
        os.environ["WANDB_MODE"] = "dryrun"

    # Set the debug level for the spawned runs
    os.environ["DEBUG_LVL"] = str(args.debug_lvl)

    # Create (and optionally deploy) the jobs
    run(args)
