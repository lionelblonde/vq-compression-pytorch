import random
from pathlib import Path
import subprocess

import numpy as np
import yaml

from helpers import logger


def uuid(num_syllables=2, num_parts=3):
    """Randomly create a semi-pronounceable uuid"""
    part1 = ['s', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr']
    part2 = ['a', 'oo', 'ee', 'e', 'u', 'er']
    seps = ['_']  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2, strict=True):
            result += part1[i1] + part2[i2]
    return result


class ConfigDumper:

    def __init__(self, args, path):
        """Log the job config into a file"""
        self.args = args
        Path(path).mkdir(exist_ok=True)
        self.path = path

    def dump(self):
        hpmap = self.args.__dict__
        path = Path(self.path) / 'hyperparameters.yml'
        path.write_text(yaml.dump(hpmap, default_flow_style=False))  # sanity check: print(path.read_text())


class ExperimentInitializer:

    def __init__(self, args):
        """Initialize the experiment"""
        self.uuid_provided = (args.uuid is not None)
        self.uuid = args.uuid if self.uuid_provided else uuid()
        self.args = args
        # Set printing options
        np.set_printoptions(precision=3)

    def configure_logging(self, train=True):
        """Configure the experiment"""
        if train:
            log_path = Path(self.args.log_dir) / self.get_name()
            formats_strs = ['stdout', 'log', 'csv']
            logger.info("configuring logger")
            logger.configure(dir_=str(log_path), format_strs=formats_strs)
            logger.info("logger configured")
            logger.info(f"  directory: {log_path}")
            logger.info(f"  output formats: {formats_strs}")
            # In the same log folder, log args in a YAML file
            config_dumper = ConfigDumper(args=self.args, path=str(log_path))
            config_dumper.dump()
            logger.info("experiment configured")
        else:
            logger.info("configuring logger for evaluation")
            logger.configure(dir_=None, format_strs=['stdout'])

    def get_name(self):
        """Assemble long experiment name"""
        if self.uuid_provided:
            # If the uuid has been provided, use it.
            return self.uuid
        # Assemble the uuid
        name = self.uuid + '.'
        try:
            out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            name += f"gitSHA_{out.strip().decode('ascii')}."
        except OSError:
             pass
        if self.args.task == 'eval':
            name += "INFERENCE"
        else:
            name += "TRAINING"
        name += f".{self.args.algo_handle}"
        name += f".seed{str(self.args.seed).zfill(2)}"
        return name
