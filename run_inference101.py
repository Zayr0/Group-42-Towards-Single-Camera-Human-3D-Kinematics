import pickle as pkl
import numpy as np
import math
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).absolute().parent / "ms_model_estimation"))
from ms_model_estimation.training.train_os_spatialTemporal_infer101 import Training
from ms_model_estimation.training.config.config_os_spatialtemporal_time import get_cfg_defaults

cwd = Path(__file__).absolute().parent

cfg = get_cfg_defaults()

# cfg.STARTPOSMODELPATH = str((cwd / r'checkpoints\OS_ALL_L1_ANGLE006.pt').as_posix())
# cfg.STARTTEMPORALMODELPATH = str((cwd / r'checkpoints\model_best_OS_TEMPORAL_TRANSFORMER_L1.pt').as_posix())
cfg.STARTPOSMODELPATH = str((cwd / r'checkpoints/OS_ALL_L1_ANGLE006.pt').as_posix())
cfg.STARTTEMPORALMODELPATH = str((cwd / r'checkpoints/model_best_OS_TEMPORAL_TRANSFORMER_L1.pt').as_posix())
cfg.MODEL_FOLDER = str((cwd / r'checkpoints').as_posix())
cfg.BML_FOLDER = str((cwd / r'_dataset_full').as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")    

    args = parser.parse_args()    
    
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    trainingProgram.run_inference(datasplit='test')

