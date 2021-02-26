from src.Faster_RCNN import fasterRCNN
from src.VRDDataLoader import VRD_DataModule

import torch, torchvision
import pytorch_lightning as pl
from torchvision.models.vgg import vgg16

import yaml
import argparse
import random
import numpy as np


from pytorch_lightning.loggers import CometLogger

backbone_dic = {'vgg16':vgg16}

# --- config
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE')
parser.add_argument('--gpus', '-g',
                    dest='gpus',
                    type=int,
                    default=1)

args = parser.parse_args()
with open(args.filename, 'r') as f:
    config = yaml.safe_load(f)
    
random_seed = config['random_seed']    

# --- Reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# --- Callback Function
callback_save_model = pl.callbacks.ModelCheckpoint(**config['Modelcheckpoint'])

# --- Loger
comet_logger = CometLogger(**config['logger_params'])

if __name__ == '__main__':
    # --- DataLoader
    dataloader = VRD_DataModule(**config['DataModule_params'])

    # --- Model
    model = fasterRCNN(
        backbone=backbone_dic[config['backbone']](pretrained=True).features,
        anachor_size= ((64, 128, 256),),
        anachor_ratio = ((0.5, 1, 2),),
        **config['model_params'])
    
    # --- Trainer
    runner = pl.Trainer(default_root_dir=f"{comet_logger.save_dir}",
                    gpus=args.gpus,
                    logger=comet_logger,
                    callbacks=[callback_save_model],
                    **config['trainer_params'])
    
    runner.fit(model, dataloader)

