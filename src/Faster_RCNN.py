import torch, torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch import optim, nn

from torchvision.models.vgg import vgg16

import pytorch_lightning as pl

import numpy as np
from typing import Tuple, List
from functools import reduce

class fasterRCNN(pl.LightningModule):
    def __init__(self,
                backbone : nn.Module = vgg16().features,
                backbone_out_channel : int = 512,
                anachor_size = ((128, 256, 512)),
                anachor_ratio  = ((0.5, 1, 2)),
                roi_output_size : int = 7,
                roi_sampling_ratio : int = 2,
                num_classes : int = 100,
                optim_rate:float = 0.001,
                optim_moment:float = 0.9,
                optim_weight_decay:float = 0.0005,
                logger_type:str = 'test_tube',
                rpn_pre_nms_top_n_train=300,
                rpn_pre_nms_top_n_test=300,
                rpn_nms_thresh=0.4,
                box_score_thresh=0.05,
                rpn_fg_iou_thresh=0.7):
        super(fasterRCNN, self).__init__()
        self.save_hyperparameters()
        self.optim_rate = optim_rate
        self.optim_moment = optim_moment
        self.optim_weight_decay  =optim_weight_decay
        
        backbone = backbone
        backbone.out_channels = backbone_out_channel
        
        anachor_generator = AnchorGenerator(sizes=anachor_size,
                                            aspect_ratios=anachor_ratio)
        roi_pooler =  MultiScaleRoIAlign(featmap_names=['0'],
                                         output_size=roi_output_size,
                                         sampling_ratio=roi_sampling_ratio)
        self.logger_type = logger_type
        self.model = FasterRCNN(backbone,
                                num_classes=num_classes,
                                rpn_anchor_generator=anachor_generator,
                                box_roi_pool=roi_pooler,
                                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                rpn_nms_thresh=rpn_nms_thresh,
                                box_score_thresh=box_score_thresh,
                                rpn_fg_iou_thresh=rpn_fg_iou_thresh)
        
        
    def setup(self, stage):
        self.logger.experiment.log_hyperparams(self.hparams)
        if self.logger_type == 'comet':
            self.logger.experiment.set_model_graph(str(self.model))
        
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.model.training :
            self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx)
        self.logger.log_metrics({key : val.item() for key, val in loss.items()},
                                           step=self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss_sum = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_classifier = torch.stack([x['loss_classifier'] for x in outputs]).mean()
        avg_loss_box_reg = torch.stack([x['loss_box_reg'] for x in outputs]).mean()
        avg_loss_objectness = torch.stack([x['loss_objectness'] for x in outputs]).mean()
        avg_loss_rpn_box_reg = torch.stack([x['loss_rpn_box_reg'] for x in outputs]).mean()
        self.logger.log_metrics({'avg_loss_sum' : avg_loss_sum.item(),
                                            'avg_loss_classifier':avg_loss_classifier.item(),
                                            'avg_loss_box_reg' : avg_loss_box_reg.item(),
                                            'avg_loss_objectness':avg_loss_objectness.item(),
                                            'avg_loss_rpn_box_reg':avg_loss_rpn_box_reg.item()},
                                           step=self.current_epoch)
        return 
    
    def share_step(self, batch, batch_idx):
        batch_img = []
        batch_targets = []
        for batch_item in batch:
            img, _, _, labels, bbs = batch_item
            batch_img.append(img)
            batch_targets.append({'boxes' : bbs,
                            'labels' : labels})
        
        self.model.train()
        loss = self.model(batch_img,batch_targets)
        loss_sum = reduce(lambda x, y : x+y, list(loss.values()))
        return dict(**{'loss' : loss_sum}, **loss)
    
    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(),
                         lr=self.optim_rate,
                         momentum=self.optim_moment,
                         weight_decay=self.optim_weight_decay)
    
    
