import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from datasets_internal import build_dataset, get_coco_api_from_dataset
from datasets_internal.coco_eval import CocoEvaluator
from models import build_model

import lightning as L

from swarm_one.pytorch import Client

swarm_one_client = Client(api_key="bSNxj8I21w")

# pip install torchvision==0.16.0 torch==2.1.0 timm

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=5e-6, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=24, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', default=False, action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='swin', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=5, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=900, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--assign_first_stage', default=True, action='store_true')
    parser.add_argument('--assign_second_stage', default=True, action='store_true')
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=1.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/home/user/workspace/swarmone/swarm_poc/data',
                        type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--bigger', default=True, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', default='/home/user/workspace/swarmone/converted_deta_swin_o365_finetune.pth', help='finetune from checkpoint')
    # parser.add_argument('--finetune', default='/app/weights/adet_swin_ft.pth', help='finetune from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])

## Custom args
custom_args = '--with_box_refine --two_stage \
    --num_feature_levels 5 --num_queries 900 \
    --dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
    --assign_first_stage --assign_second_stage \
    --epochs 24 --lr_drop 20 \
    --lr 5e-5 --lr_backbone 5e-6 --batch_size 1 \
    --backbone swin \
    --bigger'.split()

args = parser.parse_args(custom_args)

model, criterion, postprocessors = build_model(args)

dataset_train = build_dataset(image_set='train', args=args)
dataset_val = build_dataset(image_set='val', args=args)

data_loader_train = DataLoader(dataset_train, 2, shuffle=True,
                               collate_fn=utils.collate_fn, num_workers=0,
                               pin_memory=True)

data_loader_val = DataLoader(dataset_val, 2,
                             drop_last=False, collate_fn=utils.collate_fn, num_workers=0,
                             pin_memory=True)


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


if args.dataset_file == "coco_panoptic":
    # We also evaluate AP during panoptic training, on original coco DS
    coco_val = datasets.coco.build("val", args)
    base_ds = get_coco_api_from_dataset(coco_val)
else:
    base_ds = get_coco_api_from_dataset(dataset_val)

if args.frozen_weights is not None:
    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    model.detr.load_state_dict(checkpoint['model'])

if args.finetune:
    checkpoint = torch.load(args.finetune, map_location='cpu')
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if 'class_embed' in k:
            print('removing', k)
            del state_dict[k]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))


# if args.eval:
#     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
#                                           data_loader_val, base_ds, device, args.output_dir)
#     if args.output_dir:
#         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")


class DETAModel(L.LightningModule):
    def __init__(self, model, criterion, postprocessors, base_ds, args):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.base_ds = base_ds
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self.log("loss_ce", loss_dict["loss_ce"])
        self.log("loss_giou",loss_dict["loss_giou"])
        self.log("loss_bbox", loss_dict["loss_bbox"])

        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        samples = utils.nested_tensor_from_tensor_list(samples)

        outputs = self(samples)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        current_step_results = self.format_for_evaluation(outputs, targets)
        self.coco_evaluator.update(current_step_results)

        self.log("val_loss_ce", loss_dict["loss_ce"])
        self.log("val_loss_giou", loss_dict["loss_giou"])
        self.log("val_loss_bbox", loss_dict["loss_bbox"])

        return losses

    def format_for_evaluation(self, outputs, targets):
        batch_orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_results = self.postprocessors['bbox'](outputs, batch_orig_target_sizes)
        final_res = {target['image_id'].item(): output for target, output in zip(targets, batch_results)}
        return final_res

    def on_validation_epoch_start(self):
        iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        self.coco_evaluator = CocoEvaluator(self.base_ds, iou_types)
        
    def on_validation_epoch_end(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        map_metrics = self.coco_evaluator.coco_eval['bbox'].stats
        print(map_metrics)
        self.log("val_map", map_metrics[0])
        self.log("val_map50", map_metrics[1])
        self.log("val_map75", map_metrics[2])

    def configure_optimizers(self):
        param_dicts = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                     if not match_name_keywords(n, self.args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                            self.args.lr_linear_proj_names) and p.requires_grad],
                "lr": self.args.lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           match_name_keywords(n, self.args.lr_backbone_names) and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
                "lr": self.args.lr * self.args.lr_linear_proj_mult,
            }
        ]
        if self.args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=self.args.lr, momentum=0.9,
                                        weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)
        return [optimizer], [lr_scheduler]
    
    
    
model = DETAModel(model, criterion, postprocessors, base_ds, args)

hyperparameters = {
    "max_epochs": [50],
    "batch_sizes": [8],
}

# job_id = swarm_one_client.fit(
#     model='FU3HsIkF',
#     train_dataloaders=data_loader_val,
#     val_dataloaders=data_loader_val,
#     hyperparameters=hyperparameters
# )
job_id = '76RP82oY'
loggin_dir = f"/home/user/workspace/swarmone/swarm_poc/data/tf_log/{job_id}"
from logging_swarmone_clearml import log_job_metrics

log_job_metrics(job_id, "DETA", swarm_one_client)
# swarm_one_client.download_tensorboard_logs(job_id, log_dir=loggin_dir, show_tensorboard=False)
