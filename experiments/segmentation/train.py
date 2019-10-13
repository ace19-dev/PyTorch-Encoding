import os
import time
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm, OHEMSegmentationLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model

from experiments.segmentation.option import Options
from experiments.segmentation.transforms import *

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable


DATA_DIR = '/home/ace19/dl_data/steel_defect_detection'


class Trainer():
    def __init__(self, args):
        self.args = args

        self.losses = {'train':[], 'val':[]}
        self.dice_scores = {'train':[], 'val':[]}
        self.iou_scores = {'train':[], 'val':[]}
        self.best_loss = float("inf")

        # # data transforms
        # input_transform = transform.Compose([
        #     transform.ToTensor(),
        #     transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        # dataset
        data_kwargs = {'transform': get_training_augmentation(), 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset,
                               root=DATA_DIR,
                               source=['train.csv'],
                               split=['train_#2fold_11940.npy'],
                               mode='train',
                               **data_kwargs)
        testset = get_dataset(args.dataset,
                              root=DATA_DIR,
                              source=['train.csv'],
                              split=['valid_#2fold_628.npy'],
                              mode ='val',
                              **data_kwargs)
        # dataloader
        # kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        #     if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
        self.nclass = trainset.num_class

        # model
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size)
        print(model)

        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

        # criterions
        # self.criterion = SegmentationLosses(se_loss=args.se_loss,
        #                                     aux=args.aux,
        #                                     nclass=self.nclass,
        #                                     se_weight=args.se_weight,
        #                                     aux_weight=args.aux_weight)
        # self.criterion = OHEMSegmentationLosses(se_loss=args.se_loss,
        #                                         aux=args.aux,
        #                                         nclass=self.nclass,
        #                                         se_weight=args.se_weight,
        #                                         aux_weight=args.aux_weight)
        self.criterion = nn.BCEWithLogitsLoss()
        self.model, self.optimizer = model, optimizer
        # using cuda
        # if args.cuda:
        #     self.model = DataParallelModel(self.model).cuda()
        #     self.criterion = DataParallelCriterion(self.criterion).cuda()
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.model = nn.DataParallel(model)

        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.best_pred = checkpoint['best_pred']
            self.best_loss = checkpoint['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))
        self.best_pred = 0.0

    def training(self, epoch):
        self.model.train()

        # global losses, iou_scores, dice_scores

        train_loss = 0.0
        total_batches = len(self.trainloader)

        meter = Meter('train', epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Start train > ⏰: {start}")

        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            if args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)[0]
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            output = output.detach().cpu()
            meter.update(target, output)
            dices, iou = meter.get_metrics()
            tbar.set_description('\rLoss: %.5f | IoU: %.5f | dice: %.5f' %
                                 (train_loss / (i + 1), iou, dices[0]))

        epoch_loss = train_loss / total_batches
        dice, iou = epoch_log(epoch_loss, meter, start)
        self.losses['train'].append(epoch_loss)
        self.dice_scores['train'].append(dice)
        self.iou_scores['train'].append(iou)
        torch.cuda.empty_cache()


    def validation(self, epoch):
        self.model.eval()

        # global losses, iou_scores, dice_scores
        is_best = False

        # accumulation_steps = 32 // args.val_batch_size
        train_loss = 0.0
        total_batches = len(self.valloader)

        meter = Meter('val', epoch)
        start = time.strftime("%H:%M:%S")
        print(f"\nStart val > ⏰: {start}")

        # # Fast test during the training
        # def eval_batch(model, image, target):
        #     outputs = model(image)
        #     outputs = gather(outputs, 0, dim=0)
        #     pred = outputs[0]
        #     correct, labeled = utils.batch_pix_accuracy(pred.data, target)
        #     inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
        #     return correct, labeled, inter, union

        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)[0]
                loss = self.criterion(output, target)

                train_loss += loss.item()
                output = output.detach().cpu()
                meter.update(target, output)

            dices, iou = meter.get_metrics()
            tbar.set_description('\rLoss: %.5f | IoU: %.5f | dice: %.5f' %
                                 (train_loss / (i + 1), iou, dices[0]))
        # end of for

        # epoch_loss = (train_loss * accumulation_steps) / total_batches
        epoch_loss = train_loss / total_batches
        dice, iou = epoch_log(epoch_loss, meter, start)
        self.losses['val'].append(epoch_loss)
        self.dice_scores['val'].append(dice)
        self.iou_scores['val'].append(iou)
        torch.cuda.empty_cache()
        # TODO: below with ReduceLROnPlateau
        # self.scheduler.step(epoch_loss)

        # save checkpoint
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            is_best = True
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, args=args, is_best=is_best)


def plot(scores, name, args):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot');
    plt.xlabel('Epoch');
    plt.ylabel(f'{name}');
    plt.legend();
    # plt.show()

    # save results
    from datetime import datetime
    results_dir = '/home/ace19/dl_results/steel_defect_detection/' + \
                  datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '/' + \
                  args.model + '/' + args.backbone
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, name))


def _predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def _metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1).cpu()
        truth = truth.view(batch_size, -1).cpu()
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


# The IoU score is calculated for each class separately and
# then averaged over all classes to provide a global, mean IoU score of our semantic segmentation prediction.
def _compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels.cpu()) # tensor to np
    # control RuntimeWarning: Mean of empty slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for pred, label in zip(preds, labels):
            ious.append(np.nanmean(_compute_ious(pred, label, classes)))
        iou = np.nanmean(ious)

    return iou


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = _metric(probs, targets, self.base_threshold)
        # https://www.kaggle.com/wh1tezzz/correct-dice-metrics-for-this-competition
        # dice = dice_channel_torch(probs, targets, self.base_threshold)
        # print('Avg Dice score in this batch is {}'.format(dice))
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = _predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("\nLoss: %0.5f | IoU: %0.5f | dice: %0.5f | dice_neg: %0.5f | dice_pos: %0.5f" %
          (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou



if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    if args.eval:
        trainer.validation(trainer.args.start_epoch)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)

        # plot training
        plot(trainer.losses, "BCE loss", args)
        plot(trainer.dice_scores, "Dice score", args)
        plot(trainer.iou_scores, "IoU score", args)
