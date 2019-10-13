
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.utils import data

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.datasets import get_dataset
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
from experiments.segmentation.transforms import *

from experiments.segmentation.option import Options


DATA_DIR = '/home/ace19/dl_data/steel_defect_detection'
BEST_THRESHOLD = 0.7
MIN_SIZE = 3500



def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    # TypeError: OpenCV TypeError: Expected cv::UMat for argument 'src'
    # TODO: https://stackoverflow.com/questions/54249728/opencv-typeerror-expected-cvumat-for-argument-src-what-is-this
    mask = cv2.threshold(np.float32(probability), threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def sharpen(p, t=BEST_THRESHOLD):
    if t!=0:
        return p**t
    else:
        return p


def test(args):
    # # output folder
    # outdir = 'outdir'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    #
    # # data transforms
    # input_transform = transform.Compose([
    #     transform.ToTensor(),
    #     transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    # dataset
    # if args.eval:
    #     testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
    #                                        transform=input_transform)
    # elif args.test_val:
    #     testset = get_segmentation_dataset(args.dataset, split='val', mode='test',
    #                                        transform=input_transform)
    # else:
    #     testset = get_segmentation_dataset(args.dataset, split='test', mode='test',
    #                                        transform=input_transform)
    data_kwargs = {'transform': get_inference_augmentation(), 'base_size': args.base_size}
    testset = get_dataset(args.dataset,
                          root=DATA_DIR,
                          source=['sample_submission.csv'],
                          split=['test_1801.npy'],
                          mode='test',
                          **data_kwargs)

    # dataloader
    # loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
    #     if args.cuda else {}
    test_loader = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

    # # model
    # if args.model_zoo is not None:
    #     model = get_model(args.model_zoo, pretrained=True)
    #     #model.base_size = args.base_size
    #     #model.crop_size = args.crop_size
    # else:
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone = args.backbone, aux = args.aux,
                                   se_loss = args.se_loss, norm_layer = SyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size)
    print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    # strict=False, so that it is compatible with old pytorch saved models
    model.module.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
    #     [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # evaluator = MultiEvalModule(model, testset.num_class, scales=scales).cuda()
    # evaluator.eval()
    # metric = utils.SegmentationMetric(testset.num_class)

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    model.eval()
    predictions = []

    tbar = tqdm(test_loader)
    for i, (impaths, images) in enumerate(tbar):
        if args.cuda:
            images = images.cuda()

        with torch.no_grad():

            # logits = model(images)
            # probability_mask = torch.sigmoid(logits)
            # # probability_mask = torch.softmax(logits, 1)

            # TTA
            num_augment = 0
            # original
            # logits = data_parallel(model, input)  # net(input)
            logits = model(images)[0]
            # probability = torch.softmax(logits, 1)
            probability = torch.sigmoid(logits)
            probability_mask = sharpen(probability, 0)
            num_augment += 1

            # flip_lr
            # logits = data_parallel(model, torch.flip(input, dims=[3]))
            logits = model(torch.flip(images, dims=[3]))[0]
            # probability = torch.softmax(torch.flip(logits, dims=[3]), 1)
            probability = torch.sigmoid(torch.flip(logits, dims=[3]))
            probability_mask += sharpen(probability)
            num_augment += 1

            # flip_ud
            # logits = data_parallel(model, torch.flip(input, dims=[2]))
            logits = model(torch.flip(images, dims=[2]))[0]
            # probability = torch.softmax(torch.flip(logits, dims=[2]), 1)
            probability = torch.sigmoid(torch.flip(logits, dims=[2]))
            probability_mask += sharpen(probability)
            num_augment += 1

            # 5 224 crop
            # if '5crop' in augment.py:
            #     raise NotImplementedError
            #     for sx, sy in[ (16,16),(0,0),(32,0),(32,0),(32,32) ]:
            #         crop = input[:,:,sy:sy+224,sx:sx+1568]
            #         #print(crop.shape)
            #         logits = data_parallel(model,crop)
            #         probability = torch.sigmoid(logits)
            #
            #         probability_mask += sharpen(probability)
            #         num_augment+=1

            # scale/shift -> TODO: multi scale ?
            # input_pad = F.pad(images, [8,8,8,8], mode='constant', value=0)
            # for sx, sy in[ (8,0),(8,16),(0,8),(16,8) ]:
            #     # logits =  data_parallel(model,input_pad[:,:,sy:sy+H,sx:sx+W])
            #     logits = model(input_pad[:,:,sy:sy+H,sx:sx+W])[0]
            #     probability = torch.sigmoid(logits)
            #     probability_mask += sharpen(probability)
            #     num_augment+=1

            probability_mask = probability_mask / num_augment
            probability_mask = probability_mask.detach().cpu()
            for fname, preds in zip(impaths, probability_mask):
                for cls, pred in enumerate(preds):
                    pred, num = post_process(pred, BEST_THRESHOLD, MIN_SIZE)
                    rle = mask2rle(pred)
                    name = fname + f"_{cls + 1}"
                    predictions.append([name, rle])

    arch = 'resnet50-encnet-fold-2'
    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(os.path.join(args.result, arch + "_submission.csv"), index=False)
    # end of for


if __name__ == "__main__":
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # args.test_batch_size = torch.cuda.device_count()
    test(args)
