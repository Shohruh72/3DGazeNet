import argparse
import os
import cv2
import csv
import copy
import tqdm
import torch
import numpy as np
import menpo.io as mio

from nets import nn
from timm import utils
from utils import util
from utils.dataset import Dataset
from torch.utils.data import DataLoader


def train(args):
    model = nn.GazeNet(args.backbone_id).cuda()
    dataset = Dataset(args, True)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(dataset, args.batch_size, not args.distributed, sampler, num_workers=2, pin_memory=True,
                        drop_last=True)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

    best = float('inf')
    num_steps = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    with open('./weights/logs.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'Loss'])
            writer.writeheader()

        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            for images, labels in p_bar:
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    if args.distributed:
                        loss = model.module.loss(outputs, labels, hm=model.module.hard_mining)
                    else:
                        loss = model.loss(outputs, labels, hm=model.hard_mining)
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update(None)

                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                avg_loss.update(loss.item(), images.size(0))
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                last = test(args, copy.deepcopy(model.module if args.distributed else model))
                scheduler.step(last)
                writer.writerow({'Loss': str(f'{last:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                f.flush()
                if best > last:
                    best = last

                ckpt = {'model': copy.deepcopy(model.module if args.distributed else model).half()}

                torch.save(ckpt, './weights/last.pt')
                if best == last:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt
                print(f"Best Loss = {best:.3f}")

    if args.local_rank == 0:
        util.strip_optimizer('./outputs/weights/best.pt')
        util.strip_optimizer('./outputs/weights/last.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    if model is None:
        model = torch.load(f=f'./weights/best.pt', map_location='cuda')
        model = model['model'].float().cuda()

    model.eval()

    dataset = Dataset(args, False)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    avg_loss = util.AverageMeter()
    for images, labels in tqdm.tqdm(loader, '%10s' % 'Losses'):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = model.loss(outputs, labels, hm=model.hard_mining)

        avg_loss.update(loss.item(), images.size(0))

    print(f"Last Loss = {avg_loss.avg:.3f}")

    model.float()  # for training
    return avg_loss.avg


def demo(args):
    model = torch.load(f=f'./weights/last.pt', map_location='cuda')
    model = model['model'].float().eval()
    detector = util.FaceDetector('weights/detection.onnx')

    eyes_mean = mio.import_pickle('weights/eyes3d.pkl')
    tri481 = eyes_mean['mask481']['trilist']
    iris_idx_481 = eyes_mean['mask481']['idxs_iris']

    stream = cv2.VideoCapture(0)
    fps = int(stream.get(cv2.CAP_PROP_FPS))
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    vid_out = cv2.VideoWriter('demo/demo.avi', fourcc, fps, (2 * w, h))

    if not stream.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, image = stream.read()

        if not ret:
            break

        clone_image = image.copy()
        if image is None:
            continue

        results = []
        boxes, kps = detector.detect(image, input_size=(640, 640))
        kps = kps.reshape(kps.shape[1], kps.shape[2])
        boxes = boxes.astype('int32')

        for box in boxes:
            x_min, y_min, x_max, y_max = box[:4]
            box_w, box_h = x_max - x_min, y_max - y_min
            center = (kps[0] + kps[1]) / 2.0
            _size = max(box_w / 1.5, np.abs(kps[1][0] - kps[0][0])) * 1.5
            _scale = args.input_size / _size
            image, m = util.transform(image, center, args.input_size, _scale, 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = image.copy()
            inputs = np.transpose(inputs, (2, 0, 1))
            inputs = np.expand_dims(inputs, 0)
            images = torch.Tensor(inputs).cuda()
            images.div_(255).sub_(0.5).div_(0.5)
            output = model(images).detach().cpu().numpy().flatten().reshape((-1, 3))
            output[:, 0:2] += 1
            output[:, 0:2] *= (args.input_size // 2)
            output[:, 2] *= 10.0
            i_m = cv2.invertAffineTransform(m)
            pred = util.trans_points(output, i_m)
            result = (box[:4], kps, pred)
            results.append(result)

        if len(results) == 0:
            continue

        e_img = util.draw_on(iris_idx_481, tri481, clone_image, results)
        o_img = np.concatenate((clone_image, e_img), axis=1)

        cv2.imshow('output', o_img)
        vid_out.write(o_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    vid_out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../Datasets/Gaze/gaze_refine')
    parser.add_argument('--backbone_id', type=str, default='resnet101d')
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--input-size', type=int, default=160)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.makedirs('weights', exist_ok=True)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
