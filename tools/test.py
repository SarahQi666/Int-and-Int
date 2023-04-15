

import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config, load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from pyskl.datasets import build_dataloader, build_dataset 
from thop import profile
from thop import clever_format


import copy
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from pyskl.core import mean_average_precision, mean_class_accuracy, top_k_accuracy

from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port




def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:


        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips


    model = build_model(cfg.model)





    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    return outputs


def main():
    args = parse_args()



    
    cfg = Config.fromfile(args.config)

    out = osp.join(cfg.work_dir, 'result2.pkl') if args.out is None else args.out


    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')


    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)


    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:


        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    dist.barrier()
    outputs = inference_pytorch(args, cfg, data_loader) 


    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\nwriting results to {out}') 

        mmcv.dump(outputs, out) 



        if eval_cfg:




            results=outputs
            metrics='top_k_accuracy'
            metric_options=dict(top_k_accuracy=dict(topk=(1, 5)))
            logger=None
            data_prefix=''

            metric_options = copy.deepcopy(metric_options)
            metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
            eval_results = OrderedDict()


            data_2s = mmcv.load(cfg.data.test.ann_file)

            if cfg.data.test.split:
                split, data_2s = data_2s['split'], data_2s['annotations']
                identifier = 'filename' if 'filename' in data_2s[0] else 'frame_dir'
                split = set(split[cfg.data.test.split])
                data_2s = [x for x in data_2s if x[identifier] in split]

            for item in data_2s: 

                if 'filename' in item:
                    item['filename'] = osp.join(data_prefix, item['filename'])
                if 'frame_dir' in item:
                    item['frame_dir'] = osp.join(data_prefix, item['frame_dir'])
            gt_labels = [ann['label'] for ann in data_2s] 

            for metric in metrics:
                msg = f'Evaluating {metric} ...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                if metric == 'top_k_accuracy':
                    topk = metric_options.setdefault('top_k_accuracy',
                                                    {}).setdefault(
                                                        'topk', (1, 5))
                    if not isinstance(topk, (int, tuple)):
                        raise TypeError('topk must be int or tuple of int, '
                                        f'but got {type(topk)}')
                    if isinstance(topk, int):
                        topk = (topk, )

                    top_k_acc = top_k_accuracy(results, gt_labels, topk)
                    log_msg = []
                    for k, acc in zip(topk, top_k_acc):
                        eval_results[f'top{k}_acc'] = acc
                        log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                    log_msg = ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                    continue

                if metric == 'mean_class_accuracy':
                    mean_acc = mean_class_accuracy(results, gt_labels)
                    eval_results['mean_class_accuracy'] = mean_acc
                    log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                    print_log(log_msg, logger=logger)
                    continue
            eval_res=eval_results
            
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')















    dist.barrier()
    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()








