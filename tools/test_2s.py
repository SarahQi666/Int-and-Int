

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

import copy
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from pyskl.core import mean_average_precision, mean_class_accuracy, top_k_accuracy

from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config_j', help='test config_j file path') 
    parser.add_argument('config_b', help='test config_b file path') 
    parser.add_argument('-C_j', '--checkpoint_j', help='checkpoint_j file', default=None)
    parser.add_argument('-C_b', '--checkpoint_b', help='checkpoint_b file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result_2s file in pkl/yaml/json format')
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


def inference_pytorch_j(args, cfg_j, data_loader_j):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:


        if cfg_j.model.get('test_cfg') is None and cfg_j.get('test_cfg') is None:
            cfg_j.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg_j.model.get('test_cfg') is not None:
                cfg_j.model.test_cfg.average_clips = args.average_clips
            else:
                cfg_j.test_cfg.average_clips = args.average_clips


    model = build_model(cfg_j.model)

    if args.checkpoint_j is None:
        work_dir = cfg_j.work_dir
        args.checkpoint_j = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint_j)

    args.checkpoint_j = cache_checkpoint(args.checkpoint_j)

    load_checkpoint(model, args.checkpoint_j, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)



    outputs_j = multi_gpu_test(model, data_loader_j, args.tmpdir) 





    return outputs_j

def inference_pytorch_b(args, cfg_b, data_loader_b):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:


        if cfg_b.model.get('test_cfg') is None and cfg_b.get('test_cfg') is None:
            cfg_b.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg_b.model.get('test_cfg') is not None:
                cfg_b.model.test_cfg.average_clips = args.average_clips
            else:
                cfg_b.test_cfg.average_clips = args.average_clips


    model = build_model(cfg_b.model)

    if args.checkpoint_b is None:
        work_dir = cfg_b.work_dir
        args.checkpoint_b = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint_b)

    args.checkpoint_b = cache_checkpoint(args.checkpoint_b)
    load_checkpoint(model, args.checkpoint_b, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs_b = multi_gpu_test(model, data_loader_b, args.tmpdir)





    return outputs_b

def main():
    args = parse_args()





    
    cfg_j = Config.fromfile(args.config_j)
    cfg_b = Config.fromfile(args.config_b)

    out = osp.join(cfg_j.work_dir_2s, 'result_2s.pkl') if args.out is None else args.out 


    eval_cfg_j = cfg_j.get('evaluation', {})
    eval_cfg_b = cfg_b.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg_j.pop(key, None)
        eval_cfg_b.pop(key, None)
    if args.eval:
        eval_cfg_j['metrics'] = args.eval
        eval_cfg_b['metrics'] = args.eval

    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')


    if cfg_j.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg_b.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg_j.data.test.test_mode = True
    cfg_b.data.test.test_mode = True

    if not hasattr(cfg_j, 'dist_params'):
        cfg_j.dist_params = dict(backend='nccl')
    if not hasattr(cfg_b, 'dist_params'):
        cfg_b.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg_j.dist_params)

    rank, world_size = get_dist_info()
    cfg_j.gpu_ids = range(world_size)
    cfg_b.gpu_ids = range(world_size)


    dataset_j = build_dataset(cfg_j.data.test, dict(test_mode=True))
    dataset_b = build_dataset(cfg_b.data.test, dict(test_mode=True))
    dataloader_setting_j = dict(
        videos_per_gpu=cfg_j.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg_j.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting_b = dict(
        videos_per_gpu=cfg_b.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg_b.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting_j = dict(dataloader_setting_j, **cfg_j.data.get('test_dataloader', {}))
    dataloader_setting_b = dict(dataloader_setting_b, **cfg_b.data.get('test_dataloader', {}))
    data_loader_j = build_dataloader(dataset_j, **dataloader_setting_j)
    data_loader_b = build_dataloader(dataset_b, **dataloader_setting_b)

    default_mc_cfg = ('localhost', 22077)
    memcached_j = cfg_j.get('memcached', False)
    memcached_b = cfg_b.get('memcached', False)

    if rank == 0 and memcached_j:


        mc_cfg_j = cfg_j.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg_j, tuple) and mc_cfg_j[0] == 'localhost'
        if not test_port(mc_cfg_j[0], mc_cfg_j[1]):
            mc_on(port=mc_cfg_j[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg_j[0], mc_cfg_j[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    if rank == 0 and memcached_b:


        mc_cfg_b = cfg_b.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg_b, tuple) and mc_cfg_b[0] == 'localhost'
        if not test_port(mc_cfg_b[0], mc_cfg_b[1]):
            mc_on(port=mc_cfg_b[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg_b[0], mc_cfg_b[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '


    dist.barrier()




    outputs_j = inference_pytorch_j(args, cfg_j, data_loader_j) 
    outputs_b = inference_pytorch_b(args, cfg_b, data_loader_b) 



    if outputs_j!=None and outputs_b!=None:


        outputs_2s=[] 
        for m,n in zip(outputs_j,outputs_b):
            per_outputs_2s=m+n
            outputs_2s.append(per_outputs_2s)

    
    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\nwriting results to {out}') 

        mmcv.dump(outputs_2s, out)
        if eval_cfg_j:

            results=outputs_2s
            metrics='top_k_accuracy'
            metric_options=dict(top_k_accuracy=dict(topk=(1, 5)))
            logger=None
            data_prefix=''

            metric_options = copy.deepcopy(metric_options)
            metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
            eval_results = OrderedDict()


            data_2s = mmcv.load(cfg_j.data.test.ann_file)

            if cfg_j.data.test.split:
                split, data_2s = data_2s['split'], data_2s['annotations']
                identifier = 'filename' if 'filename' in data_2s[0] else 'frame_dir'
                split = set(split[cfg_j.data.test.split])
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
    if rank == 0 and memcached_j:
        mc_off()


if __name__ == '__main__':
    main()
