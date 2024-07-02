import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from thop.vision.basic_hooks import count_linear
import thop
from torch.nn import Linear
import copy
from torch.profiler import profile as t_profile, record_function, ProfilerActivity
# Register the hook for Linear layers

import os
os.environ["http_proxy"] = "http://10.21.61.124:10809"
os.environ["https_proxy"] = "https://10.21.61.124:10809"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='simtrack', choices=['ostrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, template_bb):
    '''Speed Test'''
    model_ = copy.deepcopy(model)
    
    # macs1, params1 = profile(model, inputs=(template, search, template_bb),
    #                          custom_ops=None, verbose=False)
    # macs, params = clever_format([macs1, params1], "%.3f")
    # print('overall macs is ', macs)
    # print('overall params is ', params)
    
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            x_dict = model_.forward_backbone([template, search, template_bb])
            _ = model_.forward_head(seq_dict=[x_dict], run_box_head=True)
        start = time.time()
        for i in range(T_t):
            x_dict = model_.forward_backbone([template, search, template_bb])
            _ = model_.forward_head(seq_dict=[x_dict], run_box_head=True)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))
    # with t_profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model_(template, search)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=100))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.DATA.TEMPLATE.SIZE
    x_sz = cfg.DATA.SEARCH.SIZE

    if args.script == "simtrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_simtrack
        model = model_constructor(cfg)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        template_bb = torch.tensor([[0.5,0.5,0.5,0.5]])
    
        # template = torch.randn(bs, 64, 768)
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device)
        model.eval()
        template = template.to(device)
        search = search.to(device)
        template_bb = template_bb.to(device)

        # merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER

        evaluate_vit(model, template, search, template_bb)

    else:
        raise NotImplementedError
