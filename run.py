# coding: UTF-8
# Author: JayChan
import argparse
import os
import time
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='SimBERT Torch version')
# parameters of datasets
parser.add_argument('--dataset_path', type=str, default="./corpus/data_similarity.json", help='dataset_path')
# parameters of model
parser.add_argument('--model', type=str, default="simbert", help='choose a model: simbert,simCSE')
parser.add_argument('--ptm_path', type=str, default="./original_ptms_zoo/bert_base_chinese"
                    , help='pretrained model path(including config.json,model.bin and vocab.txt)'
                           'or token of huggingface model hub')
parser.add_argument('--cache_dir', type=str, default=None
                    , help='model_cache_path if using token of huggingface model hub')
parser.add_argument('--save_path', type=str, default="saved_models", help='ptm save_path')
# parameters of training
parser.add_argument('--gpu_id', type=str, default="0", help='choose a gpu to train if you have muti_gpus')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=2e-6)
parser.add_argument('--report_steps', type=int, default=300, help='每隔多少步，打印训练信息')
parser.add_argument('--save_steps', type=int, default=300, help='每隔多少步，判断是否保存最优模型')
parser.add_argument('--train', type=bool, default=True, help='是否是训练模型')

args = parser.parse_args()


if __name__ == '__main__':

    if args.gpu_id is None:
        raise print('选择GPU训练必须指定显卡')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证结果可复现

    """动态导包"""
    model_name = args.model
    data_utils_package = import_module('models.data_utils.' + model_name + '_data')
    trainer_package = import_module('models.trainer.train_eval_' + model_name)
    model_package = import_module('models.layers.' + model_name)
    """END"""

    # loading model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = model_package.Model(args).to(device)

    # loading data_utils
    start_time = time.time()
    print("Loading data_utils...")
    data_set = data_utils_package.bulid_dataset(args)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    time_dif = data_utils_package.get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    trainer_package.train(args, model, data_loader)
