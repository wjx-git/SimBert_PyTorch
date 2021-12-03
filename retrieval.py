"""
@Project ：SimBert_PyTorch
@File ：retrieval.py
@IDE ：PyCharm
@Author ：wujx
SimBERT 相似度任务测试
"""
import torch
import numpy as np
from models.layers.simbert import Model

from run import args


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i == 0:
                continue
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = Model(args).to(device)
# 加载数据集
test_data = load_data('corpus/lcqmc/test.tsv')
database = test_data[:100]


def test(test_data):
    """
    测试模型的文本相似度效果
    :return:
    """
    a_token, b_token, labels = [], [], []
    a_token_pads, b_token_pads = [], []

    for d in test_data:
        a_token.append(d[0])
        a_token_pads.append('[PAD]' * len(d[0]))
        b_token.append(d[1])
        b_token_pads.append('[PAD]' * len(d[1]))
        labels.append(d[2])

    a_vecs, _, _ = model([a_token, a_token_pads])
    b_vecs, _, _ = model([b_token, b_token_pads])

    labels = np.array(labels)

    a_vecs = a_vecs / (a_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    b_vecs = b_vecs / (b_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    sims = (a_vecs * b_vecs).sum(axis=1).detach().numpy()

    # 以0.95为阈值
    print('acc:', ((sims > 0.95) == labels.astype('bool')).mean())


def batch_retrieval(text, topn=10):
    """
    从数据库中检索与text最相似文本
    :param text: string
    :param topn:
    :return:
    """
    a_token, b_token = [], []
    a_token_pads, b_token_pads = [], []
    texts = []

    for d in database:
        a_token.append(d[0])
        a_token.append(d[1])
        a_token_pads.append('[PAD]' * len(d[0]))
        a_token_pads.append('[PAD]' * len(d[1]))
        texts.extend(d[:2])

    a_vecs, _, _ = model([a_token, a_token_pads])
    b_vec, _, _ = model([[text], ['[PAD]' * len(text)]])

    a_vecs = a_vecs / (a_vecs**2).sum(axis=1, keepdims=True)**0.5
    b_vec = b_vec / (b_vec**2).sum(axis=1, keepdims=True)**0.5
    sims = torch.squeeze(torch.mm(a_vecs, b_vec.t())).detach().numpy()
    return [(texts[i], sims[i]) for i in sims.argsort()[::-1][:topn]]


if __name__ == '__main__':
    print(batch_retrieval('多吃韭菜有什么好处', 2))
