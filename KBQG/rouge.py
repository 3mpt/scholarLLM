#!/usr/bin/env python
#
# 文件名 : rouge.py
#
# 描述 : 计算 ROUGE-L 指标，如 Lin 和 Hovey (2004) 所描述
#
# 创建日期 : 2015-01-07 06:03
# 作者 : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np


def my_lcs(string, sub):
    """
    计算两个分词字符串的最长公共子序列（LCS）
    :param string : list of str : 使用空格分隔的字符串的标记
    :param sub : list of str : 较短的字符串，也使用空格分隔
    :returns: length (int): 两个字符串之间的最长公共子序列的长度

    注意: my_lcs 只返回最长公共子序列的长度，而不是实际的 LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge:
    """
    类，用于计算 MS COCO 测试集的一组候选句子的 ROUGE-L 指标

    """

    def __init__(self):
        # vrama91: 根据与 Hovey 的讨论更新了以下值
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        计算给定一个候选句子和一组参考句子的 ROUGE-L 分数
        :param candidate: list of str : 要评估的候选句子
        :param refs: list of str : 特定图像的 COCO 参考句子
        :returns score: float (候选句子相对于参考句子的 ROUGE-L 分数)
        """
        assert len(candidate) == 1
        assert len(refs) > 0
        prec = []
        rec = []

        # 分词
        token_c = candidate[0].split(" ")

        for reference in refs:
            # 分词
            token_r = reference.split(" ")
            # 计算最长公共子序列
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta**2) * prec_max * rec_max) / float(
                rec_max + self.beta**2 * prec_max
            )
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        计算数据集的一组参考句子和候选句子的 ROUGE-L 分数
        由 evaluate_captions.py 调用
        :param hypo_for_image: dict : 候选 / 测试句子，键为 "图像名称"，值为 "分词句子" 的列表
        :param ref_for_image: dict : 参考 MS-COCO 句子，键为 "图像名称"，值为 "分词句子" 的列表
        :returns: average_score: float (通过为所有图像平均分数计算的平均 ROUGE-L 分数)
        """
        assert gts.keys() == res.keys()
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            score.append(self.calc_score(hypo, ref))

            # 安全检查
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"
