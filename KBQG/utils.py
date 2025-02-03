from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def calculate_metrics(reference, candidate):
    """
    计算 BLEU、METEOR 和 ROUGE-L 分数。
    """
    # BLEU
    bleu = sentence_bleu([reference], candidate)

    # METEOR
    meteor = meteor_score([reference], candidate)

    rouge = Rouge()

    # 计算 ROUGE-L 分数
    rouge_l = rouge.calc_score([" ".join(reference)], [" ".join(candidate)])  
    return bleu, meteor, rouge_l
