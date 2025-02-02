from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def calculate_metrics(reference, candidate):
    """
    计算 BLEU、METEOR 和 ROUGE-L 分数。
    """
    # BLEU
    bleu = sentence_bleu([reference], candidate)

    # METEOR
    meteor = meteor_score([reference], candidate)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(" ".join(reference), " ".join(candidate))["rougeL"].fmeasure

    return bleu, meteor, rouge_l
