import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def calculate_metrics(reference, candidate):
    # BLEU
    bleu = sentence_bleu([reference.split()], candidate.split())

    # METEOR
    meteor = meteor_score([reference], candidate)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, candidate)["rougeL"].fmeasure

    return bleu, meteor, rouge_l
