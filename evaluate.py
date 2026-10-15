import json
import pandas as pd
import re
import string

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(path):
    df = pd.read_json(path, lines=True)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    for prediction, answer in zip(df.pred.tolist(), df.answer.tolist()):

        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    acc = sum(acc_list) * 100
    hit = sum(hit_list) * 100 
    f1 = sum(f1_list) * 100
    pre = sum(precission_list) * 100
    recall = sum(recall_list) * 100 


    return acc, hit, f1, pre, recall, len(hit_list)

