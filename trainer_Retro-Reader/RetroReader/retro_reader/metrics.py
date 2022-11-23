import datasets
from transformers.trainer_utils import EvalPrediction
import json

accuracy = datasets.load_metric("accuracy").compute
precision = datasets.load_metric("precision").compute
recall = datasets.load_metric("recall").compute
f1 = datasets.load_metric("f1").compute
squad_v2 = datasets.load_metric("squad_v2").compute


def compute_classification_metric(p: EvalPrediction):
    """
    'predictions': Value(dtype='int32', id=None)
    'references': Value(dtype='int32', id=None)
    """
    predictions = p.predictions.argmax(axis=1)
    references = p.label_ids
    metric = accuracy(predictions=predictions, references=references)
    metric.update(precision(predictions=predictions, references=references))
    metric.update(recall(predictions=predictions, references=references))
    metric.update(f1(predictions=predictions, references=references))
    return metric


def compute_squad_v2(p: EvalPrediction):
    """
    'predictions': {
        'id': Value(dtype='string', id=None),
        'prediction_text': Value(dtype='string', id=None),
        'no_answer_probability': Value(dtype='float32', id=None)
    }
    'references': {
        'id': Value(dtype='string', id=None),
        'answers': Sequence(
            feature={
                'text': Value(dtype='string', id=None), 
                'answer_start': Value(dtype='int32', id=None)
            },
            length=-1, id=None
        )
    }
    """
    predictions = p.predictions
    references = p.label_ids

    squad_v2_plus_am = squad_v2(predictions=predictions, references=references)

    s = 0
    gt_pred = dict()
    for t in zip(predictions, references):
        pred, ids = t[0]['prediction_text'], t[0]['id']
        if t[1]['answers']['text']:
            gt = t[1]['answers']['text'][0]
        else:
            gt = ""
        am = compute_am_score(pred, gt, 0.5)
        gt_pred[ids] = {"score": [am, {"prediction": pred, "answer": gt}]}
        s += am
    am_score = s / len(predictions) * 100
    squad_v2_plus_am["am_score"] = am_score

    return squad_v2_plus_am


def compute_am_score(pred, gt, prob=0.5):
    if gt == pred:  # 정답과 완전히 일치하는 경우 (EM=1)
        return 1
    elif (len(gt) == 0) or (len(pred) == 0):  # 하나는 null string인데 하나는 답이 있는 경우
        return 0

    s, l = pred, gt.strip()
    if len(gt) < len(pred):
        s, l = gt, pred

    idx = l.find(s.split(" ")[0])

    hit = 0
    if idx != -1:
        for s_ in s:  # 문자 단위로 비교
            if l[idx] == s_:
                hit += 1
            else:
                break

            if idx < len(l) - 1:
                idx += 1
            else:
                break

    v = hit / len(l)
    if v >= prob:
        return 1
    else:
        return 0
