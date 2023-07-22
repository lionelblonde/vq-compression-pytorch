import sklearn.metrics as skm


def compute_classif_eval_metrics(true_y, pred_y):

    metrics = {}
    accuracy = skm.accuracy_score(true_y, pred_y)
    balanced_accuracy = skm.balanced_accuracy_score(true_y.argmax(axis=1), pred_y.argmax(axis=1))
    zero_one_loss = skm.zero_one_loss(true_y, pred_y)
    hamming_loss = skm.hamming_loss(true_y, pred_y)
    precision = skm.precision_score(true_y, pred_y, average='samples')
    recall = skm.recall_score(true_y, pred_y, average='samples')
    f1 = skm.f1_score(true_y, pred_y, average='samples')
    f2 = skm.fbeta_score(true_y, pred_y, beta=2., average='samples')
    metrics.update({
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'zero_one_loss': zero_one_loss,
        'hamming_loss': hamming_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
    })

    return metrics
