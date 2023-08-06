import torch


class Metrics(object):

    @staticmethod
    def hamming_loss(pred, answer, weights, use_weights=False):
        # for multi-label
        # for a given sample with multiple predicted labels
        # computes the fraction of incorrectly predicted labels
        # it penalizes only the individual labels, and does not
        # impose for the entire set of labels to match
        if not use_weights:
            weights = torch.ones_like(weights)  # replace with just ones
        else:
            weights /= weights.sum()  # just in case
        pred, answer = pred.byte(), answer.byte()  # not an inplace method!
        out = (
            (torch.bitwise_and(
                pred, answer
            ) * weights).sum(dim=1) /
            (torch.bitwise_or(
                pred, answer
            ) * weights).sum(dim=1)
        ).float().mean()
        if out.isnan():  # denom can cause NaN issues
            out = torch.tensor(1.0)
        # this is the score; the loss is simply its complement to 1
        return 1. - out

    @staticmethod
    def zero_one_loss(pred, answer, weights, use_weights=False):
        # for multi-label
        # returns the fraction of misclassifications
        # incorrect means there is not a 100% match with
        # the true set of labels => less forgiving that Hamming
        if not use_weights:
            weights = torch.ones_like(weights)  # replace with just ones
        else:
            weights /= weights.sum()  # just in case
        pred, answer = pred.byte(), answer.byte()  # not an inplace method!
        out = (
            (torch.bitwise_and(
                pred, answer
            ) * weights).prod(dim=1) /
            (torch.bitwise_or(
                pred, answer
            ) * weights).sum(dim=1)
        ).float().mean()
        # this is the score; the loss is simply its complement to 1
        return 1. - out

    @staticmethod
    def accu_prec_reca_spec(pred, answer, weights, use_weights=False):
        if not use_weights:
            weights = torch.ones_like(weights)  # replace with just ones
        else:
            weights /= weights.sum()  # just in case

        # unpack the sizes for us to use
        n, num_labels = pred.size()  # arbitrary

        tot_corr = 0
        tot_corr_true = 0
        tot_corr_false = 0
        tot_p_i_true = 0
        tot_a_i_true = 0
        tot_a_i_false = 0

        # loop over every label
        for i in range(num_labels):

            p_i, a_i, w_i = pred[:, i], answer[:, i], weights[i]

            p_i_true = p_i.sum(dim=0)
            a_i_true = a_i.sum(dim=0)
            tot_p_i_true += p_i_true
            tot_a_i_true += a_i_true

            a_i_false = n - a_i_true
            tot_a_i_false += a_i_false

            corr = (p_i == a_i).sum(dim=0).float()
            corr_true = (p_i * a_i).sum(dim=0).float()
            corr_false = ((1 - p_i) * (1 - a_i)).sum(dim=0).float()

            tot_corr += corr * w_i
            tot_corr_true += corr_true * w_i
            tot_corr_false += corr_false * w_i

        # assemble the metrics
        accu = tot_corr / (n * num_labels)
        prec = tot_corr_true / tot_p_i_true
        reca = tot_corr_true / tot_a_i_true
        spec = tot_corr_false / tot_a_i_false

        return accu, prec, reca, spec

    @staticmethod
    def accuracy(*args):
        accu, _, _, _ = Metrics.accu_prec_reca_spec(*args)
        return accu

    @staticmethod
    def precision(*args):
        _, prec, _, _ = Metrics.accu_prec_reca_spec(*args)
        return prec

    @staticmethod
    def recall(*args):
        _, _, reca, _ = Metrics.accu_prec_reca_spec(*args)
        return reca

    @staticmethod
    def fbeta(prec, reca, beta):
        return (1. + beta**2) * prec * reca / ((beta**2 * prec) + reca)

    @staticmethod
    def f1(*args):
        # weights precision and recall equally
        _, prec, reca, _ = Metrics.accu_prec_reca_spec(*args)
        return Metrics.fbeta(prec, reca, beta=1.)

    @staticmethod
    def f2(*args):
        # weights recall higher than precision
        _, prec, reca, _ = Metrics.accu_prec_reca_spec(*args)
        return Metrics.fbeta(prec, reca, beta=2.)

    @staticmethod
    def specificity(*args):
        _, _, _, spec = Metrics.accu_prec_reca_spec(*args)
        return spec

    @staticmethod
    def balanced_accuracy(*args):
        _, _, reca, spec = Metrics.accu_prec_reca_spec(*args)
        # arithm. mean between recall and specificity
        b_accu = (reca + spec) / 2
        return b_accu


def compute_metrics(pred_y, true_y, weights):
    # classification-specific eval metrics
    metrics = {}
    keys = [
        'hamming_loss', 'zero_one_loss',
        'accuracy', 'precision', 'recall', 'f1', 'f2',
        'specificity', 'balanced_accuracy',
    ]
    metric_factory = Metrics()
    for k in keys:
        metrics[k] = getattr(metric_factory, k)(
            pred_y, true_y, weights,
        ).item()

    return metrics


class MetricsAggregator(object):

    def __init__(self, num_labels, batch_size):
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        # reset the stats
        self.n = 0  # num of samples seen since last reset
        self.tot_corr = 0
        self.tot_corr_true = 0
        self.tot_corr_false = 0
        self.tot_p_i_true = 0
        self.tot_a_i_true = 0
        self.tot_a_i_false = 0

    def step(self, pred, answer):
        # integrate the stats to the system

        # add the new samples to the count
        self.n += pred.size(dim=0)  # arbitrary

        # loop over every label
        for i in range(self.num_labels):

            p_i, a_i = pred[:, i], answer[:, i]

            p_i_true = p_i.sum(dim=0)
            a_i_true = a_i.sum(dim=0)
            self.tot_p_i_true += p_i_true
            self.tot_a_i_true += a_i_true

            a_i_false = self.batch_size - a_i_true
            self.tot_a_i_false += a_i_false

            corr = (p_i == a_i).sum(dim=0).float()
            corr_true = (p_i * a_i).sum(dim=0).float()
            corr_false = ((1 - p_i) * (1 - a_i)).sum(dim=0).float()

            self.tot_corr += corr
            self.tot_corr_true += corr_true
            self.tot_corr_false += corr_false

    def compute(self):
        # assemble the metrics
        accu = self.tot_corr / (self.n * self.num_labels)
        prec = self.tot_corr_true / self.tot_p_i_true
        reca = self.tot_corr_true / self.tot_a_i_true
        spec = self.tot_corr_false / self.tot_a_i_false

        f1 = Metrics.fbeta(prec, reca, beta=1.)
        f2 = Metrics.fbeta(prec, reca, beta=2.)
        b_accu = (reca + spec) / 2

        metrics = {
            'accuracy': accu,
            'precision': prec,
            'recall': reca,
            'f1': f1,
            'f2': f2,
            'specificity': spec,
            'balanced_accuracy': b_accu,
        }
        metrics = {k: v.item() for k, v in metrics.items()}

        return metrics

