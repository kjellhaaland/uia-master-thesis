import random

b = 0.5


def get_counts(g, p):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(g)):
        if p[i] == 1 and g[i] == 1:
            tp += 1
        if p[i] == 0 and g[i] == 1:
            fn += 1
        if p[i] == 1 and g[i] == 0:
            fp += 1
        if p[i] == 0 and g[i] == 0:
            tn += 1
    return tp, fn, fp, tn


def fb_score(g, p):
    tp, fn, fp, tn = get_counts(g, p)

    # correct formula taken from
    # https://papers.nips.cc/paper_files/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf

    a = ((1 + b ** 2) * (tp + tn) + fp + b ** 2 * fn)

    if a == 0:
        return 0.0

    return (1 + b ** 2) * (tp + tn) / a
