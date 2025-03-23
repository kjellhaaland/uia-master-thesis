import utility


def calc(g, p):
    tp, fn, fp, tn = utility.get_counts(g, p)
    return tn / (fp + tn)
