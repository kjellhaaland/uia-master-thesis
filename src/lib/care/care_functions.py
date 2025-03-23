import numpy as np
import src.lib.care.utility as utility


##### ----- Accuracy ----- #####

def calculate_accuracy(g, p):
    tp, fn, fp, tn = utility.get_counts(g, p)
    return tn / (fp + tn)


##### ----- Coverage ----- #####

def calculate_coverage(g, p):
    fb_score = utility.fb_score(g, p)
    return fb_score


##### ----- Earliness ----- #####

def weight_function(x):
    return np.where(x < 0.5, 1, -2 * x + 2)


def calculate_earliness(p):
    if len(p) == 0:
        return 0
    
    size = len(p)
    s1 = 0
    s2 = 0
    for i in range(size):
        x = i / (size - 1)
        w = weight_function(x)
        s1 += w * p[i]
        s2 += w
    return s1 / s2


##### ----- Reliability ----- #####

def criticality_function(s, p):
    n = len(s)
    crit = [0] * (n + 1)
    for i in range(1, n):
        if s[i - 1] == 0:
            if p[i - 1] == 1:
                crit[i] = crit[i - 1] + 1
            else:
                crit[i] = max(crit[i - 1] - 1, 0)
        else:
            crit[i] = crit[i - 1]
    return max(crit[1:n + 1], key=lambda x: x)  # return max from array


def calculate_reliability(s, p):
    event_prediction = 0
    crit = criticality_function(s, p)

    if crit > 72:
        event_prediction = 1

    return event_prediction
