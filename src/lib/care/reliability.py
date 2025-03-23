import utility


def _criticality(s, p):
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


def calc(s, p):
    event_prediction = 0
    crit = _criticality(s, p)

    if crit > 72:
        event_prediction = 1

    return event_prediction
