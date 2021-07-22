# for computing TP and FP rates for a single image
def data(X, Y, threshold):
    rates = [0, 0, 0, 0]
    for f in X:              # X is Positive data set
        if f >= threshold:
            rates[0] += 1
        else:
            rates[2] += 1
    for f in Y:             # Y is Negative data set
        if f >= threshold:
            rates[1] += 1
        else:
            rates[3] += 1

    TP_rate = rates[0] / (rates[0] + rates[2])
    FP_rate = rates[1] / (rates[1] + rates[3])
    return (TP_rate, FP_rate)
