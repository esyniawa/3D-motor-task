import numpy as np

# sigmoid isn't perfectly symmetrical around its turning point
def normal_space(start, stop, num):
    def sigmoid(num):
        values = np.linspace(0, 20, num-1)
        res = 1/(1+np.exp(-values+5))
        return res

    if start < 0:
        offset = abs(start) + stop
    else:
        offset = start

    x = sigmoid(num)
    # uniform linspace
    y = np.linspace(0, offset, num-1)

    return np.append(x * y + start, stop)


def sin_space(start, stop, num):
    if start < 0:
        offset = abs(start) + stop
    else:
        offset = start

    x = np.sin(np.linspace(0, np.pi/2, num, endpoint=True))

    # uniform linspace
    y = np.linspace(0, offset, num, endpoint=True)

    return x * y + start