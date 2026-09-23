"""Small original softmax implementation used to exercise numeric scoring."""
import math


def softmax(values):
    numerators = [math.exp(value) for value in values]
    return [value / sum(numerators) for value in numerators]
