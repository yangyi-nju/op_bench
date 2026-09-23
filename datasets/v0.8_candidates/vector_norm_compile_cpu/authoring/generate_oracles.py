"""Independent scalar norm calculations; no Torch or candidate imports."""
from decimal import Decimal, localcontext
import itertools
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent


def operation(name, shape, *, dim, order, dtype='float32', keepdim=False, layout='contiguous', values=None):
    choices = (-16, -4, -1, -0.25, -0.0625, 0.0625, 0.25, 1, 4, 16)
    size = math.prod(shape)
    return {'name': name, 'shape': shape, 'dim': dim, 'ord': order, 'dtype': dtype,
            'keepdim': keepdim, 'layout': layout,
            'values': list(values) if values is not None else [choices[index % len(choices)] for index in range(size)]}


def batches():
    return {
        'singleton_reductions': {'group': 'fail_to_pass', 'atol': 1e-6, 'rtol': 1e-6, 'operations': [
            operation('negative_batch_axis', [4, 1, 4], dim=[1], order=-41),
            operation('positive_batch_axis_keepdim', [4, 1, 4], dim=[1], order=41, keepdim=True),
            operation('negative_axis0', [1, 8], dim=[0], order=-41),
            operation('positive_two_axes', [1, 6, 1], dim=[0, 2], order=41),
            operation('negative_strided_axis', [4, 1, 4], dim=[1], order=-41, layout='noncontiguous'),
            operation('global_one_element', [1, 1], dim=None, order=41, values=[16]),
        ]},
        'float32_regressions': {'group': 'pass_to_pass', 'atol': 2e-5, 'rtol': 2e-6, 'operations': [
            operation('ordinary_l2', [2, 4], dim=[1], order=2),
            operation('negative_multi_element', [2, 4], dim=[1], order=-1, keepdim=True),
            operation('l1_noncontiguous', [3, 4], dim=[0], order=1, layout='noncontiguous'),
            operation('infinity', [2, 4], dim=None, order='inf'),
            operation('negative_infinity', [2, 4], dim=[1], order='-inf'),
            operation('zero_order', [2, 4], dim=[1], order=0, values=[0, 1, -1, 0, 2, 0, 3, 4]),
            operation('ordinary_singleton', [2, 1, 3], dim=[1], order=2),
        ]},
        'float64_regressions': {'group': 'pass_to_pass', 'atol': 1e-12, 'rtol': 1e-12, 'operations': [
            operation('double_negative_singleton', [4, 1, 4], dim=[1], order=-41, dtype='float64'),
            operation('double_positive_singleton', [4, 1, 4], dim=[1], order=41, dtype='float64', keepdim=True),
            operation('double_multi_l2', [2, 4], dim=[0, 1], order=2, dtype='float64'),
        ]},
    }


def reference(item):
    axes = set(range(len(item['shape']))) if item['dim'] is None else set(item['dim'])
    groups = {}
    for coordinate, number in zip(itertools.product(*(range(size) for size in item['shape'])), item['values']):
        key = tuple(value for axis, value in enumerate(coordinate) if axis not in axes)
        groups.setdefault(key, []).append(abs(Decimal(str(number))))
    result = []
    with localcontext() as context:
        context.prec = 80
        order = Decimal(str(item['ord']))
        for values in groups.values():
            if order == 0:
                value = Decimal(sum(number != 0 for number in values))
            elif order == Decimal('Infinity'):
                value = max(values)
            elif order == Decimal('-Infinity'):
                value = min(values)
            elif len(values) == 1:
                value = values[0]
            elif order < 0 and 0 in values:
                value = Decimal(0)
            else:
                total = sum((number ** int(order) for number in values), Decimal(0))
                value = (total.ln() / order).exp() if total else Decimal(0)
            number = float(value)
            if item['dtype'] == 'float32':
                number = struct.unpack('f', struct.pack('f', number))[0]
            result.extend((1.0, number))
    return result


def main():
    (ROOT / 'grader').mkdir(exist_ok=True)
    for name, batch in batches().items():
        expected = [value for item in batch['operations'] for value in reference(item)]
        oracle = {'schema_version': 1, 'input': {'operations': batch['operations']},
                  'expected': {'shape': [len(expected)], 'values': expected},
                  'atol': batch['atol'], 'rtol': batch['rtol']}
        (ROOT / 'grader' / (name + '.json')).write_text(json.dumps(oracle, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
