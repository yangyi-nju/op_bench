"""Public CLI: one input JSON on stdin, one numeric observation on stdout."""
import json
import sys

from ops import softmax


request = json.load(sys.stdin)
values = softmax(request["values"])
print(json.dumps({"shape": [len(values)], "values": values}, allow_nan=False))
