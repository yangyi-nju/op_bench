"""Scripted full-framework integration control; this does not call a model.

Run on the selected GPU host after building the trusted, unpatched baseline.
The scripted repair is deliberately known in advance. It validates native Git,
public feedback, incremental builds, submission capture and independent grading;
it is not an Agent capability measurement or task-admission decision.
"""
import argparse
import json
from pathlib import Path

from op_bench.benchmark.experiment import AgentSpec, run_attempt
from op_bench.benchmark.io import write_json
from op_bench.benchmark.spec import TaskSpec
from op_bench.benchmark.verification import verify_evaluation


SCRIPT = r'''
import json
from pathlib import Path
import subprocess
import sys

def command(argv):
    return subprocess.run(argv, text=True, capture_output=True)

history = command(['git', 'rev-list', '--all', '--count'])
assert history.returncode == 0 and history.stdout.strip() == '1', history
remotes = command(['git', 'remote'])
assert remotes.returncode == 0 and not remotes.stdout.strip(), remotes
original = command(['git', 'cat-file', '-e', '2409b49a33c0ef594d89f9f477d56abad47e65bf^{commit}'])
assert original.returncode != 0, 'Original upstream Git object entered the solver'
print(json.dumps({'event': 'fresh_git_workspace', 'commit_count': 1,
                  'remotes': [], 'upstream_base_object_absent': True}), flush=True)

reproduce = [sys.executable, '-I', 'opbench_public/reproduce.py']
before = command(reproduce)
print(before.stdout, end='', flush=True)
print(before.stderr, end='', file=sys.stderr, flush=True)
assert before.returncode != 0 and 'CUDA log_softmax disagrees with the independent scalar reference' in before.stderr
print(json.dumps({'event': 'public_reproducer', 'state': 'baseline',
                  'outcome': 'declared_numeric_defect'}), flush=True)

source = Path('aten/src/ATen/native/cuda/SoftMax.cu')
text = source.read_text()
# Match the two repaired loops; other loops legitimately use the same decrement.
for following in ['    data += blockDim.x;', '    input += blockDim.x;\n    output += blockDim.x;']:
    old = '    size -= blockDim.x;\n' + following
    assert text.count(old) == 1, repr(old)
    text = text.replace(old, '    size -= blockDim.x > size ? size : blockDim.x;\n' + following)
source.write_text(text)
subprocess.run([sys.executable, 'setup.py', 'build'], check=True)
subprocess.run(reproduce, check=True)
print(json.dumps({'event': 'public_reproducer', 'state': 'repaired',
                  'outcome': 'passed'}), flush=True)
subprocess.run(['git', 'add', str(source)], check=True)
subprocess.run(['git', '-c', 'user.name=OpBench integration control',
                '-c', 'user.email=control@example.invalid', 'commit', '-qm',
                'Scripted integration control repair'], check=True)
print(json.dumps({'event': 'local_commit_created'}), flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--baseline-artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = TaskSpec.load(args.task)
    agent = AgentSpec('scripted-cuda-source-control', (task.environment.python, '-I', '-c', SCRIPT),
                      timeout_sec=900, metadata={'purpose': 'scripted_integration_control', 'model': None})
    record = run_attempt(task, agent, args.output, baseline_artifact=args.baseline_artifact)
    evaluation = record.get('evaluation') or {}
    verified = verify_evaluation(args.output / 'evaluation') if (args.output / 'evaluation/result.json').is_file() else {'valid': False}
    passed = (record['terminal_status'] == 'finished' and record['submission']['status'] == 'frozen'
              and evaluation.get('resolved') is True and verified['valid']
              and not record.get('cleanup_errors'))
    summary = {'kind': 'scripted_full_source_native_control', 'passed': passed,
               'task_identity': task.identity_dict(), 'model_called': False,
               'terminal_status': record['terminal_status'], 'submission': record['submission'],
               'evaluation_status': evaluation.get('status'), 'verification': verified,
               'information_boundary': record.get('information_boundary'),
               'limitation': 'Known scripted repair; not a model result or formal task admission.'}
    write_json(args.output / 'canary-summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
