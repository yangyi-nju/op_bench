"""Assemble the candidate without changing the existing CPU build environment."""
from copy import deepcopy
import json
from pathlib import Path

from generate_oracles import batches

ROOT = Path(__file__).resolve().parent.parent


def main():
    environment_task = json.loads((ROOT.parent / 'layer_norm_cpu/task.json').read_text())
    if (environment_task['source']['revision'] != '240aa77ad01c4f0cd9b2417748272f2f617c112f'
            or environment_task['environment']['revision'] != '240aa77-cpu-gcc12-openblas-r3'):
        raise ValueError('Review the candidate revision before changing its shared source/build environment')
    worker = (ROOT / 'public/numeric_worker.py').read_text()
    loader = worker.split('\ndef main():')[0]
    reproduce = (ROOT / 'public/reproduce.py').read_text().replace(
        'from numeric_worker import load_source_torch', loader)
    task = {
        'schema_version': 2, 'task_id': 'opbench-vector-norm-compile-cpu-001',
        'task_revision': 'candidate.2', 'scoring_revision': 'candidate.1',
        'scope': 'operator_integration', 'defect_group': 'pytorch:vector-norm-singleton-pow-overflow',
        'statement': (
            'CPU Inductor compilation of torch.linalg.vector_norm can produce nonfinite or zero results '
            'when the selected reduction axes contain one element and ord is +41 or -41, even though '
            'the correct result is the finite absolute value of that element. A public reproducer uses '
            'a float32 (4,1,4) tensor containing +/-16 and +/-0.0625, reducing dim=1 with ord=-41. '
            'Repair the complete provided framework source so the actual compiled API returns the '
            'correct values, shape and dtype; preserve ordinary multi-element norms and float64 behavior. '
            'The declared domain is nonempty rank-2/rank-3 CPU float32/float64 tensors with finite '
            'values in [-16,16], contiguous or noncontiguous layouts, dim=None or unique nonnegative '
            'dimension indices, and keepdim=False/True. Singleton reduction groups cover ord=-41/+41; '
            'ordinary multi-element regressions cover ord=-1,0,1,2,+inf,-inf. Negative orders follow '
            'the usual zero-input norm convention. Float32 singleton comparisons with ord=-41/+41 use '
            'abs(actual-reference)<=1e-6+1e-6*abs(reference); other float32 checks use '
            '2e-5+2e-6*abs(reference), and float64 checks use 1e-12+1e-12*abs(reference). '
            'Outputs retain the input floating dtype and CPU device, with the declared axes removed '
            'or retained at length one. Complex/low-precision types, explicit dtype conversion, empty '
            'or symbolic shapes, negative dimension indices, gradients, out=, CUDA and performance '
            'are outside this candidate contract. '
            'Use torch.compile(fullgraph=True, backend="inductor", dynamic=False) and execute its '
            'returned callable. CPU C++ kernel generation/execution must remain enabled; replacing '
            'compilation by eager execution or routing the API around compilation does not meet the '
            'contract. The published fixed transport invokes the source-built API and records generated '
            'kernel and loaded compiler-cache library diagnostics. These diagnostics are not universal '
            'attestation against hostile candidate code. The transport encodes each output as a finite '
            'flag and value (zero placeholder when nonfinite); a separate controller compares these '
            'observations with private independent scalar references. The transport contains no oracle '
            'or numerical pass/fail verdict. '
            'Build the complete configured framework with python3 -I /opt/opbench-layernorm/build_source.py '
            '/workspace. Both Torch Python modules and native libraries must load from '
            '/workspace/build/opbench-package. Dependencies are preinstalled; builds and execution stay '
            'offline, with compiler caches inside the workspace runtime. Public commands include the '
            'reproducer and full build. Any source or build change consistent with this contract is '
            'allowed; there is no reference-patch file whitelist. This is a relocated upstream-derived '
            'candidate on commit 240aa77ad01c4f0cd9b2417748272f2f617c112f, not the original PR parent '
            '06e9deabb623e004eb6024e703a976c5748d51e6, and is not formally admitted.'
        ),
        'source': deepcopy(environment_task['source']),
        'environment': deepcopy(environment_task['environment']),
        'grader_dir': 'grader',
        'public_commands': [
            ['{python}', '-I', '/opt/opbench-layernorm/build_source.py', '{workspace}'],
            ['{python}', '-I', '-c', reproduce, '{workspace}'],
        ],
        'tests': [{'id': name, 'group': batch['group'], 'kind': 'numeric', 'oracle': name + '.json',
                   'argv': ['{python}', '-I', '-c', worker, '{workspace}'], 'timeout_sec': 900}
                  for name, batch in batches().items()],
        'metadata': {
            'status': 'candidate', 'admission_status': 'not_admitted', 'formal_benchmark_member': False,
            'origin': 'upstream_issue_relocated_to_pre_fix_base',
            'upstream_pr': 'https://github.com/pytorch/pytorch/pull/144073',
            'upstream_issue': 'https://github.com/pytorch/pytorch/issues/143960',
            'original_parent': '06e9deabb623e004eb6024e703a976c5748d51e6',
            'upstream_fix': '957faaadca78ec453d60f2fe986c1191e2e7c5b6',
            'relocation_review': 'migration_review.md', 'validation_records': 'validation/',
            'oracle_method': 'Independent 80-digit Decimal scalar norm, explicit singleton absolute value, then target-dtype rounding; no torch imports.',
            'environment_reuse': 'Same complete environment/source declaration as the CPU LayerNorm r3 candidate; environment names identify a shared build, not task semantics.',
            'trust_limit': 'Fixed published driver imports candidate Torch; compiler counters, import paths and process maps are ordinary diagnostics and can be spoofed by hostile code. Numerical comparison is performed outside the candidate process.',
        },
    }
    (ROOT / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
