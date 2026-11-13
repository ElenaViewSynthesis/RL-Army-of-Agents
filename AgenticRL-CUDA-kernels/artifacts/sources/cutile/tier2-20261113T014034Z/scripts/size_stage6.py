"""CPU symbolic storage probe of largest Stage 6 references and Torch fixtures.

Meta tensors allocate no payloads. Counts describe live Python-visible storage,
not GPU allocator peaks, hidden operator workspaces, or authored kernel latency.
"""

import argparse
import json
import time
import weakref

from scripts.plan_stage6 import size_case
from utils.lora_contracts import LORA_TASKS
from utils.task_catalog import select_cases


def probe():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from utils.lora_controls import control
    from utils.lora_reference import forward, make_inputs
    from utils.task_fixtures import wrong_output

    class StorageTrace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.refs, self.storages = {}, {}
            self.live = self.peak = 0

        def record(self, value):
            if not isinstance(value, torch.Tensor) or id(value) in self.refs:
                return
            storage = value.untyped_storage()
            key, size, ident = storage._cdata, storage.nbytes(), id(value)
            if key not in self.storages:
                self.storages[key] = [0, size]
                self.live += size
                self.peak = max(self.live, self.peak)
            self.storages[key][0] += 1
            def release(ref):
                self.refs.pop(ident, None)
                entry = self.storages[key]
                entry[0] -= 1
                if not entry[0]:
                    self.live -= entry[1]
                    del self.storages[key]
            self.refs[ident] = weakref.ref(value, release)

        def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
            output = fn(*args, **(kwargs or {}))
            for value in tree_flatten(output)[0]:
                self.record(value)
            return output

    results = []
    for name in LORA_TASKS:
        item = select_cases(name, 'acceptance')[0]
        sizes = size_case(name, item)
        variants = ('reference', 'honest_storage_proxy', 'hack', 'wrong')
        if name != 'lora_segment_metadata':
            variants += ('wrong_padding', 'wrong_skip', 'fp16_intermediate')
        for variant in variants:
            start = time.monotonic()
            trace = StorageTrace()
            with trace, torch.device('meta'), torch.no_grad():
                inputs = make_inputs(name, item)
                builder_peak = trace.peak
                input_bytes = sum(t.numel()*t.element_size() for t in inputs)
                assert input_bytes == sizes['input_bytes'], (name, input_bytes, sizes['input_bytes'])
                trace.peak = trace.live
                expected = forward(name, *inputs)
                if variant in ('reference', 'honest_storage_proxy', 'hack'):
                    actual = forward(name, *inputs)
                elif variant == 'wrong':
                    actual = wrong_output(forward(name, *inputs))
                else:
                    actual = control(name, variant, *inputs)
                shapes = [list(t.shape) for t in tree_flatten(actual)[0]]
                output_bytes = sum(t.numel()*t.element_size() for t in tree_flatten(actual)[0])
                assert output_bytes == sizes['output_bytes']
                reference_peak = trace.peak
                del actual, expected, inputs
            assert max(builder_peak, reference_peak) < sizes['gpu_planning_budget_bytes']
            row = dict(task=name, case=item.id, fixture=variant, input_bytes=input_bytes,
                       output_shapes=shapes, output_bytes=output_bytes,
                       symbolic_builder_peak_bytes=builder_peak,
                       symbolic_reference_plus_candidate_peak_bytes=reference_peak,
                       cpu_symbolic_probe_wall_s=time.monotonic()-start)
            results.append(row)
            print(json.dumps(row), flush=True)
    return dict(schema='stage6-symbolic-sizing-v1', torch_version=torch.__version__,
                scope='Meta payloads only: no GPU execution or measured allocator peak',
                honest_proxy='Authored kernels require GPU. Reference working storage is used as a conservative proxy; native sources allocate outputs and fused fp32 scratch.',
                gate='Full-array float64 masks/errors/sort are budgeted separately by plan_stage6; dynamic boolean selection is not executed on meta.',
                rows=results)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
