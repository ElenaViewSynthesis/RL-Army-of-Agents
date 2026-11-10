"""Symbol evidence captured from H200 run 20261106T185236Z; no GPU needed."""
import unittest
from utils.backend import BACKENDS, GENERIC_KERNEL_NAMES, agent_kernel_names
from utils.provenance import classify_kernel
from tests.support import scratch, make_spoof_task

# Verbatim profiler names captured from run 20261106T185236Z.
CAPTURED = {
    'cuda': ('axpby_kernel', 'void axpby_kernel<256>(float*, float const*, float const*, float, int)'),
    'triton': ('axpby_kernel', 'axpby_kernel'),
    'tilelang': ('axpby_tilelang', 'axpby_tilelang_kernel'),
    'cutile': ('axpby_cutile_kernel', 'axpby_cutile_kernel_Kt1_A1f32_1i16t1_p16_A1f32_1i16t1_p16_A1f32_1i16t1_p16_Sf32_I128'),
    'cutedsl': ('axpby_cute_kernel', 'kernel_cutlass_axpby_cute_kernel_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensor00o1512011105121____0'),
}


class SymbolProvenanceTests(unittest.TestCase):
    def test_captured_symbols(self):
        for backend, (authored, event) in CAPTURED.items():
            with self.subTest(backend=backend):
                self.assertEqual(classify_kernel(event, [authored], backend), 'agent')

    def test_captured_symbols_under_other_backends_discovered_names(self):
        for origin, (_, event) in CAPTURED.items():
            for backend, (authored, _) in CAPTURED.items():
                if origin == backend:
                    continue
                with self.subTest(origin=origin, backend=backend):
                    # Preserve the explicitly approved CUDA library allowance.
                    expected = 'library' if origin == 'cutedsl' and backend == 'cuda' else 'torch'
                    self.assertEqual(classify_kernel(event, [authored], backend), expected)

    def test_torch_namespace_spoofs_fail_on_every_backend(self):
        for backend in BACKENDS:
            for authored, event in [
                *[(n, 'at::native::vectorized_elementwise_kernel') for n in ('kernel', 'main', 'add')],
                ('elementwise_kernel', 'at::native::elementwise_kernel<128, 4, …>'),
                ('triton_poi_fused_add_0', 'triton_poi_fused_add_0'),
                ('authored_name', 'aten::authored_name()'),
                ('authored_name', 'c10::authored_name()'),
                *[(f'triton_{kind}_fused_custom', f'triton_{kind}_fused_custom') for kind in ('poi', 'per', 'red', 'tem')],
            ]:
                with self.subTest(backend=backend, event=event, authored=authored):
                    self.assertEqual(classify_kernel(event, [authored], backend), 'torch')

    def test_generic_names_rejected_even_when_pattern_matches(self):
        for backend, spec in BACKENDS.items():
            for n in GENERIC_KERNEL_NAMES | {'abc', 'x', ''}:
                event = {'cuda': n+'()', 'triton': n, 'tilelang': n+'_kernel',
                         'cutile': n+'_Kt1_args', 'cutedsl': 'kernel_cutlass_'+n+'_args'}[backend]
                with self.subTest(backend=backend, name=n):
                    # CUDA retains its separate allowed-library policy.
                    expected = 'library' if backend == 'cuda' and 'cutlass' in event else 'torch'
                    self.assertEqual(classify_kernel(event, [n], backend), expected)

    def test_discovery_rejects_generic_dummy_on_all_backends(self):
        for backend in BACKENDS:
            with scratch() as tmp:
                task = make_spoof_task(tmp, backend)
                self.assertEqual(agent_kernel_names(task, backend), [])

    def test_all_rejected_names_are_removed_from_source_discovery(self):
        from pathlib import Path
        for backend, spec in BACKENDS.items():
            with scratch() as tmp:
                root = Path(tmp)
                path = root / spec.kernel_glob.replace('*', 'names')
                path.parent.mkdir(parents=True, exist_ok=True)
                names = sorted(GENERIC_KERNEL_NAMES | {'abc', 'x', 'valid_kernel'})
                if backend == 'cuda':
                    source = '\n'.join('__global__ void '+n+'() {}' for n in names)
                else:
                    source = '\n'.join('@'+spec.decorators[0]+'\ndef '+n+'(): pass' for n in names)
                path.write_text(source)
                self.assertEqual(agent_kernel_names(root, backend), ['valid_kernel'])

    def test_library_memory_and_unknown_on_all_backends(self):
        for backend in BACKENDS:
            for event in ('cublasLtMatmul', 'cutlass_gemm', 'cudnn_conv'):
                self.assertEqual(classify_kernel(event, [], backend),
                                 'library' if backend == 'cuda' else 'torch')
            for event in ('Memcpy HtoD', 'c10::memset', 'aten::memcpy'):
                self.assertEqual(classify_kernel(event, ['memcpy'], backend), 'memory')
            self.assertEqual(classify_kernel('unknown', [], backend), 'torch')

    def test_regex_escapes_authored_names_and_boundaries(self):
        for backend, (_, event) in CAPTURED.items():
            self.assertEqual(classify_kernel(event, ['.*kernel'], backend), 'torch')
        self.assertEqual(classify_kernel('void not_axpby_kernel<1>()', ['axpby_kernel'], 'cuda'), 'torch')
        self.assertEqual(classify_kernel('axpby_kernel_bad', ['axpby_kernel'], 'triton'), 'torch')
        self.assertEqual(classify_kernel('prefix_axpby_tilelang_kernel', ['axpby_tilelang'], 'tilelang'), 'torch')
