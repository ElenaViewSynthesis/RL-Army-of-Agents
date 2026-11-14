"""CPU ownership repro for capture helper; fake graphs use weak buffer references."""
import ast
import contextlib
import gc
import types
import unittest
import weakref
from tests.support import ROOT
from utils.benchmark import _capture_graph


class Buffer:
    def clone(self):return Buffer()
    def copy_(self,other):pass


class Graph:
    def replay(self):
        if any(ref() is None for ref in self.refs):
            raise RuntimeError('captured buffer was released')


class GraphLifetime(unittest.TestCase):
    def capture(self,helper):
        state=types.SimpleNamespace(active=None)
        @contextlib.contextmanager
        def capture(graph):
            state.active=graph
            try:yield
            finally:state.active=None
        stream=types.SimpleNamespace(wait_stream=lambda other:None)
        cuda=types.SimpleNamespace(Stream=lambda:stream,current_stream=lambda:stream,
             stream=lambda stream:contextlib.nullcontext(),graph=capture,CUDAGraph=Graph,synchronize=lambda:None)
        def model(*inputs):
            output=Buffer()
            if state.active is not None:state.active.refs=[weakref.ref(x) for x in (*inputs,output)]
            return output
        return helper(model,[Buffer()],[Buffer()],types.SimpleNamespace(cuda=cuda),model,lambda a,b:None)

    def test_frozen_helper_reproduces_released_buffers(self):
        path=ROOT/'tests/fixtures/benchmark_capture_before_repair.py'
        tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_capture_graph')
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        entry,graph=self.capture(namespace['_capture_graph'])
        self.assertTrue(entry['graph_replay_ok'])
        gc.collect()
        with self.assertRaisesRegex(RuntimeError,'released'):graph.replay()

    def test_fixed_helper_retains_buffers_across_rotating_replays(self):
        handles=[self.capture(_capture_graph)[1] for _ in range(3)]
        refs=[ref for h in handles for ref in h.graph.refs]
        gc.collect()
        for _ in range(40):
            for h in handles:h.replay()
        self.assertTrue(all(ref() is not None for ref in refs))
        del h
        handles.clear();gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
