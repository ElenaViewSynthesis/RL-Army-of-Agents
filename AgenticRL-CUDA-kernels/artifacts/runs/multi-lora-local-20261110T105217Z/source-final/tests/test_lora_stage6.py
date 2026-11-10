"""Stage 6 independent oracles, negative controls and synthetic interpreter probes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests.support import ROOT, scratch
from utils.lora_contracts import LORA_TASKS, slot_ranks, token_slots
from utils.task_catalog import case, get_task, select_cases, validate_usage, workload
from utils.task_suite import scaffold, select_tasks

try:
    import torch
except ImportError:
    torch = None


def tiny(dtype="float16", pattern="mixed", mixed=True, slices=3):
    return case(dtype, tokens=7, hidden=32, out=19, rank=16,
                slots=4, slices=slices, pattern=pattern, mixed=mixed)


class LoraContracts(unittest.TestCase):
    def test_catalog_coverage_and_permissive_sources(self):
        self.assertEqual(select_tasks(["stage6"]), list(LORA_TASKS))
        for name in LORA_TASKS:
            task = get_task(name)
            validate_usage(name, "train")
            self.assertFalse(task.eval_only)
            self.assertEqual(select_cases(name, "stratified"), task.cases)
            self.assertEqual(len({c.id for c in task.cases}), len(task.cases))
            for key, values in (("hidden", {2048,4096,5120,8192,14336}),
                                ("rank", {8,16,32,64,128,256}), ("slices", {1,2,3}),
                                ("slots", {1,4,16})):
                self.assertEqual({c.params[key] for c in task.cases}, values)
            if not name.endswith("bgmv"):
                p = select_cases(name, "acceptance")[0].params
                self.assertEqual((p["tokens"],p["hidden"],p["rank"],p["slots"],p["slices"]),
                                 (4096,14336,256,16,3))
            else:
                self.assertLessEqual(max(c.params["tokens"] for c in task.cases), 256)
            if name.endswith("sgmv"):
                self.assertGreaterEqual(min(c.params["tokens"] for c in task.cases),512)
            for c in task.cases:
                json.dumps(workload(name,c), allow_nan=False)

    def test_modal_graph_option_is_forwarded_without_launching_modal(self):
        import ast
        import contextlib
        import io
        from types import SimpleNamespace
        source=ast.parse((ROOT/"cloud/modal_tasks.py").read_text())
        main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        main.decorator_list=[]
        calls=[]
        namespace={"evaluate":SimpleNamespace(spawn=lambda *a: calls.append(a) or SimpleNamespace(object_id="synthetic")),
                   "json":json,"wait_for_call":lambda call:{"synthetic":True}}
        exec(compile(ast.Module(body=[main],type_ignores=[]),"modal_tasks_main","exec"),namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["main"](run_id="synthetic-no-launch",cuda_graphs=True)
        self.assertTrue(calls[0][-1])
        with self.assertRaisesRegex(ValueError,"requires benchmark"):
            namespace["main"](run_id="synthetic-no-launch",cuda_graphs=True,benchmark=False)
        self.assertEqual(len(calls),1)

    def test_malformed_numeric_diagnostic_cannot_change_reward(self):
        from unittest.mock import patch
        from utils.task_suite import evaluate_case
        def run(path,cfg,turns_used,profile_out):
            profile_out.update({"provenance/agent":1.,"provenance/library":0.,
                                "provenance/torch":0.,"provenance/memory":0.})
            return {"env/verify_ok":True,"env/t_eager_us":100.,"env/t_compile_us":80.,
                    "env/t_gen_us":90.,"reward/value":2.,"reward/base":2,
                    "reward/reason":"faster_than_eager"}
        def diagnostic(module,args,path,environment,deadline):
            (path/"lora-numerics.json").write_text("{invalid diagnostic")
            return 0,"","",False
        with scratch() as tmp, patch("utils.task_suite.cuda_unavailable",return_value=None), \
                patch("utils.task_suite.available_backends",return_value={"triton":None}), \
                patch("utils.task_suite.run_task",side_effect=run), \
                patch("utils.task_suite._stage",side_effect=diagnostic):
            item=get_task("lora_shrink_bgmv").cases[0]
            result=evaluate_case("lora_shrink_bgmv",item.id,"triton","honest",Path(tmp)/"case")
        self.assertEqual(result["suite/status"],"passed")
        self.assertEqual(result["reward/value"],2.)
        self.assertIsNone(result["artifact/lora_numerics"])
        self.assertEqual(result["suite/numerics_unavailable_reason"],"diagnostic_invalid_report")

    def test_torch_free_imports(self):
        p = subprocess.run([sys.executable, "-S", "-c",
            'import sys; from utils import lora_contracts, lora_reference; from utils.task_catalog import TASKS; assert "torch" not in sys.modules'],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_scaffolds_and_authored_names(self):
        from utils.backend import agent_kernel_names
        with scratch() as tmp:
            for name in LORA_TASKS:
                for backend in ("triton", "cuda"):
                    for variant in ("honest", "hack", "wrong"):
                        dest = scaffold(name, backend, variant, Path(tmp)/name/backend/variant)
                        self.assertTrue((dest/'model_new.py').is_file())
                        if variant == "honest":
                            self.assertTrue(any("lora_" in n for n in agent_kernel_names(dest,backend)))

    def test_active_work_excludes_padding_and_skips(self):
        p = tiny().params
        ranks = slot_ranks(p)
        total = sum(ranks[i] for i in token_slots(p) if i >= 0)
        w = workload("lora_shrink_bgmv", tiny())
        self.assertEqual(w["flops"], 2*p["slices"]*p["hidden"]*total)
        self.assertEqual(workload("lora_expand_bgmv",tiny(pattern="none"))["flops"],0)


@unittest.skipIf(torch is None, "Torch unavailable; Stage 6 numeric tests need Torch")
class LoraReferences(unittest.TestCase):
    def test_reference_disables_reduced_precision_matmul(self):
        previous=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            spec=importlib.util.spec_from_file_location("lora_precision_reference",ROOT/"tasks/lora_shrink_bgmv/model.py")
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.Model()
            self.assertEqual(torch.get_float32_matmul_precision(),"highest")
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_references_match_independent_loop(self):
        from tests.fixtures.lora_reference_loop import oracle
        from utils.lora_reference import make_inputs, forward
        torch.set_num_threads(2)
        for dtype in ("float16", "bfloat16"):
            for pattern in ("mixed", "none", "single", "distinct", "round_robin"):
                for name in LORA_TASKS:
                    with self.subTest(dtype=dtype,pattern=pattern,task=name):
                        args = make_inputs(name,tiny(dtype,pattern))
                        actual, expected = forward(name,*args), oracle(name,*args)
                        if isinstance(actual,tuple):
                            for a,b in zip(actual,expected):
                                self.assertTrue(torch.equal(a,b))
                        else:
                            torch.testing.assert_close(actual,expected,rtol=1e-2,atol=1e-2)
                            self.assertTrue(torch.isfinite(actual).all())

    def test_dyadic_oracle_is_bitwise_equal(self):
        from tests.fixtures.lora_reference_loop import oracle
        from utils.lora_reference import make_inputs,forward
        for name in LORA_TASKS:
            if name == "lora_segment_metadata":
                continue
            args=make_inputs(name,tiny())
            # Powers of two keep every sum exactly representable: checks equations,
            # scales, layouts, scatter offsets and masks without reduction-order noise.
            for value in args:
                if value.is_floating_point():
                    value.copy_(torch.where(torch.isnan(value),value,torch.full_like(value,.5)))
            torch.testing.assert_close(forward(name,*args),oracle(name,*args),atol=0,rtol=0)

    def test_poison_skip_and_half_diagnostic(self):
        from utils.lora_reference import make_inputs,forward
        from utils.lora_controls import control
        for name in LORA_TASKS:
            if name == "lora_segment_metadata":
                continue
            args = make_inputs(name,tiny())
            expected = forward(name,*args)
            for variant in ("wrong_padding","wrong_skip"):
                with self.subTest(task=name,variant=variant):
                    with self.assertRaises(AssertionError):
                        torch.testing.assert_close(control(name,variant,*args),expected,atol=1e-2,rtol=1e-2)
            half = control(name,"fp16_intermediate",*args)
            self.assertTrue(torch.isfinite(half).all())
            idx = args[4] if "fused" in name else args[2] if "shrink" in name else args[3]
            if "shrink" in name:
                self.assertTrue((expected[:,idx<0]==0).all())
            else:
                y = args[3] if "fused" in name else args[2]
                self.assertTrue(torch.equal(expected[idx<0],y[idx<0]))

    def test_accidental_adapter_zero_on_skips_is_detectable(self):
        from utils.lora_reference import make_inputs,forward
        for name in ("lora_shrink_bgmv","lora_expand_bgmv","lora_fused_sgmv"):
            args=make_inputs(name,tiny())
            correct=forward(name,*args)
            changed=[v.clone() for v in args]
            index=4 if "fused" in name else 2 if "shrink" in name else 3
            changed[index].clamp_min_(0)
            # This common implementation bug must fail the unchanged tolerance,
            # even though the skipped y rows contain large sentinel values.
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(forward(name,*changed),correct,atol=1e-2,rtol=1e-2)

    def test_rank64_half_distribution_is_diagnostic_only(self):
        from utils.lora_diagnostics import error_distribution
        from utils.lora_reference import make_inputs,forward
        from utils.lora_controls import control
        item=case("float16",tokens=3,hidden=32,out=19,rank=64,slots=4,slices=2,pattern="mixed",mixed=True)
        name="lora_fused_sgmv"
        args=make_inputs(name,item)
        expected=forward(name,*args)
        honest=error_distribution(expected,expected,args[0].dtype)
        half=error_distribution(control(name,"fp16_intermediate",*args),expected,args[0].dtype)
        self.assertEqual(honest["max_abs"],0)
        self.assertTrue(half["finite"])
        json.dumps({"honest":honest,"fp16_intermediate":half},allow_nan=False)
        # No assertion that half exceeds tolerance: that would be an unapproved gate.

    def test_cpu_interpreter(self):
        if importlib.util.find_spec('triton') is None or importlib.util.find_spec('numpy') is None:
            self.skipTest("Triton and NumPy required for CPU interpreter")
        p=subprocess.run([sys.executable,'-B',str(ROOT/'tests/lora_interpreter_probe.py')],cwd=ROOT,
                         env={**os.environ,'TRITON_INTERPRET':'1','CUDA_AGENT_DEVICE':'cpu',
                              'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
                         capture_output=True,text=True,timeout=300)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn('STAGE6_INTERPRETER_OK',p.stdout)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                     "CUDA GPU unavailable; Stage 6 builds, provenance and graph replay unverified")
class LoraGPU(unittest.TestCase):
    def test_graph_replay_after_slot_and_adapter_changes(self):
        from utils.lora_reference import make_inputs,forward
        from tests.lora_interpreter_probe import load_kernel
        candidate=load_kernel()
        for name in LORA_TASKS:
            args=[v.cuda() for v in make_inputs(name,tiny())]
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    candidate.forward(name,*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result=candidate.forward(name,*args)
            fresh=[v.cuda() for v in make_inputs(name,tiny(pattern="distinct"))]
            for dest,src in zip(args,fresh):
                dest.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            expected=forward(name,*args)
            if isinstance(result,tuple):
                for a,b in zip(result,expected):
                    self.assertTrue(torch.equal(a,b))
            else:
                torch.testing.assert_close(result,expected,atol=1e-2,rtol=1e-2)
