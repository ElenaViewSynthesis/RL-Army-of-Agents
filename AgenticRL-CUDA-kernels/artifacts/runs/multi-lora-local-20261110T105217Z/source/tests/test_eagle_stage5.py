"""Stage 5 contract and correctness tests; synthetic probes are not GPU evidence."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests.support import ROOT, scratch
from utils.eagle_contracts import ancestor_rows, tree_parents
from utils.task_catalog import TASKS, case, get_task, validate_usage, workload
from utils.task_suite import scaffold, select_tasks, expected_outcome


NAMES = tuple(name for name, task in TASKS.items() if task.stage == 5)


class EagleContracts(unittest.TestCase):
    def test_grid_and_eval_gate(self):
        self.assertEqual(len(NAMES), 9)
        self.assertEqual(select_tasks(['stage5']), list(NAMES))
        for name in NAMES:
            with self.subTest(task=name):
                task = get_task(name)
                self.assertTrue(task.eval_only)
                validate_usage(name, 'eval')
                with self.assertRaisesRegex(ValueError, 'eval-only'):
                    validate_usage(name, 'train')
                self.assertEqual({c.params['b'] for c in task.cases}, {1, 8, 32})
                if name != 'eagle_candidate_topk':
                    self.assertEqual({c.params['tree'] for c in task.cases}, {'chain','wide','balanced','ragged'})
                    self.assertEqual(task.cases[0].params['n'], 1)
                    self.assertEqual(max(c.params['n'] for c in task.cases), 64)
                if 'attention' in name:
                    self.assertEqual({c.params['s'] for c in task.cases}, {512,4096,32768})
                    self.assertEqual({c.params['ratio'] for c in task.cases}, {1,4,8})
                    self.assertEqual({c.params['d'] for c in task.cases}, {64,128})
                if name in ('eagle_candidate_topk','eagle_accept_tail_sampled'):
                    self.assertEqual({c.params['v'] for c in task.cases}, {32000,128256,151936})

    def test_ancestor_shapes_and_depth(self):
        self.assertEqual(ancestor_rows([-1]), [[0]])
        self.assertEqual(ancestor_rows([-1,0,0]), [[0,-1,-1],[1,0,-1],[2,0,-1]])
        for shape in ('chain','wide','balanced','ragged'):
            parents = tree_parents(64,shape)
            self.assertTrue(all(-1 <= p < i for i,p in enumerate(parents)))
        self.assertEqual(ancestor_rows(tree_parents(4,'chain'))[-1], [3,2,1,0])

    def test_work_counts_and_logical_tensor_bytes(self):
        for name in NAMES:
            for item in get_task(name).cases:
                record = workload(name,item)
                self.assertIsInstance(record['flops'],int)
                self.assertGreaterEqual(record['flops'],0)
                self.assertGreater(record['logical_io_bytes'],0)
        record = workload('eagle_candidate_topk',case('bfloat16',b=1,h=2,v=3,top_k=1))
        self.assertEqual((record['flops'],record['logical_io_bytes']),(16,28))
        chain = workload('eagle_tree_verify_attention',case('bfloat16',b=1,n=3,tree='chain',hkv=1,ratio=1,d=2,s=4,window=4))
        wide = workload('eagle_tree_verify_attention',case('bfloat16',b=1,n=3,tree='wide',hkv=1,ratio=1,d=2,s=4,window=4))
        self.assertEqual(chain['flops'],4*2*(3*4+6))
        self.assertEqual(wide['flops'],4*2*(3*4+5))

    def test_every_fixture_scaffolds_and_chain_is_honest(self):
        for name in NAMES:
            variants = ['honest','hack','wrong'] + ([] if name=='eagle_candidate_topk' else ['wrong_ancestor'])
            for backend in ('cuda','triton'):
                for variant in variants:
                    with scratch('eagle-scaffold-') as tmp:
                        out=scaffold(name,backend,variant,Path(tmp)/'task')
                        self.assertTrue((out/'binding.cpp').exists())
                        compile((out/'model_new.py').read_text(),str(out/'model_new.py'),'exec')
        with scratch('eagle-chain-') as tmp:
            out=scaffold('eagle_gdn_tree_recurrence','triton','chain_replay',Path(tmp)/'task')
            self.assertIn('chain=True',(out/'model_new.py').read_text())
        self.assertTrue(expected_outcome({'env/verify_ok':True,'reward/base':1},'chain_replay'))
        self.assertTrue(expected_outcome({'env/verify_ok':False,'reward/value':-1,'reward/reason':'verification'},'wrong_ancestor'))

    def test_legal_sources_are_pinned_and_freeze_preserves_notices(self):
        source=json.loads((ROOT/'legal/eagle4/source.json').read_text())
        for name,metadata in source['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'legal/eagle4'/name).read_bytes()).hexdigest(),metadata['sha256'])
        from scripts.freeze_task_suite import freeze
        with scratch('eagle-freeze-') as tmp:
            manifest=freeze(Path(tmp)/'source',['eagle_tree_step'])
            for name in ('LICENSE','NOTICE','TRADEMARKS.md','source.json'):
                self.assertIn('legal/eagle4/'+name,manifest['files'])

    def test_pure_imports(self):
        p=subprocess.run([sys.executable,'-S','-c',"import sys;import utils.eagle_contracts,utils.eagle_reference;assert 'torch' not in sys.modules"],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch unavailable')
class EagleReferences(unittest.TestCase):
    def test_candidate_ties_rescoring_and_frontier_ties(self):
        from utils.eagle_reference import candidate_topk,tree_step
        ids,scores=candidate_topk(torch.ones(1,2,dtype=torch.bfloat16),torch.ones(3,2,dtype=torch.bfloat16),2)
        self.assertEqual(ids.tolist(),[[0,1]])
        self.assertEqual(scores.tolist(),[[2.,2.]])
        values,parents,paths,flat=tree_step(torch.zeros(1,3),torch.zeros(1,3,2),torch.tensor([[-1,0,0]],dtype=torch.int32))
        self.assertEqual(parents.tolist(),[[0,0,1]])
        self.assertEqual(flat.tolist(),[[0,1,2]])
        self.assertEqual(paths.tolist(),[[[3,0,-1,-1],[4,0,-1,-1],[5,1,0,-1]]])

    def test_full_window_attention_excludes_siblings_and_includes_self(self):
        from utils.eagle_reference import attention
        parents=torch.tensor([[-1,0,0]],dtype=torch.int32)
        q=torch.zeros(1,1,3,2,dtype=torch.bfloat16)
        k=torch.zeros(1,1,2,2,dtype=torch.bfloat16)
        tk=torch.zeros_like(q)
        v=torch.zeros_like(k);tv=torch.tensor([[[[2.,2.],[4.,4.],[100.,100.]]]],dtype=torch.bfloat16)
        result,paths=attention(q,k,v,tk,tv,parents,2)
        self.assertEqual(paths.tolist(),[[[0,-1,-1],[1,0,-1],[2,0,-1]]])
        torch.testing.assert_close(result.float()[0,0,1],torch.tensor([1.5,1.5]),atol=1e-2,rtol=1e-2)

    def test_conv_history_and_mamba_layout(self):
        from utils.eagle_reference import tree_conv
        x=torch.tensor([[[4.],[5.],[9.]]],dtype=torch.bfloat16)
        history=torch.tensor([[[1.],[2.],[3.]]],dtype=torch.bfloat16)
        parents=torch.tensor([[-1,0,0]],dtype=torch.int32)
        result,_,windows=tree_conv(x,history,torch.ones(1,4),torch.zeros(1),parents)
        self.assertEqual(windows.tolist(),[[[0,-1,-2,-3],[1,0,-1,-2],[2,0,-1,-2]]])
        self.assertEqual(result.tolist(),[[[10.],[14.],[18.]]])
        mamba=tree_conv(x.transpose(1,2).contiguous(),history,torch.ones(1,4),torch.zeros(1),parents,True)[0]
        self.assertTrue(torch.equal(mamba,result.transpose(1,2)))

    def test_recurrence_branches_inherit_parent_not_previous_node(self):
        from utils.eagle_reference import recurrence
        q=key=torch.ones(1,3,1,1)
        v=torch.tensor([[[[1.]],[[2.]],[[7.]]]])
        gates=torch.ones(1,3,1,2);gates[...,1]=.5
        output,states,_,_=recurrence(q,key,v,gates,torch.zeros(1,1,1,1),torch.tensor([[-1,0,0]],dtype=torch.int32))
        self.assertEqual(output.flatten().tolist(),[.5,1.25,3.75])

    def test_greedy_paths_and_exact_large_slots(self):
        from utils.eagle_reference import accept
        from utils.task_verification import check_equal
        result=accept(torch.tensor([[7,8,9]],dtype=torch.int32),torch.tensor([[-1,0,0]],dtype=torch.int32),torch.tensor([[7,9,1,4]],dtype=torch.int32),torch.tensor([[100000,100001,100002]],dtype=torch.int32))
        self.assertEqual(result[0].tolist(),[2])
        self.assertEqual(result[1].tolist(),[[0,2,-1]])
        self.assertEqual(result[2].tolist(),[4])
        wrong=list(result);wrong[3]=wrong[3].clone();wrong[3][0,0]+=1
        with self.assertRaises(AssertionError):check_equal(tuple(wrong),result)

    def test_sampled_rejection_residual_and_seeded_inputs(self):
        from utils.eagle_reference import accept_sampled,make_inputs
        tokens=torch.tensor([[0]],dtype=torch.int32);parents=torch.tensor([[-1]],dtype=torch.int32)
        target=torch.tensor([[[.1,.9],[.4,.6]]]);draft=torch.tensor([[[.8,.2],[.4,.6]]]);slots=torch.tensor([[100000]],dtype=torch.int32)
        uniforms=torch.tensor([[.9]]);bonus_uniforms=torch.tensor([[.1,.1]])
        result=accept_sampled(tokens,parents,target,draft,slots,uniforms,bonus_uniforms)
        self.assertEqual(result[0].tolist(),[0]);self.assertEqual(result[2].tolist(),[1]);self.assertEqual(result[-1].tolist(),[1.])
        same=accept_sampled(tokens,parents,target,draft,slots,uniforms,bonus_uniforms)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(result,same)))
        torch.manual_seed(5);a=make_inputs('eagle_accept_tail_sampled',get_task('eagle_accept_tail_sampled').cases[0])
        torch.manual_seed(5);b=make_inputs('eagle_accept_tail_sampled',get_task('eagle_accept_tail_sampled').cases[0])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))

    def test_hacks_verify_and_both_wrong_controls_fail(self):
        from unittest.mock import patch
        from utils.eagle_reference import make_inputs
        from utils.task_verification import check_equal
        for name in NAMES:
            params={'b':1,'n':3,'tree':'balanced','h':4,'v':17,'top_k':2,'hkv':1,'ratio':1,'d':4,'s':4,'window':4,'c':6,'heads':1,'dk':4,'dv':4,'seed':17}
            dtype='bfloat16' if name=='eagle_candidate_topk' or 'attention' in name or 'conv' in name else 'int32' if name=='eagle_accept_tail' else 'float32'
            inputs=make_inputs(name,case(dtype,**params))
            refpath=ROOT/'tasks'/name/'model.py'
            spec=importlib.util.spec_from_file_location('_control_ref_'+name,refpath)
            reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
            init=[2] if name=='eagle_candidate_topk' else [4] if 'attention' in name else []
            expected=reference.Model(*init)(*inputs)
            for fixture in ('hack','wrong','wrong_ancestor'):
                path=ROOT/'tasks'/name/f'model_new_{fixture}.py'
                if not path.exists():continue
                with self.subTest(task=name,fixture=fixture),patch.dict(sys.modules,{'model':reference}):
                    spec=importlib.util.spec_from_file_location('_control_candidate_'+name,path)
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                    actual=module.ModelNew(*init)(*inputs)
                    if fixture=='hack':check_equal(actual,expected)
                    else:
                        with self.assertRaises(AssertionError):check_equal(actual,expected)

    def test_logical_bytes_equal_actual_reference_tensors(self):
        import importlib.util
        for name in NAMES:
            # Synthetic small probes keep this unit test independent of large-case resources.
            params={'b':1,'n':3,'tree':'wide','h':4,'v':7,'top_k':2,'hkv':1,'ratio':1,'d':4,'s':4,'window':4,'c':6,'heads':1,'dk':4,'dv':4,'seed':17}
            dtype='bfloat16' if name=='eagle_candidate_topk' or 'attention' in name or 'conv' in name else 'int32' if name=='eagle_accept_tail' else 'float32'
            item=case(dtype,**params)
            from utils.eagle_reference import make_inputs
            inputs=make_inputs(name,item)
            path=ROOT/'tasks'/name/'model.py'
            spec=importlib.util.spec_from_file_location('_ref_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            init=[2] if name=='eagle_candidate_topk' else [4] if 'attention' in name else []
            result=module.Model(*init)(*inputs)
            actual=sum(t.numel()*t.element_size() for t in (*inputs,*result))
            self.assertEqual(workload(name,item)['logical_io_bytes'],actual,name)


@unittest.skipIf(torch is None or importlib.util.find_spec('triton') is None or importlib.util.find_spec('numpy') is None,
                 'CPU interpreter requires Torch, Triton and NumPy')
class EagleInterpreter(unittest.TestCase):
    def test_branching_and_chain_replay_against_references(self):
        command=[sys.executable,'-B',str(ROOT/'tests/eagle_interpreter_probe.py')]
        process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600,
                               env={**os.environ,'CUDA_AGENT_DEVICE':'cpu','TRITON_INTERPRET':'1',
                                    'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        target=os.environ.get('STAGE5_PROBE_OUTPUT')
        if target:
            Path(target).write_text(process.stdout+process.stderr)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertIn('STAGE5_INTERPRETER_PROBES_OK',process.stdout)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable; Stage 5 GPU correctness/provenance unverified')
class EagleGPU(unittest.TestCase):
    def test_triton_and_cuda_smoke_and_negative_controls(self):
        from utils.task_suite import run_suite
        with scratch('eagle-gpu-') as tmp:
            report=run_suite(['stage5'],Path(tmp)/'suite',variants=('all',),timeout_s=600)
            self.assertEqual(report['counts']['failed'],0)
            self.assertEqual(report['counts']['unavailable'],0)
