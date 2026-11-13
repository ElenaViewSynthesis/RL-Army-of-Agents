"""Synthetic tests for the approved Stage 6 verification mode."""
import dataclasses
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.support import ROOT, scratch
from utils.config import RolloutConfig
from utils.lora_verification import evaluate_outputs, check_outputs
try:
    import torch
except ImportError:
    torch = None


class GateRouting(unittest.TestCase):
    def test_defaults_cli_and_torch_free(self):
        self.assertFalse(RolloutConfig().relative_error_bounds)
        legacy_positional=RolloutConfig("triton",10,False,True)
        self.assertTrue(legacy_positional.benchmark)
        self.assertFalse(legacy_positional.relative_error_bounds)
        args=RolloutConfig.add_args(__import__('argparse').ArgumentParser()).parse_args(['--relative-error-bounds'])
        self.assertTrue(RolloutConfig.from_args(args).relative_error_bounds)
        result=subprocess.run([sys.executable,'-S','-c',
            'import sys;import utils.lora_verification,utils.config;assert "torch" not in sys.modules'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_rollout_selects_only_opted_in_verifier(self):
        from utils.rollout_env import run_task
        for relative,integer,module in ((False,False,'utils.verification'),(False,True,'utils.task_verification'),
                                       (True,True,'utils.lora_verification')):
            seen=[]
            def stage(mod,*args):
                seen.append(mod)
                return (0,'compile ok','',False) if len(seen)==1 else (1,'','synthetic stop',False)
            with scratch() as tmp, patch('utils.rollout_env._stage',side_effect=stage):
                result=run_task(tmp,dataclasses.replace(RolloutConfig(),backend='triton',
                    exact_integer_outputs=integer,relative_error_bounds=relative))
            self.assertEqual(seen,['utils.compile',module])
            self.assertEqual(result['reward/reason'],'verification')

    def test_suite_enables_gate_only_for_stage6_arithmetic(self):
        from utils.task_catalog import get_task
        from utils.task_suite import evaluate_case
        for name,enabled in (('gemm_div_sum_scale',False),('row_softmax',False),
                             ('rmsnorm',False),('grpo_group_norm',False),
                             ('residual_rmsnorm',False),('eagle_tree_step',False),
                             ('lora_shrink_bgmv',True),('lora_segment_metadata',False)):
            captured=[]
            def run(path,cfg,**kwargs):
                captured.append(cfg)
                return {'env/verify_ok':False,'reward/value':-1,'reward/reason':'verification'}
            with scratch() as tmp,patch('utils.task_suite.cuda_unavailable',return_value=None), \
                    patch('utils.task_suite.available_backends',return_value={'triton':None}), \
                    patch('utils.task_suite.run_task',side_effect=run):
                evaluate_case(name,get_task(name).cases[0].id,'triton','wrong',Path(tmp)/'case',
                              cfg=dataclasses.replace(RolloutConfig(),relative_error_bounds=True))
            self.assertEqual(captured[0].relative_error_bounds,enabled)


@unittest.skipIf(torch is None,'Torch required for synthetic relative-bound tests')
class GateMath(unittest.TestCase):
    def args(self,dtype=torch.float16 if torch else None,idx=None):
        idx=torch.tensor([0,-1] if idx is None else idx,dtype=torch.int32)
        return [torch.ones(len(idx),2,dtype=dtype),None,None,torch.zeros(len(idx),2,dtype=dtype),idx]

    def test_boundary_and_over_boundary_replaces_old_assert_close(self):
        for dtype,maximum in ((torch.float16,200.),(torch.bfloat16,100.)):
            args=self.args(dtype,idx=[0])
            expected=torch.tensor([[maximum,0.]],dtype=torch.float32)
            actual=expected.clone();actual[0,1]=1.
            check_outputs('lora_shrink_bgmv',actual[None],expected[None],[args[0],None,args[4]])
            actual[0,1]=1.00001
            with self.assertRaises(AssertionError):check_outputs('lora_shrink_bgmv',actual[None],expected[None],[args[0],None,args[4]])
            from utils.task_verification import check_equal as legacy_check
            with self.assertRaises(AssertionError):
                at_bound=expected.clone();at_bound[0,1]=1.
                legacy_check(at_bound,expected)
            # Shrink uses activation coefficient even though its output is fp32.
            shargs=[args[0],None,args[4]]
            check_outputs('lora_shrink_bgmv',expected[None],expected[None],shargs)

    def test_skips_one_ulp_small_large_and_signed_zero_are_bit_exact(self):
        for dtype,ulp in ((torch.float16,4.),(torch.bfloat16,32.)):
            args=self.args(dtype);args[3][1]=4096
            expected=args[3].clone();expected[0]=1
            for change in (ulp,32.,10000.):
                actual=expected.clone();actual[1,0]+=change
                report=evaluate_outputs('lora_fused_sgmv',actual,expected,args)
                self.assertFalse(report['passed'])
                self.assertEqual(report['outputs'][0]['max_abs_ref'],1.)
                self.assertFalse(report['outputs'][0]['skipped_rows_bit_exact'])
            args[3][1]=0;expected[1]=0;actual=expected.clone();actual[1,0]=-0.
            self.assertFalse(evaluate_outputs('lora_fused_sgmv',actual,expected,args)['passed'])

    def test_nonfinite_padding_and_zero_reference(self):
        args=self.args(idx=[0]);expected=torch.zeros(1,1,16)
        for value in (float('nan'),float('inf'),-float('inf'),.00001):
            actual=expected.clone();actual[0,0,15]=value
            with self.assertRaises(AssertionError):check_outputs('lora_shrink_bgmv',actual,expected,[args[0],None,args[4]])
        args=self.args(idx=[-1,-1]);args[3].fill_(4096)
        check_outputs('lora_fused_sgmv',args[3].clone(),args[3],args)
        actual=args[3].clone();actual[0,0]+=4
        with self.assertRaises(AssertionError):check_outputs('lora_fused_sgmv',actual,args[3],args)
        report=evaluate_outputs('lora_fused_sgmv',args[3],args[3],args)
        self.assertEqual(report['outputs'][0]['checked_elements'],0)
        json.dumps(report,allow_nan=False)

    def test_integer_metadata_and_structure_stay_exact(self):
        args=[torch.tensor([0,-1],dtype=torch.int32),torch.tensor([8],dtype=torch.int32)]
        expected=(torch.tensor([0,1],dtype=torch.int32),)
        check_outputs('lora_segment_metadata',expected,expected,args)
        with self.assertRaises(AssertionError):check_outputs('lora_segment_metadata',(expected[0]+1,),expected,args)

    def test_mixed_ranks_and_controls(self):
        from utils.lora_reference import make_inputs,forward
        from utils.lora_controls import control
        from utils.lora_contracts import LORA_TASKS
        from tests.test_lora_stage6 import tiny
        for dtype in ('float16','bfloat16'):
            for name in LORA_TASKS:
                args=make_inputs(name,tiny(dtype));expected=forward(name,*args)
                check_outputs(name,expected,expected,args)
                if name=='lora_segment_metadata':continue
                for variant in ('wrong_skip','wrong_padding'):
                    with self.assertRaises(AssertionError):check_outputs(name,control(name,variant,*args),expected,args)

    def test_case_json_retains_gate_failure_without_using_report_for_rewards(self):
        from utils.task_catalog import get_task
        from utils.task_suite import evaluate_case
        def run(path,cfg,**kwargs):
            target=Path(path)/'.cache/relative-error-bounds.json'
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps({'passed':False,'checks':[{'passed':False,'outputs':[{'bound':1.,'max_abs_ref':200.,'passed':False}]}]}))
            return {'env/verify_ok':False,'reward/value':-1.,'reward/reason':'verification'}
        with scratch() as tmp,patch('utils.task_suite.cuda_unavailable',return_value=None), \
                patch('utils.task_suite.available_backends',return_value={'triton':None}), \
                patch('utils.task_suite.run_task',side_effect=run):
            target=Path(tmp)/'case'
            result=evaluate_case('lora_shrink_bgmv',get_task('lora_shrink_bgmv').cases[0].id,'triton','wrong',target)
            payload=json.loads((target/'suite-case.json').read_text())
            self.assertFalse(payload['verification_bounds']['passed'])
            self.assertEqual(result['reward/value'],-1.)
            self.assertIn('artifact/relative_error_bounds',result)
