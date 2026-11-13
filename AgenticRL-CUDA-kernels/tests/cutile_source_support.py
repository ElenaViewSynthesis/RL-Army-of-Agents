"""Source-only cuTile checks. No fake CPU kernel execution or GPU claims."""
import ast
from pathlib import Path

from tests.support import ROOT

CUTILE_TASKS = ('rmsnorm', 'row_softmax', 'cispo_clip_weight', 'fused_adamw', 'grpo_group_norm',
                'log_softmax_gather', 'residual_rmsnorm', 'gemm_div_sum_scale', 'gemm_tensor_core',
                'relu_squared', 'silu_and_mul', 'geglu_tanh', 'rotary_embedding')
TORCH_CALLS = {'torch.empty', 'torch.empty_like', 'torch.cuda.current_stream', 'torch.cuda.device'}
METADATA_CALLS = {'view', 'numel', 'is_contiguous', 'bit_length'}
# Export spellings inspected in the pinned cuda-tile 1.8.0 wheel, not a claim
# that using a valid export guarantees compilation on any particular target.
CT_EXPORTS_USED = {'kernel', 'Constant', 'arange', 'bid', 'gather', 'scatter', 'sum',
                   'max', 'exp', 'rsqrt', 'sqrt', 'log', 'minimum', 'maximum', 'where',
                   'full', 'load', 'store', 'mma', 'reshape', 'cdiv', 'PaddingMode', 'int32', 'float32', 'launch'}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    return ''


def task_trees(name):
    base = ROOT/'tasks'/name
    paths = (base/'model_new_cutile.py', base/'kernels/cutile/__init__.py',
             base/'kernels/cutile'/f'{name}.py')
    return [(path, ast.parse(path.read_text(), filename=str(path))) for path in paths]


def check_host_source(test, node, tensor_args=()):
    tensor_args = set(tensor_args) & {'x', 'weight', 'log_ratio', 'advantage', 'mask', 'rewards',
                                      'parameter', 'gradient', 'first_moment', 'second_moment',
                                      'a', 'b', 'bias', 'logits', 'token_ids', 'residual', 't'}
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = dotted(child.func)
            if name.startswith('torch.'):
                test.assertIn(name, TORCH_CALLS)
            if name.startswith('ct.'):
                test.assertEqual(name, 'ct.launch')
            if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                if child.func.value.id in tensor_args:
                    test.assertIn(child.func.attr, METADATA_CALLS)
            test.assertNotIn(name, ('ct.tune', 'ct.matmul', 'torch.nn.functional'))
        if isinstance(child, (ast.BinOp, ast.UnaryOp)):
            # Direct tensor arithmetic is banned; shape/numel arithmetic is metadata.
            operands = (child.left, child.right) if isinstance(child, ast.BinOp) else (child.operand,)
            for operand in operands:
                test.assertFalse(isinstance(operand, ast.Name) and operand.id in tensor_args)
        if isinstance(child, ast.Attribute):
            test.assertNotIn(child.attr, ('item', 'tolist', 'nonzero', 'tune'))
