from pathlib import Path

def edit(name, fn):
 p=Path(name);b=p.read_bytes();newline='\r\n' if b'\r\n' in b else '\n';s=b.decode().replace('\r\n','\n');out=fn(s);assert out!=s,name;p.write_bytes(out.replace('\n',newline).encode())

def config(s):
 s=s.replace('    relative_error_bounds: bool = False\n','    relative_error_bounds: bool = False\n    gpu_lock: bool = True\n    gpu_lock_timeout_s: float = 600.0\n')
 s=s.replace('("timeout_s", "provenance_threshold", "speedup_margin",','("timeout_s", "gpu_lock_timeout_s", "provenance_threshold", "speedup_margin",')
 s=s.replace('        if self.timeout_s <= 0:\n            raise ValueError("timeout_s must be positive")','        if self.timeout_s <= 0:\n            raise ValueError("timeout_s must be positive")\n        if isinstance(self.gpu_lock_timeout_s, bool) or self.gpu_lock_timeout_s <= 0:\n            raise ValueError("gpu_lock_timeout_s must be positive")')
 s=s.replace('"exact_integer_outputs", "relative_error_bounds"):','"exact_integer_outputs", "relative_error_bounds", "gpu_lock"):')
 return s
edit('utils/config.py',config)

def rollout(s):
 s=s.replace('import argparse\n','import argparse\nimport contextlib\n',1)
 s=s.replace('from utils.config import RolloutConfig\n','from utils.config import RolloutConfig\nfrom utils.gpu_lock import MeasurementLock\n',1)
 s=s.replace('The deadline covers compile + verification + profiling together. Each child','The deadline covers all stages, excluding GPU lock wait. Each child')
 s=s.replace('        "meta/seed": cfg.seed,','        "meta/seed": cfg.seed,\n        "meta/gpu_uuid": None,\n        "meta/gpu_name": None,\n        "meta/gpu_uuid_reason": "measurement stages not reached",\n        "meta/gpu_lock_key": None,\n        "meta/gpu_lock_wait_s": 0.0,\n        "meta/gpu_lock": "not_reached",')
 s=s.replace('    stage = "workdir"\n    try:','    stage = "workdir"\n    measurement_stack = contextlib.ExitStack()\n    try:')
 s=s.replace('        for stage, module, args in stages:\n            stage_started', '''        for stage, module, args in stages:
            if stage == "profiling":
                lock = measurement_stack.enter_context(MeasurementLock(
                    env, cfg.gpu_lock, cfg.gpu_lock_timeout_s))
                result.update(lock.metadata)
                if lock.busy:
                    return finish("gpu_busy")
                # Only waiting is exempt; discovery and every stage still count.
                deadline += lock.metadata["meta/gpu_lock_wait_s"]
            stage_started''')
 s=s.replace('        for field in ("t_eager_us", "t_compile_us", "t_gen_us", "provenance"):', '        measurement_stack.close()\n        for field in ("t_eager_us", "t_compile_us", "t_gen_us", "provenance"):',1)
 s=s.replace('        return finish(stage)\n\n\ndef main():','        return finish(stage)\n    finally:\n        measurement_stack.close()\n\n\ndef main():')
 return s
edit('utils/rollout_env.py',rollout)

def grpo(s):
 s=s.replace('Each task must have a whole number of groups; incomplete groups are errors.','Each task must have a whole number of submitted groups; incomplete groups are\nerrors. Only gpu_busy environment failures are omitted within their original\ngroup before normalization/filtering; groups never borrow rows from each other.')
 s=s.replace('    kept, dropped = filter_groups(groups, cfg.filter_tied_groups)','''    excluded = sum(r.get("reward/reason") == "gpu_busy" for group in groups for r in group)
    eligible = [[r for r in group if r.get("reward/reason") != "gpu_busy"] for group in groups]
    eligible = [group for group in eligible if group]
    kept, dropped = filter_groups(eligible, cfg.filter_tied_groups)''')
 s=s.replace('dropped / len(groups) if groups else 0.0','dropped / len(eligible) if eligible else 0.0')
 s=s.replace('        "adv/mean": mean(advantages),','        "adv/gpu_busy_excluded": excluded,\n        "adv/mean": mean(advantages),')
 return s
edit('utils/grpo.py',grpo)

def catalog(s):
 s=s.replace('    if selection == "stratified" and task.stage == 6:\n        return task.cases','''    if selection == "stratified" and task.stage == 6:
        # Paired catalog rows: six stable representatives, never the full grid.
        # single/rank8/1 slice; mixed; distinct; none; rank256/3 slices;
        # then the bf16 widest acceptance shape. Both dtypes and mixed ranks.
        indices = (0, 3, 8, 11, 12, len(task.cases) - 1)
        return tuple(task.cases[i] for i in indices)''')
 return s
edit('utils/task_catalog.py',catalog)
edit('tests/support.py',lambda s:s.replace('    "log/compile_tail", "log/stderr_tail"})','    "meta/gpu_uuid", "meta/gpu_name", "meta/gpu_uuid_reason",\n    "meta/gpu_lock_key", "meta/gpu_lock_wait_s", "meta/gpu_lock",\n    "log/compile_tail", "log/stderr_tail"})'))
edit('tests/test_lora_stage6.py',lambda s:s.replace('            self.assertEqual(select_cases(name, "stratified"), task.cases)','            self.assertEqual(len(select_cases(name, "stratified")), 6)'))
edit('utils/task_suite.py',lambda s:s.replace('    graded = apply_task_policy(raw, profile, config, turns_used=1)','    deadline += raw.get("meta/gpu_lock_wait_s", 0.0)\n    graded = apply_task_policy(raw, profile, config, turns_used=1)'))

def adr(s):
 lines=s.splitlines(True);out=[]
 for line in lines:
  out.append(line)
  if line.startswith('| `') and '/relative_error_bounds`' in line:
   out.append(line.replace('/relative_error_bounds`','/gpu_lock`').replace('Stage 6 relative verification/config; forced true only for Stage 6 arithmetic','GPU measurement serialization/config'))
   out.append(line.replace('/relative_error_bounds`','/gpu_lock_timeout_s`').replace('| bool | 1 |','| float | s |').replace('Stage 6 relative verification/config; forced true only for Stage 6 arithmetic','GPU wait cap/config'))
  if line.startswith('| `result:meta/task_id`'):
   for key,typ,unit,null,role in [
    ('gpu_uuid','str','UUID','CPU/unavailable/not reached','measured identity'),
    ('gpu_name','str','device name','CPU/unavailable/not reached','diagnostic'),
    ('gpu_uuid_reason','str','reason','UUID resolved','diagnostic'),
    ('gpu_lock_key','str','lock identity','CPU/not reached','diagnostic'),
    ('gpu_lock_wait_s','float','s','never; zero uncontended/no lock','diagnostic; excluded from deadline only'),
    ('gpu_lock','str','lock status','never','diagnostic')]:
    out.append(f'| `result:meta/{key}` | table | meta | {typ} | {unit} | {null} | {role} | environment |\n')
  if line.startswith('| `') and '/adv/std`' in line:
   out.append(line.replace('/adv/std`','/adv/gpu_busy_excluded`').replace('| float | 1 |','| int | rollouts |').replace('| diagnostic |','| excluded environment failure count |'))
 return ''.join(out)
edit('Architecture-decision/0002-wandb-tracking-contract.md',adr)
