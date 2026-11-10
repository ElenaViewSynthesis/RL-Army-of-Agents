from pathlib import Path
legacy='''- Verification uses five fresh random inputs; floating outputs keep atol=rtol=1e-2.
  The progressive suite requires exact equality for integer outputs, including
  token IDs and prefix sums. `utils.task_verification` exposes the same strict
  check directly; stock `utils.verification` and AXPBY defaults stay unchanged.'''
arithmetic='''- Verification uses five fresh random inputs with the approved Stage 6 bound:
  max absolute error <= 0.005 * checked max reference for fp16 activations,
  or 0.01 for bf16, including fp32 shrink buffers. Skipped y rows are bit-exact
  and excluded from the reductions; zero reference is exact and non-finites
  fail. `utils.lora_verification` replaces the old float check for these tasks.
  Integers stay exact. Stock and Stages 0–5 keep their existing checks.'''
metadata='''- Verification uses five fresh random inputs and exact integer equality for
  every metadata output. This task has no floating relative gate. Stock and
  Stages 0–5 keep their existing checks.'''
p=Path('scripts/task_readme.py');s=p.read_text();mark='    return f"""# `{task.name}`'
pos=s.index(mark)
s=s[:pos]+f'    verification_note = {legacy!r}\n    if task.stage == 6:\n        verification_note = {metadata!r} if task.name == "lora_segment_metadata" else {arithmetic!r}\n'+s[pos:]
s=s.replace(legacy,'{verification_note}')
# The inserted repr contains escaped newlines and is not altered by the multiline replacement.
p.write_text(s)
from utils.lora_contracts import LORA_TASKS
for name in LORA_TASKS:
 p=Path('tasks')/name/'README.md';text=p.read_text()
 assert legacy in text,name
 p.write_text(text.replace(legacy,metadata if name=='lora_segment_metadata' else arithmetic))
