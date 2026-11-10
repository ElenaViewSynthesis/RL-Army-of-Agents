from pathlib import Path

def edit(path, fn):
    p=Path(path);raw=p.read_bytes();s=raw.decode().replace('\r\n','\n')
    out=fn(s)
    p.write_bytes(out.replace('\n','\r\n').encode() if b'\r\n' in raw else out.encode())
edit('utils/config.py',lambda s:s.replace('    exact_integer_outputs: bool = False','    exact_integer_outputs: bool = False\n    relative_error_bounds: bool = False').replace('"exact_integer_outputs"):', '"exact_integer_outputs", "relative_error_bounds"):'))
edit('utils/rollout_env.py',lambda s:s.replace('verification_module = "utils.task_verification" if cfg.exact_integer_outputs else "utils.verification"','verification_module = ("utils.lora_verification" if cfg.relative_error_bounds else\n                               "utils.task_verification" if cfg.exact_integer_outputs else "utils.verification")'))
edit('utils/task_suite.py',lambda s:s.replace('        exact_integer_outputs=True,','        exact_integer_outputs=True,\n        relative_error_bounds=spec.stage == 6 and task_name != "lora_segment_metadata",').replace('    payload = {"schema":', '''    bounds_path = path / ".cache" / "relative-error-bounds.json"
    bounds = None
    if config.relative_error_bounds and bounds_path.is_file():
        try:
            bounds = json.loads(bounds_path.read_text())
            result["artifact/relative_error_bounds"] = str(bounds_path)
        except (OSError, ValueError):
            pass  # Artifact content never determines a reward.
    payload = {"schema":''').replace('"profile": profile, "result": result,', '"profile": profile, "result": result,\n               **({"verification_bounds": bounds} if bounds is not None else {}),'))
edit('Architecture-decision/0002-wandb-tracking-contract.md',lambda s:'\n'.join(line+'\n'+line.replace('exact_integer_outputs','relative_error_bounds').replace('forced true only by progressive suite','forced true only for Stage 6 arithmetic').replace('verification/config;', 'Stage 6 relative verification/config;') if line.startswith('| `') and 'exact_integer_outputs' in line else line for line in s.split('\n')))
edit('utils/benchmark.py',lambda s:s.replace('import argparse\n','import argparse\nfrom dataclasses import dataclass\n').replace('def _capture_graph(','''@dataclass
class CapturedGraph:
    """Own the static buffers for the entire replay lifetime, including outputs."""
    graph: object
    inputs: object
    outputs: object

    def replay(self):
        return self.graph.replay()


def _capture_graph(''').replace('        return entry, graph\n','        return entry, CapturedGraph(graph, static_inputs, static_out)\n'))
