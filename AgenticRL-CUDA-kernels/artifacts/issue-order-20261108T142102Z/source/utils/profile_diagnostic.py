import argparse
import json
from pathlib import Path

from utils.backend import BACKENDS, agent_kernel_names, get_spec
from utils.profiling import benchmark_model, get_prof_ctx, initialize_models, profile_models


PREPARATIONS = ("none", "empty-prime", "eager-prime", "warmup-all")
ORDERS = ("gen,eager,compile", "eager,compile,gen")


def capture(backend, order, preparation):
    import torch

    torch.manual_seed(0)
    eager, generated, eager_inputs, generated_inputs = initialize_models()
    models = {"gen": generated, "eager": eager, "compile": torch.compile(eager)}
    inputs = {"gen": generated_inputs, "eager": eager_inputs, "compile": eager_inputs}
    warmup = get_spec(backend).profile_warmup_iters
    primer = None
    if preparation == "warmup-all":
        with torch.no_grad():
            for session in ("gen", "eager", "compile"):
                for _ in range(warmup):
                    models[session](*inputs[session])
                torch.cuda.synchronize()
    elif preparation == "empty-prime":
        torch.cuda.synchronize()
        with get_prof_ctx() as ctx:
            torch.cuda.synchronize()
        primer = {"intentional_empty": True, "cuda_events": sum(
            event.device_type.name == "CUDA" for event in ctx.events())}
    elif preparation == "eager-prime":
        timing, events = benchmark_model(eager, eager_inputs, warmup, 10, True)
        primer = {"intentional_empty": False, "timing_us": timing,
                  "cuda_events": len(events), "events": events}
    elif preparation != "none":
        raise ValueError("unknown preparation")
    profile = profile_models(models, inputs, backend, 10, order=order,
                             agent_names=agent_kernel_names(Path.cwd(), backend))
    profile["profile/diagnostic"] = True
    return {"preparation": preparation, "primer": primer, "profile": profile,
            "diagnostic_only": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--preparation", choices=PREPARATIONS, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.backend, args.order, args.preparation), allow_nan=False))


if __name__ == "__main__":
    main()
