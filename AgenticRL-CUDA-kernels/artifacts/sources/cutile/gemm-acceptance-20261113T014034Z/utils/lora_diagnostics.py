"""Supplemental distributions; enforcement lives in utils.lora_verification."""


def error_distribution(actual, reference, dtype):
    import torch
    error = (actual.float()-reference.float()).abs().flatten()
    finite = torch.isfinite(error)
    maximum = reference.float().abs().max().item()
    if not bool(finite.all()):
        return {"finite": False, "max_abs": None, "p50_abs": None, "p99_abs": None,
                "max_abs_over_max_ref": None, "original_relative_bound": .01 if dtype == torch.bfloat16 else .005}
    q = torch.quantile(error, torch.tensor([.5,.99],device=error.device)).tolist()
    return {"finite": True, "max_abs": error.max().item(), "p50_abs": q[0], "p99_abs": q[1],
            "max_abs_over_max_ref": error.max().item()/maximum if maximum else None,
            "original_relative_bound": .01 if dtype == torch.bfloat16 else .005}


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import torch
    from utils.verification import initialize_models, build_inputs
    from utils.lora_reference import forward
    from utils.lora_verification import evaluate_outputs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    name=os.environ['CUDA_AGENT_TASK']
    torch.manual_seed(17)
    reference,candidate=initialize_models()
    inputs,other=build_inputs()
    with torch.no_grad():
        expected=reference(*inputs)
        actual=candidate(*other)
        half=forward(name,*inputs,half=True)
    dtype=inputs[0].dtype if 'shrink' in name or 'fused' in name else inputs[2].dtype
    honest_bounds=evaluate_outputs(name,actual,expected,inputs)
    half_bounds=evaluate_outputs(name,half,expected,inputs)
    report={"schema":"multi-lora-numerics-v1","task":name,
            "diagnostic_only":True,"fp16_intermediate_expected_failure":False,
            "honest_gate":honest_bounds,
            "fp16_intermediate_gate":half_bounds,
            "fp16_intermediate_would_fail_relative_bound":not half_bounds["passed"],
            "honest":error_distribution(actual,expected,dtype),
            "fp16_intermediate":error_distribution(half,expected,dtype)}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"artifact/lora_numerics":str(args.output)}))


if __name__=='__main__':
    main()
