def forward(name, *args):
    import cuda_extension
    op = 2 if name == "lora_segment_metadata" else 3 if "fused" in name else 0 if "shrink" in name else 1
    values = cuda_extension.lora_forward(list(args), op, name.endswith("sgmv"))
    return tuple(values) if op == 2 else values[0]
