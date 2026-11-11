"""Archived pre-repair helper; regression fixture, not production code.
Source: H200 stage6-smoke-20261110T105217Z frozen utils/benchmark.py.
Full source SHA256: 4d40754fe64826e082babe5b78e21aafb1fd7d06abeecaa593e070d3b19104ee
"""

def _capture_graph(model, inputs, fresh, torch, torch_model, check_equal):
    """Side-stream warmup then capture; failure is data, never a crash."""
    entry = {"graph_capturable": False, "graph_error": None,
             "graph_replay_ok": None, "graph_us": []}
    try:
        static_inputs = [t.clone() for t in inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model(*static_inputs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = model(*static_inputs)
        entry["graph_capturable"] = True
        try:
            for dst, src in zip(static_inputs, fresh):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            check_equal(static_out, torch_model(*fresh))
            entry["graph_replay_ok"] = True
        except Exception:
            entry["graph_replay_ok"] = False
        return entry, graph
    except Exception as exc:
        entry["graph_error"] = f"{type(exc).__name__}: {exc}"[:512]
        return entry, None
