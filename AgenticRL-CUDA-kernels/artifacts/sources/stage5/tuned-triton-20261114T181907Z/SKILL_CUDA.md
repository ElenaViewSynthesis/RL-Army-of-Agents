# CUDA provenance naming supplement

The protected upstream policy remains in SKILL.md. This supplement describes
trusted harness attribution; it does not change its library-use permissions.

Authored names must have at least four characters and must not be any of:
`kernel`, `main`, `add`, `sub`, `mul`, `div`, `sum`, `copy`, `fill`, `forward`,
`fused`, `elementwise`, `vectorized`, `triton`, `aten`, `native`.
Rejected names are excluded from source discovery and cannot earn agent credit.

The CUDA symbol pattern is `(^|[\s:]){name}\s*[<(]`, used with `re.search`
and `re.escape(authored_name)` substituted for `{name}`. Evidence: H200 run
`20261106T185236Z`, `void axpby_kernel<256>(float*, float const*, float const*, float, int)`.

Memory events are handled first. Torch namespace markers (`at::native`,
`aten::`, `c10::`) and Inductor names matching `^triton_(poi|per|red|tem)_fused`
count as Torch before authored matching. Library markers follow authored matches;
CUDA retains its existing allowed-library attribution. Unknown events count as
Torch. Name matching remains a heuristic, not proof of authorship.
