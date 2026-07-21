"""A tiny auto-fusion compiler: computation graph -> generated Triton kernel.

A miniature of what torch.compile/Inductor does for pointwise+reduction
chains: instead of hand-writing every fused kernel, build a small DAG of
ops and codegen ONE Triton kernel that keeps every intermediate in
registers — no materialized temporaries between ops.

Supported ops (9): matmul (≤1, operands must be graph inputs), add, sub,
mul, div, exp, relu, rowmax, rowsum — enough to express the attention
score chain softmax(Q·Kᵀ·scale) that Phase 1 hand-fused.

Deliberate scope constraints (documented, enforced):
  * 2D tensors, one output, fp32;
  * row-tiled execution: each program owns BLOCK_M rows and the FULL
    feature dimension (BLOCK_N = next_pow2(N)) — that's what makes row
    reductions free; long rows would need the online-softmax trick the
    hand-written kernel uses, which is exactly the kind of scheduling a
    real compiler adds and this one doesn't;
  * reductions sanitize padded lanes themselves (masked -inf/0), so any
    elementwise garbage in padding never reaches a result.

Codegen is plain string emission + exec; `FusedKernel.source` exposes the
generated Triton code for inspection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl

_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_UNARY = {"exp", "relu", "neg"}
_REDUCE = {"rowmax", "rowsum"}


@dataclass(eq=False)          # identity hash: nodes are graph vertices
class Node:
    idx: int
    op: str                      # input | const | matmul | <binary> | ...
    args: tuple = ()             # upstream Node refs
    kind: str = "mat"            # mat [M,N] | col [M] | scalar
    role: str = ""               # inputs: mn | mk | nk
    value: float = 0.0           # consts

    @property
    def var(self) -> str:
        return f"v{self.idx}"


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.inputs: list[Node] = []
        self.out: Node | None = None
        self.has_matmul = False

    def _add(self, op, args=(), **kw) -> Node:
        n = Node(len(self.nodes), op, tuple(args), **kw)
        self.nodes.append(n)
        return n

    def input(self, role: str = "mn") -> Node:
        assert role in ("mn", "mk", "nk")
        n = self._add("input", kind="mat", role=role)
        self.inputs.append(n)
        return n

    def const(self, value: float) -> Node:
        return self._add("const", kind="scalar", value=float(value))

    def matmul(self, a: Node, b: Node) -> Node:
        """[M,K] @ [N,K]^T -> [M,N]. Operands must be 'mk'/'nk' inputs."""
        assert not self.has_matmul, "at most one matmul per graph"
        assert a.op == "input" and a.role == "mk", "matmul lhs must be an mk input"
        assert b.op == "input" and b.role == "nk", "matmul rhs must be an nk input"
        self.has_matmul = True
        return self._add("matmul", (a, b), kind="mat")

    def _binary(self, op, a: Node, b: Node) -> Node:
        kind = "mat" if "mat" in (a.kind, b.kind) else \
               ("col" if "col" in (a.kind, b.kind) else "scalar")
        return self._add(op, (a, b), kind=kind)

    def add(self, a, b): return self._binary("add", a, b)
    def sub(self, a, b): return self._binary("sub", a, b)
    def mul(self, a, b): return self._binary("mul", a, b)
    def div(self, a, b): return self._binary("div", a, b)

    def exp(self, a): return self._add("exp", (a,), kind=a.kind)
    def neg(self, a): return self._add("neg", (a,), kind=a.kind)
    def relu(self, a): return self._add("relu", (a,), kind=a.kind)

    def rowmax(self, a):
        assert a.kind == "mat"
        return self._add("rowmax", (a,), kind="col")

    def rowsum(self, a):
        assert a.kind == "mat"
        return self._add("rowsum", (a,), kind="col")

    def output(self, n: Node):
        assert n.kind == "mat", "output must be [M, N]"
        self.out = n


def _ref(node: Node, want: str) -> str:
    """Emit a reference to node's value broadcast to `want` kind."""
    if node.op == "const":
        return repr(node.value)
    if node.kind == "col" and want == "mat":
        return f"{node.var}[:, None]"
    return node.var


def _emit(g: Graph) -> str:
    """Generate Triton kernel source for the graph."""
    assert g.out is not None, "graph has no output"
    L: list[str] = []
    mn_inputs = [n for n in g.inputs if n.role == "mn"]
    mk = [n for n in g.inputs if n.role == "mk"]
    nk = [n for n in g.inputs if n.role == "nk"]

    ptr_params = [f"P{n.idx}" for n in g.inputs] + ["POUT"]
    stride_params = []
    for n in g.inputs:
        stride_params += [f"s{n.idx}_0", f"s{n.idx}_1"]
    stride_params += ["so_0", "so_1"]
    dims = ["M", "N"] + (["K"] if g.has_matmul else [])
    blocks = ["BLOCK_M: tl.constexpr", "BLOCK_N: tl.constexpr"] + \
             (["BLOCK_K: tl.constexpr"] if g.has_matmul else [])

    L.append("@triton.jit")
    L.append(f"def _fused_kernel({', '.join(ptr_params + dims + stride_params + blocks)}):")
    L.append("    pid = tl.program_id(0)")
    L.append("    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)")
    L.append("    offs_n = tl.arange(0, BLOCK_N)")
    L.append("    mask_m = offs_m < M")
    L.append("    mask_n = offs_n < N")
    L.append("    mmask = mask_m[:, None] & mask_n[None, :]")

    for n in mn_inputs:
        L.append(f"    {n.var} = tl.load(P{n.idx} + offs_m[:, None] * s{n.idx}_0"
                 f" + offs_n[None, :] * s{n.idx}_1, mask=mmask, other=0.0)")

    for n in g.nodes:
        if n.op in ("input", "const"):
            continue
        if n.op == "matmul":
            a, b = n.args
            L.append(f"    {n.var} = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)")
            L.append("    for k0 in range(0, K, BLOCK_K):")
            L.append("        offs_k = k0 + tl.arange(0, BLOCK_K)")
            L.append("        mask_k = offs_k < K")
            L.append(f"        _a = tl.load(P{a.idx} + offs_m[:, None] * s{a.idx}_0"
                     f" + offs_k[None, :] * s{a.idx}_1,"
                     " mask=mask_m[:, None] & mask_k[None, :], other=0.0)")
            L.append(f"        _b = tl.load(P{b.idx} + offs_n[:, None] * s{b.idx}_0"
                     f" + offs_k[None, :] * s{b.idx}_1,"
                     " mask=mask_n[:, None] & mask_k[None, :], other=0.0)")
            L.append(f"        {n.var} = tl.dot(_a, tl.trans(_b), {n.var})")
        elif n.op in _BINARY:
            a, b = n.args
            L.append(f"    {n.var} = {_ref(a, n.kind)} {_BINARY[n.op]} "
                     f"{_ref(b, n.kind)}")
        elif n.op == "exp":
            L.append(f"    {n.var} = tl.exp({n.args[0].var})")
        elif n.op == "neg":
            L.append(f"    {n.var} = -{n.args[0].var}")
        elif n.op == "relu":
            L.append(f"    {n.var} = tl.maximum({n.args[0].var}, 0.0)")
        elif n.op == "rowmax":
            L.append(f"    {n.var} = tl.max(tl.where(mask_n[None, :], "
                     f"{n.args[0].var}, float('-inf')), axis=1)")
        elif n.op == "rowsum":
            L.append(f"    {n.var} = tl.sum(tl.where(mask_n[None, :], "
                     f"{n.args[0].var}, 0.0), axis=1)")
        else:
            raise ValueError(f"unknown op {n.op}")

    L.append(f"    tl.store(POUT + offs_m[:, None] * so_0"
             f" + offs_n[None, :] * so_1, {g.out.var}, mask=mmask)")
    return "\n".join(L)


class FusedKernel:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.source = _emit(graph)
        ns = {"triton": triton, "tl": tl}
        exec(compile(self.source, "<fused_kernel>", "exec"), ns)
        self._kernel = ns["_fused_kernel"]

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        g = self.graph
        assert len(tensors) == len(g.inputs)
        by_node = dict(zip(g.inputs, tensors))
        M = N = K = None
        for n, t in by_node.items():
            assert t.dim() == 2 and t.dtype == torch.float32
            if n.role == "mn":
                M, N = t.shape
            elif n.role == "mk":
                M, K = t.shape
            else:
                N = t.shape[0]
                K = t.shape[1]
        out = torch.empty(M, N, dtype=torch.float32,
                          device=tensors[0].device)

        interp = os.environ.get("TRITON_INTERPRET") == "1"
        BLOCK_M = 16
        BLOCK_N = max(16, triton.next_power_of_2(N))
        args = [t.contiguous() for t in tensors] + [out, M, N]
        if g.has_matmul:
            args.append(K)
        for t in tensors:
            t = t.contiguous()
            args += [t.stride(0), t.stride(1)]
        args += [out.stride(0), out.stride(1)]
        kw = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        if g.has_matmul:
            kw["BLOCK_K"] = 16 if interp else 32
        self._kernel[(triton.cdiv(M, BLOCK_M),)](*args, **kw)
        return out


def softmax_graph(with_matmul: bool = False, scale: float = 1.0) -> Graph:
    """The attention score chain: softmax(Q·Kᵀ·scale) (or softmax of a
    given matrix) — the fusion Phase 1 wrote by hand."""
    g = Graph()
    if with_matmul:
        q = g.input("mk")
        k = g.input("nk")
        s = g.mul(g.matmul(q, k), g.const(scale))
    else:
        s = g.mul(g.input("mn"), g.const(scale))
    m = g.rowmax(s)
    e = g.exp(g.sub(s, m))
    z = g.rowsum(e)
    g.output(g.div(e, z))
    return g
