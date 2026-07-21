"""Paged KV-cache: fixed-size blocks + per-sequence block tables.

Why paged
---------
The naive KV cache allocates one contiguous [max_seq_len] buffer per
sequence. Two kinds of waste follow: (1) internal fragmentation — a request
that stops at 40 tokens still reserved max_seq_len slots; (2) you must pick
max concurrent sequences up front, because growing a contiguous buffer means
reallocating and copying.

Paged allocation (vLLM's core idea) splits the cache into fixed-size blocks
of `block_size` tokens. A sequence owns a list of block ids (its block
table); the last block is partially filled. Waste is bounded by one block
per sequence, admission control becomes "are there enough free blocks", and
freeing a finished request returns its blocks to the pool immediately.

The price: K/V for a sequence are no longer contiguous, so attention needs
a kernel that walks the block table (kernels/paged_attention.py).

Layout: per layer, K and V pools of shape
    [num_blocks, n_heads, block_size, head_dim]
so one (block, head) slab is contiguous — the unit the decode kernel loads.
"""

from __future__ import annotations

import torch


class PagedKVCache:
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 num_blocks: int, block_size: int = 16,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.n_layers, self.n_heads, self.head_dim = n_layers, n_heads, head_dim
        self.num_blocks, self.block_size = num_blocks, block_size
        self.device, self.dtype = device, dtype
        shape = (num_blocks, n_heads, block_size, head_dim)
        self.k_pool = [torch.zeros(shape, device=device, dtype=dtype)
                       for _ in range(n_layers)]
        self.v_pool = [torch.zeros(shape, device=device, dtype=dtype)
                       for _ in range(n_layers)]
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self.tables: dict[int, list[int]] = {}     # seq_id -> block ids
        self.lens: dict[int, int] = {}             # seq_id -> tokens stored

    # ------------------------------------------------------------ accounting

    def blocks_needed(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, n_tokens: int) -> bool:
        return len(self.free_blocks) >= self.blocks_needed(n_tokens)

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, seq_id: int, n_tokens: int):
        assert seq_id not in self.tables
        need = self.blocks_needed(n_tokens)
        if need > len(self.free_blocks):
            raise RuntimeError("out of KV cache blocks")
        self.tables[seq_id] = [self.free_blocks.pop() for _ in range(need)]
        self.lens[seq_id] = 0

    def append_slot(self, seq_id: int) -> bool:
        """Reserve room for one more token; may claim a fresh block.
        Returns False (no state change) if the pool is exhausted."""
        used = self.lens[seq_id]
        if used + 1 > len(self.tables[seq_id]) * self.block_size:
            if not self.free_blocks:
                return False
            self.tables[seq_id].append(self.free_blocks.pop())
        return True

    def free(self, seq_id: int):
        self.free_blocks.extend(reversed(self.tables.pop(seq_id)))
        del self.lens[seq_id]

    # ------------------------------------------------------------- KV writes

    def write_prefill(self, layer: int, seq_id: int, k: torch.Tensor,
                      v: torch.Tensor):
        """k, v: [n_heads, prompt_len, head_dim] — scatter into blocks.

        Does NOT advance the sequence length: callers write all layers, then
        call set_len once. Keeping length advancement explicit (rather than a
        side effect of the last layer's write) is what guarantees every layer
        of one step sees the same context length.
        """
        S = k.shape[1]
        table = self.tables[seq_id]
        bs = self.block_size
        for i in range(0, S, bs):
            blk = table[i // bs]
            n = min(bs, S - i)
            self.k_pool[layer][blk, :, :n] = k[:, i:i + n]
            self.v_pool[layer][blk, :, :n] = v[:, i:i + n]

    def write_decode(self, layer: int, seq_id: int, pos: int,
                     k: torch.Tensor, v: torch.Tensor):
        """k, v: [n_heads, head_dim] for the token at absolute position pos."""
        blk = self.tables[seq_id][pos // self.block_size]
        off = pos % self.block_size
        self.k_pool[layer][blk, :, off] = k
        self.v_pool[layer][blk, :, off] = v

    def set_len(self, seq_id: int, n: int):
        self.lens[seq_id] = n

    # ------------------------------------------------- kernel-facing views

    def batch_tables(self, seq_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded [n_seqs, max_blocks] int32 block tables + context lens."""
        max_b = max(len(self.tables[s]) for s in seq_ids)
        tables = torch.zeros(len(seq_ids), max_b, dtype=torch.int32,
                             device=self.device)
        lens = torch.zeros(len(seq_ids), dtype=torch.int32, device=self.device)
        for i, s in enumerate(seq_ids):
            t = self.tables[s]
            tables[i, :len(t)] = torch.tensor(t, dtype=torch.int32)
            lens[i] = self.lens[s]
        return tables, lens

    def gather_contiguous(self, layer: int, seq_id: int, n: int | None = None):
        """Debug/reference helper: reassemble [n_heads, len, head_dim]."""
        n = self.lens[seq_id] if n is None else n
        bs = self.block_size
        ks, vs = [], []
        for i, blk in enumerate(self.tables[seq_id]):
            take = min(bs, n - i * bs)
            if take <= 0:
                break
            ks.append(self.k_pool[layer][blk, :, :take])
            vs.append(self.v_pool[layer][blk, :, :take])
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)
