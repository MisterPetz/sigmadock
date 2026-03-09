import traceback

import torch

_orig_sqrt = torch.sqrt
_orig_sqrt_m = torch.Tensor.sqrt


def debug_sqrt(x, *args, **kwargs):  # noqa
    if (x != x).any() or torch.isinf(x).any():
        stack = "".join(traceback.format_stack())
        raise RuntimeError(f"[sqrt] bad input {tuple(x.shape)}\n{stack} and val {x}")
    out = _orig_sqrt(x, *args, **kwargs)
    if (out != out).any() or torch.isinf(out).any():
        stack = "".join(traceback.format_stack())
        raise RuntimeError(f"[sqrt] bad output {tuple(out.shape)}\n{stack} and val {out}")
    return out


def debug_sqrt_m(self, *args, **kwargs):  # noqa
    return debug_sqrt(self, *args, **kwargs)


torch.sqrt = debug_sqrt
torch.Tensor.sqrt = debug_sqrt_m
