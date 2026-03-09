import torch


def autograd_gradients(
    energy: torch.Tensor, positions: torch.Tensor, retain_graph: bool = True, require_next_order: bool = False
) -> torch.Tensor:
    # TODO: implement vmap to make this faster in batch mode.
    grad_outputs: list[torch.Tensor] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,  # Make sure the graph is not destroyed during training
        create_graph=require_next_order,  # Create graph for next order derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[0]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    # Note gradient is equivalent to (- Forces) in the context of molecular dynamics.
    return gradient
