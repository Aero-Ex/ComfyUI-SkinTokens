import torch

def fps(
    x: torch.Tensor,
    batch: torch.Tensor,
    ratio: float,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: (N, C) points.
        batch: (N,) batch indices for each point.
        ratio: sampling ratio in (0, 1].
        random_start: whether to start from a random point per batch.
    
    Returns:
        1D tensor of sampled indices in the flattened input space.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (N, C), got {tuple(x.shape)}")
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError("batch must be 1D and aligned with x")
    if not (0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    sampled_indices = []
    unique_batches = torch.unique(batch)

    for batch_id in unique_batches:
        mask = batch == batch_id
        points = x[mask]
        num_points = points.shape[0]

        if num_points == 0:
            continue

        num_samples = max(1, int(round(num_points * ratio)))
        num_samples = min(num_samples, num_points)

        if random_start:
            farthest = torch.randint(num_points, (1,), device=x.device).item()
        else:
            farthest = 0

        distances = torch.full((num_points,), float("inf"), device=x.device)
        selected_local = torch.empty(num_samples, dtype=torch.long, device=x.device)

        for i in range(num_samples):
            selected_local[i] = farthest
            centroid = points[farthest]
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()

        global_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)[selected_local]
        sampled_indices.append(global_indices)

    if not sampled_indices:
        return torch.empty((0,), dtype=torch.long, device=x.device)
    return torch.cat(sampled_indices, dim=0)


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    Args:
        src: source tensor with shape (N, ...).
        indptr: CSR index pointer with shape (S + 1,).
        reduce: one of {"sum", "mean", "min", "max"}.
    
    Returns:
        Reduced tensor with shape (S, ...).
    """
    if src.ndim < 1:
        raise ValueError(f"Expected src to have at least 1 dim, got {src.ndim}")
    if indptr.ndim != 1:
        raise ValueError(f"Expected indptr to be 1D, got shape {tuple(indptr.shape)}")
    if indptr.numel() < 1:
        raise ValueError("indptr must contain at least one element")
    if reduce not in {"sum", "mean", "min", "max"}:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    indptr = indptr.to(device=src.device, dtype=torch.long)
    segments = indptr.numel() - 1
    out_shape = (segments, *src.shape[1:])

    if reduce in {"sum", "mean"}:
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    elif reduce == "min":
        out = torch.full(out_shape, float("inf"), dtype=src.dtype, device=src.device)
    else:
        out = torch.full(out_shape, float("-inf"), dtype=src.dtype, device=src.device)

    for i in range(segments):
        start = indptr[i].item()
        end = indptr[i + 1].item()
        if end <= start:
            continue

        chunk = src[start:end]
        if reduce == "sum":
            out[i] = chunk.sum(dim=0)
        elif reduce == "mean":
            out[i] = chunk.mean(dim=0)
        elif reduce == "min":
            out[i] = chunk.min(dim=0).values
        else:
            out[i] = chunk.max(dim=0).values

    if reduce == "min":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    elif reduce == "max":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)

    return out