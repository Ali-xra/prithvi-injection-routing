"""The shuffle permutation must be a derangement for every seed we will use."""
import torch

N = 16200
for seed in range(1000, 1010):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    idx = torch.arange(N)
    before = int((perm == idx).sum())
    for _ in range(64):
        fixed = (perm == idx).nonzero().flatten()
        if len(fixed) == 0:
            break
        if len(fixed) == 1:
            f = fixed[0].item(); j = (f + 1) % N
            perm[[f, j]] = perm[[j, f]]
        else:
            perm[fixed] = perm[torch.roll(fixed, 1)]
    after = int((perm == idx).sum())
    bij = len(torch.unique(perm)) == N
    print(f"seed {seed}: fixed before {before:2d} -> after {after}   bijection {bij}"
          + ("   OK" if after == 0 and bij else "   FAIL"))
