import os, torch, torch.nn as nn
from typing import Tuple, List, Dict, Optional

# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #
def _primes(dev):   # 3-tuple of large primes
    return torch.tensor([73856093, 19349669, 83492791],
                        device=dev, dtype=torch.long)

def _corner_offsets(dev):  # (8,3) corner offsets
    return torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                         [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
                        device=dev, dtype=torch.long)

# --------------------------------------------------------------------------- #
#  dense level (train)                                                        #
# --------------------------------------------------------------------------- #
class _TrainLevel(nn.Module):
    def __init__(self, res, d, buckets, smin, smax, primes, dev):
        super().__init__()
        self.res, self.d, self.buckets, self.primes = res, d, buckets, primes
        xs = torch.arange(smin[0], smax[0], res, device=dev)
        ys = torch.arange(smin[1], smax[1], res, device=dev)
        zs = torch.arange(smin[2], smax[2], res, device=dev)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        self.coords = torch.stack([gx, gy, gz], -1).view(-1, 3)  # (N,3)
        self.N = self.coords.size(0)

        self.voxel_features = nn.Parameter(
            torch.zeros(self.N, d, device=dev).normal_(0, .01))

        self.register_buffer('hash2vox',
            torch.full((buckets,), -1, dtype=torch.long, device=dev))
        self._fill()
        self.register_buffer('access',
            torch.zeros(self.N, dtype=torch.bool, device=dev),
            persistent=False)

    def _fill(self):
        idx = torch.floor(self.coords / self.res).long()
        hv  = (idx * self.primes).sum(-1) % self.buckets
        empty = self.hash2vox[hv] == -1
        self.hash2vox[hv[empty]] = torch.arange(self.N, device=self.voxel_features.device)[empty]
        dup = hv.unique(return_counts=True)[1] > 1
        self.register_buffer('col',
            torch.tensor(int(dup.sum()), device=self.voxel_features.device),
            persistent=False)

    # ---------- public utils
    @torch.no_grad()                                  # short stats
    def collision_stats(self): return dict(total=self.N, col=int(self.col))

    @torch.no_grad()
    def get_accessed_indices(self): return torch.nonzero(self.access).flatten()

    @torch.no_grad()                                  # clear log
    def reset_access_log(self): self.access.zero_()

    @torch.no_grad()                                  # sparse dump
    def export_sparse(self):
        used = self.get_accessed_indices()
        return dict(resolution=self.res,
                    coords=self.coords[used].cpu(),
                    features=self.voxel_features[used].cpu())

    # ---------- internals
    def _lookup(self, idxg):
        hv  = (idxg * self.primes).sum(-1) % self.buckets
        vid = self.hash2vox[hv]
        valid = vid >= 0
        out = torch.zeros(*idxg.shape[:-1], self.d,
                          device=self.voxel_features.device,
                          dtype=self.voxel_features.dtype)
        if valid.any():
            self.access[vid[valid]] = True
            out[valid] = self.voxel_features[vid[valid]]
        return out

    def query(self, pts):
        q, offs = pts / self.res, _corner_offsets(pts.device)
        base    = torch.floor(q).long()
        frac    = q - base.float()
        idx     = base[:,None,:] + offs[None,:,:]
        feat    = self._lookup(idx)

        wx = torch.stack([1-frac[:,0], frac[:,0]], 1)
        wy = torch.stack([1-frac[:,1], frac[:,1]], 1)
        wz = torch.stack([1-frac[:,2], frac[:,2]], 1)
        w  = (wx[:,[0,1,0,1,0,1,0,1]] *
              wy[:,[0,0,1,1,0,0,1,1]] *
              wz[:,[0,0,0,0,1,1,1,1]])
        return (feat * w.unsqueeze(-1)).sum(1)

# --------------------------------------------------------------------------- #
#  sparse level (infer)                                                       #
# --------------------------------------------------------------------------- #
class _InferLevel(nn.Module):
    def __init__(self, pay, d, buckets, primes, dev):
        super().__init__()
        self.res, self.d, self.buckets, self.primes = float(pay['resolution']), d, buckets, primes
        coords, feats = pay['coords'].to(dev), pay['features'].to(dev)
        self.register_buffer('coords', coords, persistent=False)
        self.voxel_features = nn.Parameter(feats, requires_grad=False)

        self.register_buffer('hash2vox',
            torch.full((buckets,), -1, dtype=torch.long, device=dev),
            persistent=False)
        idx = torch.floor(coords / self.res).long()
        hv  = (idx * self.primes).sum(-1) % buckets
        self.hash2vox[hv] = torch.arange(coords.size(0), device=dev)

    # dummy stats
    def collision_stats(self): return dict(total=self.coords.size(0), col=0)

    def get_accessed_indices(self): return torch.empty(0, dtype=torch.long, device=self.coords.device)

    def reset_access_log(self): pass

    def _lookup(self, idxg):
        hv  = (idxg * self.primes).sum(-1) % self.buckets
        vid = self.hash2vox[hv]
        valid = vid >= 0
        out = torch.zeros(*idxg.shape[:-1], self.d,
                          device=self.coords.device,
                          dtype=self.voxel_features.dtype)
        if valid.any(): out[valid] = self.voxel_features[vid[valid]]
        return out

    def query(self, pts):
        q, offs = pts / self.res, _corner_offsets(pts.device)
        base    = torch.floor(q).long()
        frac    = q - base.float()
        idx     = base[:,None,:] + offs[None,:,:]
        feat    = self._lookup(idx)

        wx = torch.stack([1-frac[:,0], frac[:,0]], 1)
        wy = torch.stack([1-frac[:,1], frac[:,1]], 1)
        wz = torch.stack([1-frac[:,2], frac[:,2]], 1)
        w  = (wx[:,[0,1,0,1,0,1,0,1]] *
              wy[:,[0,0,1,1,0,0,1,1]] *
              wz[:,[0,0,0,0,1,1,1,1]])
        return (feat * w.unsqueeze(-1)).sum(1)

# --------------------------------------------------------------------------- #
#  public pyramid                                                             #
# --------------------------------------------------------------------------- #
class VoxelHashTable(nn.Module):
    """
    mode='train' → dense levels, mode='infer' → sparse levels
    """
    def __init__(
        self,
        resolution: float = 0.12,
        num_levels: int = 2,
        level_scale: float = 2.0,
        feature_dim: int = 32,
        hash_table_size: int = 2**21,
        scene_bound_min: Tuple[float,float,float]=(-2.6,-8.1,0),
        scene_bound_max: Tuple[float,float,float]=( 4.6, 4.7,3.1),
        device: str = "cuda:0",
        mode: str = "train",
        sparse_data: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode, self.d = mode, feature_dim
        dev = torch.device(device)
        primes = _primes(dev)
        self.levels = nn.ModuleList()

        if mode == "train":
            # Iterate coarse → fine by reversing the exponent.
            for lv in range(num_levels):
                res = resolution * (level_scale ** (num_levels - 1 - lv))
                self.levels.append(
                    _TrainLevel(res, feature_dim, hash_table_size,
                                scene_bound_min, scene_bound_max,
                                primes, dev))
        elif mode == "infer":
            if sparse_data is None:
                raise ValueError("sparse_data is required in infer mode")
            # Sort payloads from coarse (larger res) → fine (smaller res)
            sorted_levels = sorted(
                sparse_data['levels'],
                key=lambda p: p['resolution'],
                reverse=True
            )
            for pay in sorted_levels:
                self.levels.append(
                    _InferLevel(pay, feature_dim, hash_table_size,
                                primes, dev))
        else:
            raise ValueError("mode must be 'train' or 'infer'")

    # forward -----------------------------------------------------------------
    def query_voxel_feature(self, pts):  # (M,3) → (M, d*L)
        per = [lv.query(pts) for lv in self.levels]
        return torch.cat(per, -1)

    # utils -------------------------------------------------------------------
    @torch.no_grad()
    def collision_stats(self):
        return {f"level_{i}": lv.collision_stats() for i,lv in enumerate(self.levels)}

    @torch.no_grad()
    def get_accessed_indices(self):
        return [lv.get_accessed_indices() for lv in self.levels]

    @torch.no_grad()
    def reset_access_log(self):
        for lv in self.levels: lv.reset_access_log()

    # save / load -------------------------------------------------------------
    @torch.no_grad()
    def export_sparse(self):
        if self.mode != "train":
            raise RuntimeError("export_sparse only in train mode")
        return dict(num_levels=len(self.levels),
                    feature_dim=self.d,
                    levels=[lv.export_sparse() for lv in self.levels])

    # dense weight file
    def save_dense(self, path):
        torch.save({'state_dict': self.state_dict()}, path)

    # sparse file 
    def save_sparse(self, path):
        torch.save(self.export_sparse(), path)

    @staticmethod
    def load_dense(path, device="cuda:0"):
        chk = torch.load(path, map_location="cpu")
        vt  = VoxelHashTable(device=device)    # default ctor, train mode
        vt.load_state_dict(chk['state_dict'])
        return vt.to(device)

    @staticmethod
    def load_sparse(path, device="cuda:0"):
        sparse = torch.load(path, map_location="cpu")
        return VoxelHashTable(mode="infer", sparse_data=sparse, device=device)