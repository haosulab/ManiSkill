# Modified from https://github.com/stepjam/ARM/blob/main/arm/c2farm/voxel_grid.py
from functools import reduce as funtool_reduce
from operator import mul

import torch
from torch import nn, einsum
import torch.nn.functional as F

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,
                 max_num_coords: int,):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              device=device).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          device=device).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], device=device), max_dims,
             torch.tensor([4 + feature_size], device=device)], -1).tolist() 
        self._ones_max_coords = torch.ones((batch_size, max_num_coords, 1),
                                           device=device)
        self._num_coords = max_num_coords

        shape = self._total_dims_list

        self._result_dim_sizes = torch.tensor(
            [funtool_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [
                1], device=device)
        flat_result_size = funtool_reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float,
                                         device=device)
        self._flat_output = torch.ones(flat_result_size, dtype=torch.float,
                                       device=device) * self._initial_val
        self._arange_to_max_coords = torch.arange(4 + feature_size,
                                                  device=device)
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float,
                                       device=device)

        self._const_1 = torch.tensor(1.0, device=device)
        self._batch_size = batch_size

        # Coordinate Bounds:
        self._bb_mins = self._coord_bounds[..., 0:3]
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - self._bb_mins

        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int() # the number of voxels for each dimension after padding
        self._dims_orig = dims_orig = self._voxel_shape_spec.int() - 2 # voxel number for each dimension before padding
        self._dims_m_one = (dims - 1).int() # upper voxel index bounds
        
        # BS x 1 x 3
        self._res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)

        self._voxel_indicy_denmominator = self._res + MIN_DENOMINATOR
        self._dims_m_one_zeros = torch.zeros_like(self._dims_m_one) # lower voxel index bounds (0)

        batch_indices = torch.arange(self._batch_size, dtype=torch.int,
                                     device=device).view(self._batch_size, 1, 1)
        self._tiled_batch_indices = batch_indices.repeat(
            [1, self._num_coords, 1])

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, device=device)
        self._index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        # scatter pcd features to voxel features by taking the mean
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        # scatter the array of voxel indices and values to the voxel grid
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def _clamp_voxel_seg_grid(self, voxel_indices: torch.Tensor, pcd_seg: torch.Tensor, voxel_size: int):
        # clamp point cloud segmentations to voxel segmentations
        B, _, _ = voxel_indices.shape
        dims = torch.tensor([voxel_size] * 3, device=voxel_indices.device).int()        
        flat_voxel_indices = voxel_indices[..., 0] * (dims[1] * dims[2]) + voxel_indices[..., 1] * dims[2] + voxel_indices[..., 2]
        flat_voxel_indices = flat_voxel_indices.view(B, -1)
        
        # Create a tensor to record voxel classes
        num_classes = pcd_seg.max().item() + 1
        class_counts = torch.zeros(B, voxel_size * voxel_size * voxel_size, num_classes, device=voxel_indices.device, dtype=torch.float)

        # Create one-hot encodings of class indices
        one_hot_class_indices = F.one_hot(pcd_seg.long(), num_classes=num_classes).float()
        one_hot_class_indices = one_hot_class_indices.squeeze(2)

        # Add one-hot encodings to the class counts based on voxel indices
        class_counts.scatter_add_(1, flat_voxel_indices.unsqueeze(-1).expand(-1, -1, num_classes).to(torch.int64), 
                                  one_hot_class_indices)
        
        # Find the class with the maximum count in each voxel
        class_counts = class_counts.view(B, voxel_size, voxel_size, voxel_size, num_classes)
        voxel_class_indices = class_counts.argmax(dim=-1).unsqueeze(-1)
        return voxel_class_indices # [B, voxel_size, voxel_size, voxel_size, 1]
    
    def coords_to_bounding_voxel_grid(self, coords: torch.Tensor, coord_features=None,
                                      coord_bounds=None, clamp_vox_id=False, pcd_seg=None):
        # converting a batch of pcds with its features to a batch of voxels
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR) # reduce res so that farthest-side edge points can be clipped
            voxel_indicy_denmominator = res + MIN_DENOMINATOR
        bb_mins_shifted = bb_mins - res  
        floor = torch.floor( # use back-shifted bb_min so that 0-side edge points can be clipped
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        
        # get voxel indices before clipping (ranging [0, self._voxel_size+1])
        voxel_indices = torch.min(floor, self._dims_m_one) # clip into the maximal voxel index bounds
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros) # clip into the minimal voxel index bounds ((0,0,0))

        # global-coordinate point cloud (x, y, z)
        voxel_values = coords

        # rgb values (R, G, B)
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1) # concat rgb values (B, 128, 128, 3)

        # coordinates to aggregate over
        _, num_coords, _ = voxel_indices.shape
        all_indices = torch.cat([ # tile voxel index with batch index, so that different scenes in different batches don't mix
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1) 

        # count point cloud occupancy
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # scatter to get the voxel grid 
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size)) # 3+4=7
        vox = scattered[:, 1:-1, 1:-1, 1:-1] # clip the edges
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1] # compute each voxel's 3d position
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        # occupancy grid
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)

        # hard normalized voxel-location position encoding
        vox = torch.cat( 
           [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
            vox[..., -1:]], -1) 
        
        # add clamped voxel id if applicable
        if clamp_vox_id:
            assert pcd_seg != None, "Please provide the point cloud segmentations for clamping to voxel segmentations"
            vox_seg_grid = self._clamp_voxel_seg_grid(voxel_indices, pcd_seg, self._voxel_size+2)
            vox_seg_grid = vox_seg_grid[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], vox_seg_grid,
                vox[..., -1:]], -1) 
            
        # voxel 11D features contain: 3 (pcd xyz coordinates) + 3 (rgb) + 3 (voxel xyz indices) + 1 (seg id if applicable) + 1 (occupancy)
        return vox

    

