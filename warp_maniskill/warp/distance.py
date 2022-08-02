import warp as wp

@wp.kernel
class chamfer_distance:
    input_types = {
        "xyz1": wp.array(dtype=wp.vec3),
        "n1": wp.int32,
        "xyz2": wp.array(dtype=wp.vec3),
        "n2": wp.int32,
        "dist": wp.array(dtype=float),
        "index": wp.array(dtype=wp.int64)
    }

    cuda_source = '''
typedef float scalar_t;
typedef int64_t index_t;
constexpr int BLOCK_SIZE = 256;
extern "C" {
__global__ void chamfer_distance_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec3> xyz1,
    int n1,
    wp::array_t<wp::vec3> xyz2,
    int n2,
    wp::array_t<float> dist,
    wp::array_t<wp::int64> index)
{
    int batch_size = 1;

    // calculate the number of blocks
    const int num_block1 = (n1 + blockDim.x - 1) / blockDim.x;
    const int num_block2 = (n2 + blockDim.x - 1) / blockDim.x;
    const int total_blocks = batch_size * num_block1;

    for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
    {
        __shared__ scalar_t xyz2_buffer[BLOCK_SIZE * 3];
        const int batch_idx = block_idx / num_block1;
        const int block_idx1 = block_idx % num_block1;
        const int xyz1_idx = (block_idx1 * blockDim.x) + threadIdx.x;
        const int xyz1_offset = (batch_idx * n1 + xyz1_idx) * 3;
        scalar_t x1, y1, z1;
        if (xyz1_idx < n1)
        {
            x1 = ((scalar_t*)xyz1.data)[xyz1_offset + 0];
            y1 = ((scalar_t*)xyz1.data)[xyz1_offset + 1];
            z1 = ((scalar_t*)xyz1.data)[xyz1_offset + 2];
        }
        else
        {
            x1 = y1 = z1 = 0.0;
        }
        scalar_t min_dist = 1e32;
        index_t min_idx = -1;
        // load a block of xyz2 data to reduce the times to read data
        for (int block_idx2 = 0; block_idx2 < num_block2; ++block_idx2)
        {
            // load xyz2 data
            int xyz2_idx = (block_idx2 * blockDim.x) + threadIdx.x;
            int xyz2_offset = (batch_idx * n2 + xyz2_idx) * 3;
            if (xyz2_idx < n2)
            {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    xyz2_buffer[threadIdx.x * 3 + i] = ((scalar_t*)xyz2.data)[xyz2_offset + i];
                }
            }
            __syncthreads();
            // calculate the distance between xyz1 and xyz2, with the shared memory.
            for (int j = 0; j < blockDim.x; ++j)
            {
                xyz2_idx = (block_idx2 * blockDim.x) + j;
                const int buffer_offset = j * 3;
                scalar_t x2 = xyz2_buffer[buffer_offset + 0];
                scalar_t y2 = xyz2_buffer[buffer_offset + 1];
                scalar_t z2 = xyz2_buffer[buffer_offset + 2];
                scalar_t d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
                if (xyz2_idx < n2 && d < min_dist)
                {
                    min_dist = d;
                    min_idx = xyz2_idx;
                }
            }
            __syncthreads();
        }
        if (xyz1_idx < n1)
        {
            const int output_offset = batch_idx * n1 + xyz1_idx;
            dist.data[output_offset] = min_dist;
            index.data[output_offset] = min_idx;
        }
    }
}

__global__ void chamfer_distance_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec3> xyz1,
    int n1,
    wp::array_t<wp::vec3> xyz2,
    int n2,
    wp::array_t<float> dist,
    wp::array_t<wp::int64> index,
    wp::array_t<wp::vec3> adj_xyz1,
    int &adj_n1,
    wp::array_t<wp::vec3> adj_xyz2,
    int &adj_n2,
    wp::array_t<float> adj_dist,
    wp::array_t<wp::int64> adj_index)
{
    const int batch_size = 1;

    const uint64_t totalElements = batch_size * n1;
    for (int linearId = blockIdx.x * blockDim.x + threadIdx.x;
         linearId < totalElements;
         linearId += gridDim.x * blockDim.x)
    {
        int batch_idx = linearId / n1;
        int xyz1_offset = linearId * 3;
        scalar_t x1 = ((scalar_t*)xyz1.data)[xyz1_offset + 0];
        scalar_t y1 = ((scalar_t*)xyz1.data)[xyz1_offset + 1];
        scalar_t z1 = ((scalar_t*)xyz1.data)[xyz1_offset + 2];
        int xyz2_offset = (batch_idx * n2 + index.data[linearId]) * 3;
        scalar_t x2 = ((scalar_t*)xyz2.data)[xyz2_offset + 0];
        scalar_t y2 = ((scalar_t*)xyz2.data)[xyz2_offset + 1];
        scalar_t z2 = ((scalar_t*)xyz2.data)[xyz2_offset + 2];
        scalar_t g = adj_dist.data[linearId] * 2;
        scalar_t gx = g * (x1 - x2);
        scalar_t gy = g * (y1 - y2);
        scalar_t gz = g * (z1 - z2);
        atomicAdd(((scalar_t*)adj_xyz1.data) + xyz1_offset + 0, gx);
        atomicAdd(((scalar_t*)adj_xyz1.data) + xyz1_offset + 1, gy);
        atomicAdd(((scalar_t*)adj_xyz1.data) + xyz1_offset + 2, gz);
        atomicAdd(((scalar_t*)adj_xyz2.data) + xyz2_offset + 0, -gx);
        atomicAdd(((scalar_t*)adj_xyz2.data) + xyz2_offset + 1, -gy);
        atomicAdd(((scalar_t*)adj_xyz2.data) + xyz2_offset + 2, -gz);
    }
}
}
    '''

    cpp_source = ""


def compute_chamfer_distance(xyz1, n1, xyz2, n2, dist1, dist2, index1, index2):
    assert xyz1.dtype == wp.vec3
    assert xyz2.dtype == wp.vec3
    assert dist1.dtype == wp.float32
    assert dist2.dtype == wp.float32
    assert index1.dtype == wp.int64
    assert index2.dtype == wp.int64
    assert len(xyz1.shape) == 1
    assert len(xyz2.shape) == 1
    assert len(dist1.shape) == 1
    assert len(dist2.shape) == 1
    assert len(index1.shape) == 1
    assert len(index2.shape) == 1
    assert xyz1.shape[0] >= n1
    assert xyz2.shape[0] >= n2
    assert dist1.shape[0] >= n1
    assert dist2.shape[0] >= n2
    assert index1.shape[0] >= n1
    assert index2.shape[0] >= n2
    assert xyz1.device == "cuda"
    assert xyz2.device == "cuda"
    assert dist1.device == "cuda"
    assert dist2.device == "cuda"
    assert index1.device == "cuda"
    assert index2.device == "cuda"

    wp.launch(chamfer_distance, dim=n1, inputs=[xyz1, n1, xyz2, n2, dist1, index1], device="cuda")
    wp.launch(chamfer_distance, dim=n2, inputs=[xyz2, n2, xyz1, n1, dist2, index2], device="cuda")
