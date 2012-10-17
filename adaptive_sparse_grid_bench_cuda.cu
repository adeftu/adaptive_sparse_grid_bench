#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })



#define WARP_SIZE 32

#define MAX_NUM_DIMS    20
#define MAX_NUM_LEVELS  10

#define HI_FLOPS(num_dims, num_grid_points) \
	(((num_dims) * 3.0 * (double) (num_grid_points)) / 1000000000.0)

#define EV_FLOPS(num_dims, num_subspaces, num_evals) \
	(((num_evals) * (double) (num_subspaces) * (11 * (num_dims) + 2)) / 1000000000.0)


/* restrictions, the i-th value in a level vector must be <= h_restr_vect[i] */
static          int h_restr_vect[MAX_NUM_DIMS];

/* h_dp_mat[i][j] contains the number of vectors with i components 
    whose norm is j and the constraints from restr_vec are applied 
    i = 0..num_dims-1, j = 0..num_levels-1
*/
__constant__    int d_dp_mat[MAX_NUM_DIMS * MAX_NUM_LEVELS];
static          int h_dp_mat[MAX_NUM_DIMS * MAX_NUM_LEVELS];

#define D_DP_MAT(i,j)     d_dp_mat[(i) * num_levels + (j)]
#define H_DP_MAT(i,j)     h_dp_mat[(i) * num_levels + (j)]

/* h_lidx_vect[i] contains the number of points on level < i */
__constant__    int d_lidx_vect[MAX_NUM_LEVELS];
static          int h_lidx_vect[MAX_NUM_LEVELS];

/* precomputed parent indices */
__constant__    int d_par_lili[4 * ((1 << MAX_NUM_LEVELS) - 1)];
static          int h_par_lili[4 * ((1 << MAX_NUM_LEVELS) - 1)];

#define H_PAR_LILI(i,j)   h_par_lili[4 * ((1 << (i)) - 1) + (j)]
#define D_PAR_LILI(i,j)   d_par_lili[4 * ((1 << (i)) - 1) + (j)]

//=============================================================
//    helper functions
//=============================================================

float fct(int num_dims, float *coords)
{
    int i;
    float res = 0;
    for (i = 0; i < num_dims; i++)
        res += coords[i] * coords[i] * (1.0f - coords[i]) * (1.0f - coords[i]);
    return res;
}


__host__ __device__ void idx2gp(int num_dims, int num_levels,
                                int index, int *levels, int *indices);
__host__ __device__ int get_left_li(int l, int i, int *ll, int *li);
__host__ __device__ int get_right_li(int l, int i, int *rl, int *ri);


/* TODO: initialize restrictions vector */
void init_restr_vec(int num_dims, int num_levels)
{
    for (int i = 0; i < num_dims; ++i)
        h_restr_vect[i] = num_levels - 1;
}


void init_dp_mat(int num_dims, int num_levels)
{
    for (int j = 0; j <= MIN(num_levels - 1, h_restr_vect[0]); ++j) /* first line */
        H_DP_MAT(0,j) = 1;
    for (int i = 0; i < num_dims; ++i)                              /* first column */
        H_DP_MAT(i,0) = 1;   
    for (int i = 1; i < num_dims; ++i)
        for (int j = 1; j < num_levels; ++j)
            for (int k = 0; k <= MIN(j, h_restr_vect[i]); ++k)
                H_DP_MAT(i,j) += H_DP_MAT((i - 1),(j - k));
}


void init_lidx_vect(int num_dims, int num_levels, int *num_grid_points, int *num_subspaces)
{
    int num_grid_points_before = 0;
    *num_subspaces = 0;
    for (int i = 0; i < num_levels; ++i) {
        h_lidx_vect[i] = num_grid_points_before;
        num_grid_points_before += (H_DP_MAT((num_dims - 1),i) << i);
        *num_subspaces += H_DP_MAT((num_dims - 1),i);
    }
    *num_grid_points = num_grid_points_before;
}


float *init_sg1d(int num_dims, int num_levels, int num_grid_points)
{
    int levels[num_dims], indices[num_dims];
    float coords[num_dims];
    float *sg1d = (float *) calloc(num_grid_points, sizeof(float));
    for (int i = 0; i < num_grid_points; ++i) {
        idx2gp(num_dims, num_levels, i, levels, indices);
        for (int j = 0; j < num_dims; ++j)
            coords[j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
        sg1d[i] = fct(num_dims, coords);
    }
    return sg1d;
}


void init_par_lili(int num_levels)
{
    for (int i = 0; i < num_levels; ++i) {
        if (0 == i) {
            H_PAR_LILI(i,0) = -1;
            H_PAR_LILI(i,1) =  0;
            H_PAR_LILI(i,2) = -1;
            H_PAR_LILI(i,3) =  1;
            continue;
        }
        for (int j = 0; j < (1 << i); ++j) {
            if (0 == j) {
                H_PAR_LILI(i,0) = -1;
                H_PAR_LILI(i,1) =  0;
                get_right_li(i, j, &H_PAR_LILI(i,2), &H_PAR_LILI(i,3));
                continue;
            }
    
            if (j == (1 << i) - 1) {
                get_left_li(i, j, &H_PAR_LILI(i,j << 2), &H_PAR_LILI(i,(j << 2) + 1));
                H_PAR_LILI(i,(j << 2) + 2) = -1;
                H_PAR_LILI(i,(j << 2) + 3) =  1;  
                continue;
            }
            
            get_left_li(i, j, &H_PAR_LILI(i,j << 2), &H_PAR_LILI(i,(j << 2) + 1));
            get_right_li(i, j, &H_PAR_LILI(i,(j << 2) + 2), &H_PAR_LILI(i,(j << 2) + 3));                 
        }
    }
}


float *init_coord_mat(int num_dims, int num_levels, int num_grid_points, int num_evals)
{
    int levels[num_dims], indices[num_dims];
    const int size = num_evals * num_dims * sizeof(float);
    float *coord_mat = (float *) malloc(size);
    srandom(time(NULL));
    for (int i = 0; i < num_evals; ++i) {
        int rand_idx = random() % num_grid_points;
        idx2gp(num_dims, num_levels, rand_idx, levels, indices);
        for (int j = 0; j < num_dims; ++j)
            coord_mat[i * num_dims + j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
    }
    return coord_mat;
}

float *init_coord_mat_trans(int num_dims, int num_evals, float *coord_mat)
{
    // round up num_evals to the next multiple of WARP_SIZE and allocate that many elements
    const int size_trans = (((num_evals + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE) * num_dims * sizeof(float);
    float *coord_mat_trans = (float *) malloc(size_trans);
    for (int i = 0; i < num_evals; ++i) {
        const int line = i / WARP_SIZE;
        const int column = i % WARP_SIZE;
        const int start = line * num_dims * WARP_SIZE + column;
        for (int j = 0; j < num_dims; ++j)
            coord_mat_trans[start + j * WARP_SIZE] = coord_mat[i * num_dims + j];
    }
    return coord_mat_trans;
}



//=============================================================
//    sparse grid operations
//=============================================================

/* conversion function: grid point to index */
__device__ int gp2idx(int num_dims, int num_levels, int crt_dim, int plevel, int pindex,
                      int *levels, int *indices)
{
#define CRT_LEVEL(i) (((i) == crt_dim) ? plevel : levels[(i)])
#define CRT_INDEX(i) (((i) == crt_dim) ? pindex : indices[(i)])

    int index1, index2, index3, i, j, sum;

    index1 = CRT_INDEX(0);
    for (i = 1; i < num_dims; ++i) {
        index1 = (index1 << CRT_LEVEL(i)) + CRT_INDEX(i);
    }
        
    index2 = 0;
    sum = (0 == crt_dim) ? plevel : levels[0];
    for (i = 1; i < num_dims; ++i) {
        sum += CRT_LEVEL(i);
        for (j = 0; j < CRT_LEVEL(i); ++j)
            index2 += D_DP_MAT((i - 1),(sum - j));
    }   
    index2 <<= sum;
    
    index3 = d_lidx_vect[sum];
    
    return index1 + index2 + index3;
}


/* conversion function: index to grid point */
__host__ __device__ void idx2gp(int num_dims, int num_levels,
                                int index, int *levels, int *indices)
{
    #ifndef __CUDA_ARCH__
        #define DP_MAT(i,j)     H_DP_MAT(i,j)
        #define LIDX_VECT(i)    h_lidx_vect[(i)]
    #else
        #define DP_MAT(i,j)     D_DP_MAT(i,j)
        #define LIDX_VECT(i)    d_lidx_vect[(i)]
    #endif

    int i, index1, index3, position;
    int crt_level = 0;

    /* determine the group level */
    while (LIDX_VECT(crt_level) <= index && crt_level < num_levels)
        ++crt_level;

    index3 = LIDX_VECT(--crt_level);            /* the original index3 */
    index -= index3;

    position = index >> crt_level;              /* position of the vector inside its group */
    index1 = index & ((1 << crt_level) - 1);    /* the original index1 */

    index = position;

    for (i = num_dims - 2; i >= 0; --i) {
        int isum = 0;                           /* maximum number of vectors having i components */
        int j = crt_level;
        while (index >= isum + DP_MAT(i,j)) {
            isum += DP_MAT(i,j);
            --j;
        }
        levels[i + 1] = crt_level - j;
        indices[i + 1] = index1 & ((1 << levels[i + 1]) - 1);
        index1 >>= levels[i + 1];
        crt_level = j;
        index -= isum;
    }
    levels[0] = crt_level;
    indices[0] = index1 & ((1 << levels[0]) - 1);
}


/* conversion function: index to levels vector of the grid point */
__host__ __device__ void idx2l(int num_dims, int num_levels,
                               int index, int *levels)
{
    #ifndef __CUDA_ARCH__
        #define DP_MAT(i,j)     H_DP_MAT(i,j)
        #define LIDX_VECT(i)    h_lidx_vect[(i)]
    #else
        #define DP_MAT(i,j)     D_DP_MAT(i,j)
        #define LIDX_VECT(i)    d_lidx_vect[(i)]
    #endif

    int i, index3, position;
    int crt_level = 0;

    /* determine the group level */
    while (LIDX_VECT(crt_level) <= index && crt_level < num_levels)
        ++crt_level;

    index3 = LIDX_VECT(--crt_level);            /* the original index3 */
    index -= index3;

    position = index >> crt_level;              /* position of the vector inside its group */

    index = position;

    for (i = num_dims - 2; i >= 0; --i) {
        int isum = 0;                           /* maximum number of vectors having i components */
        int j = crt_level;
        while (index >= isum + DP_MAT(i,j)) {
            isum += DP_MAT(i,j);
            --j;
        }
        levels[i + 1] = crt_level - j;
        crt_level = j;
        index -= isum;
    }
    levels[0] = crt_level;
}


__device__ int l2idx(int num_dims, int num_levels,
                     int *levels)
{
    int i, j, sum, index;

    index = 0;
    sum = levels[0];
    for (i = 1; i < num_dims; i++) {
        sum += levels[i];
        for (j = 0; j < levels[i]; j++)
            index += D_DP_MAT((i - 1),(sum - j));
    }

    return index;
}


/* gets value of left parent in dimension crt_dim */
__device__ float get_left_parent_val(int num_dims, int num_levels,
                                     float *sg1d,
                                     int *levels, int *indices, int crt_dim)
{
    int plevel, pindex, saved_index, saved_level;

    if (indices[crt_dim] == 0)
        return 0.0f;

    saved_index = indices[crt_dim];
    saved_level = levels[crt_dim];

    plevel = saved_level - 1;
    pindex = saved_index;
    // while even
    while ((pindex & 1) == 0) {
        pindex >>= 1;
        plevel--;
    }
    pindex >>= 1;

    return sg1d[gp2idx(num_dims, num_levels, crt_dim, plevel, pindex,
                       levels, indices)];
}


/* gets value of right parent in dimension crt_dim */
__device__ float get_right_parent_val(int num_dims, int num_levels,
                                      float *sg1d,
                                      int *levels, int *indices, int crt_dim)
{
    int plevel, pindex, saved_index, saved_level;

    if (indices[crt_dim] == (1 << levels[crt_dim]) - 1)
        return 0.0f;

    // save index and level for current dimension
    saved_index = indices[crt_dim];
    saved_level = levels[crt_dim];

    plevel = saved_level - 1;
    pindex = saved_index + 1;
    // while even
    while ((pindex & 1) == 0) {
        pindex >>= 1;
        plevel--;
    }
    pindex >>= 1;

    return sg1d[gp2idx(num_dims, num_levels, crt_dim, plevel, pindex,
                       levels, indices)];
}


/* gets value of left parent in dimension crt_dim */
__device__ float get_left_parent_val_fast(int num_dims,
                                          float *sg1d,
                                          int *levels, int *indices, int crt_dim, int *par_idx)
{
#define CRT_LEVEL(i) (((i) == crt_dim) ? plevel : levels[(i)])
#define CRT_INDEX(i) (((i) == crt_dim) ? pindex : indices[(i)])

    int i, plevel, pindex, saved_index, saved_level, index1;

    if (indices[crt_dim] == 0)
        return 0.0f;

    saved_index = indices[crt_dim];
    saved_level = levels[crt_dim];

    plevel = saved_level - 1;
    pindex = saved_index;
    // while even
    while ((pindex & 1) == 0) {
        pindex >>= 1;
        plevel--;
    }
    pindex >>= 1;

    index1 = CRT_INDEX(0);
    for (i = 1; i < num_dims; ++i)
        index1 = (index1 << CRT_LEVEL(i)) + CRT_INDEX(i);

    return sg1d[par_idx[plevel] + index1];
}


/* gets value of right parent in dimension crt_dim */
__device__ float get_right_parent_val_fast(int num_dims,
                                           float *sg1d,
                                           int *levels, int *indices, int crt_dim, int *par_idx)
{
#define CRT_LEVEL(i) (((i) == crt_dim) ? plevel : levels[(i)])
#define CRT_INDEX(i) (((i) == crt_dim) ? pindex : indices[(i)])

    int plevel, pindex, saved_index, saved_level, index1, i;

    if (indices[crt_dim] == (1 << levels[crt_dim]) - 1)
        return 0.0f;

    // save index and level for current dimension
    saved_index = indices[crt_dim];
    saved_level = levels[crt_dim];

    plevel = saved_level - 1;
    pindex = saved_index + 1;
    // while even
    while ((pindex & 1) == 0) {
        pindex >>= 1;
        plevel--;
    }
    pindex >>= 1;

    index1 = CRT_INDEX(0);
    for (i = 1; i < num_dims; ++i)
        index1 = (index1 << CRT_LEVEL(i)) + CRT_INDEX(i);

    return sg1d[par_idx[plevel] + index1];
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - none
*/
__global__ void hierarchize0(int num_dims, int num_levels, int num_grid_points,
                             float *sg1d,
                             int crt_dim, int crt_level)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;           // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *indices = shared + num_dims * (blockDim.x/WARP_SIZE + threadIdx.x);

    const int subspace = warp_id;                                // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);// subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);
        
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {
            
            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {
                
                // compute (l,_) for this subspace
                if (0 == lane) {
                    idx2l(num_dims, num_levels,
                          index, levels);
                }
                
                // convert index within subspace to (_,i)
                int rest = index - subspace_start;
                for (int m = num_dims - 1; m >= 0; --m) {
                    indices[m] = rest & ((1 << levels[m]) - 1);
                    rest >>= levels[m];
                }
                
                sg1d[index] -= (get_left_parent_val(num_dims, num_levels,
                                                    sg1d,
                                                    levels, indices, crt_dim) +
                                get_right_parent_val(num_dims, num_levels,
                                                     sg1d,
                                                     levels, indices, crt_dim)) * 0.5f;
            }
        }
    }
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
*/
__global__ void hierarchize1(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim, int crt_level)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;           // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *indices = shared + num_dims * (blockDim.x/WARP_SIZE + threadIdx.x);

    const int subspace = warp_id;                                // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);// subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);
        
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {
            
            // compute (l,_) for this subspace
            if (0 == lane) {
                idx2l(num_dims, num_levels,
                      index, levels);
            }

            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {
                // convert index within subspace to (_,i)
                int rest = index - subspace_start;
                for (int m = num_dims - 1; m >= 0; --m) {
                    indices[m] = rest & ((1 << levels[m]) - 1);
                    rest >>= levels[m];
                }
                
                sg1d[index] -= (get_left_parent_val(num_dims, num_levels,
                                                    sg1d,
                                                    levels, indices, crt_dim) +
                                get_right_parent_val(num_dims, num_levels,
                                                     sg1d,
                                                     levels, indices, crt_dim)) * 0.5f;
            }
        }
    }
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
*/
__global__ void hierarchize2(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim, int crt_level)
{   
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;           // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *indices = shared + num_dims * (blockDim.x/WARP_SIZE + threadIdx.x);
	int *par_idx = shared + num_dims * (blockDim.x/WARP_SIZE + blockDim.x) + num_levels * local_warp_id;
    
    const int subspace = warp_id;                                // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);// subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {

            if (0 == lane) {
                // compute (l,_) for this subspace
                idx2l(num_dims, num_levels,
                      index, levels);

                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {

                // convert index within subspace to (_,i)
                int rest = index - subspace_start;
                for (int m = num_dims - 1; m >= 0; --m) {
                    indices[m] = rest & ((1 << levels[m]) - 1);
                    rest >>= levels[m];
                }

                sg1d[index] -= (get_left_parent_val_fast(num_dims,
                                                         sg1d,
                                                         levels, indices, crt_dim, par_idx) +
                                get_right_parent_val_fast(num_dims,
                                                          sg1d,
                                                          levels, indices, crt_dim, par_idx)) * 0.5f;
            }
        }
    }
} 


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
   - reduces the complexity O(num_dims) -> O(1) of converting indices -> index (opt3)
*/
__global__ void hierarchize3(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim, int crt_level)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;           // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * threadIdx.x;
    int *indices = shared + num_dims * (blockDim.x + threadIdx.x);
	int *par_idx = shared + num_dims * blockDim.x * 2 + num_levels * local_warp_id;
    
    const int subspace = warp_id;                                // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);// subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);
        
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {
            
            // convert index within subspace to (l,i)
            idx2gp(num_dims, num_levels,
                   index, levels, indices);
            
            if (0 == lane) {
                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {
                sg1d[index] -= (get_left_parent_val_fast(num_dims,
                                                         sg1d,
                                                         levels, indices, crt_dim, par_idx) +
                                get_right_parent_val_fast(num_dims,
                                                          sg1d,
                                                          levels, indices, crt_dim, par_idx)) * 0.5f;
                
                // compute indices for the next point in subspace (at index = index + WARP_SIZE)
                int to_add = WARP_SIZE;
                int i = num_dims - 1;
                while (to_add > 0 && i >= 0) {
                    int temp = indices[i] + to_add;
                    indices[i] = temp & ((1 << levels[i]) - 1);
                    to_add = temp >> levels[i];
                    --i;
                }
            }
        }
    }
}


__host__ __device__ int get_left_li(int l, int i, int *ll, int *li)
{
    *ll = l - 1;
    *li = i;
    // while even
    while ((*li & 1) == 0) {
        *li >>= 1;
        (*ll)--;
    }
    *li >>= 1;
    
    return 0;
}


__host__ __device__ int get_right_li(int l, int i, int *rl, int *ri)
{
    *rl = l - 1;
    *ri = i + 1;
    // while even
    while ((*ri & 1) == 0) {
        *ri >>= 1;
        (*rl)--;
    }
    *ri >>= 1;
    
    return 0;
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
   - no conversion indices -> index (opt5)
*/
__global__ void hierarchize4(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim, int crt_level)
{   
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;                  // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *par_idx = shared + num_dims * blockDim.x/WARP_SIZE + num_levels * local_warp_id;
    int *prefix_sums = shared + (num_dims + num_levels) * blockDim.x/WARP_SIZE + (num_dims + 1) * local_warp_id;
    
    const int subspace = warp_id;                                // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);// subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);
        
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {
            
            if (0 == lane) {
                // compute (l,_) for this subspace
                idx2l(num_dims, num_levels,
                      index, levels);

                prefix_sums[num_dims] = 0;
                prefix_sums[num_dims - 1] = levels[num_dims - 1];
                for (int l = num_dims - 2; l >= 0; --l)
                    prefix_sums[l] = prefix_sums[l + 1] + levels[l];

                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {
                const int l = index - subspace_start;
                const int idx_i = (l >> prefix_sums[crt_dim + 1]) & ((1 << levels[crt_dim]) - 1);
                
                float left_val;
                if (0 == idx_i) {
                    left_val = 0.0f;
                } else {
                    int left_lev_i, left_idx_i;
                    get_left_li(levels[crt_dim], idx_i, &left_lev_i, &left_idx_i);
                    const int left_idx = par_idx[left_lev_i] + 
                                         ((l >> prefix_sums[crt_dim]) << (left_lev_i + prefix_sums[crt_dim + 1])) + 
                                         (left_idx_i << prefix_sums[crt_dim + 1]) + 
                                         (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                    left_val = sg1d[left_idx];
                }
                
                float right_val;
                if (idx_i == (1 << levels[crt_dim]) - 1) {
                    right_val = 0.0f;
                } else {
                    int right_lev_i, right_idx_i;
                    get_right_li(levels[crt_dim], idx_i, &right_lev_i, &right_idx_i);
                    const int right_idx = par_idx[right_lev_i] + 
                                          ((l >> prefix_sums[crt_dim]) << (right_lev_i + prefix_sums[crt_dim + 1])) + 
                                          (right_idx_i << prefix_sums[crt_dim + 1]) +
                                          (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                    right_val = sg1d[right_idx];
                }

                sg1d[index] -= (left_val + right_val) * 0.5f;
            }
        }
    }
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
   - no conversion indices -> index (opt5)
   - use lookup table for getting the (l, i) of the parents (opt6)
*/
__global__ void hierarchize5(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim, int crt_level)
{   
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;;                 // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *par_idx = shared + num_dims * blockDim.x/WARP_SIZE + num_levels * local_warp_id;
    int *prefix_sums = shared + (num_dims + num_levels) * blockDim.x/WARP_SIZE + (num_dims + 1) * local_warp_id;
    
    const int subspace = warp_id;                                  // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);  // subspaces on crt_level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        const int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        const int subspace_end = subspace_start + (1 << crt_level);

        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {
            
            if (0 == lane) {
                // compute (l,_) for this subspace
                idx2l(num_dims, num_levels,
                      index, levels);

                prefix_sums[num_dims] = 0;
                prefix_sums[num_dims - 1] = levels[num_dims - 1];
                for (int l = num_dims - 2; l >= 0; --l)
                    prefix_sums[l] = prefix_sums[l + 1] + levels[l];

                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
            for (; index < subspace_end; index += WARP_SIZE) {
                const int l = index - subspace_start;
                const int idx_i = (l >> prefix_sums[crt_dim + 1]) & ((1 << levels[crt_dim]) - 1);
                
                float left_val;
                if (idx_i == 0) {
                    left_val = 0.0f;
                } else {
                    const int left_lev_i = D_PAR_LILI(levels[crt_dim],idx_i << 2);
                    const int left_idx_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 1);
                    const int left_idx = par_idx[left_lev_i] + 
                                         ((l >> prefix_sums[crt_dim]) << (left_lev_i + prefix_sums[crt_dim + 1])) + 
                                         (left_idx_i << prefix_sums[crt_dim + 1]) + 
                                         (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                    left_val = sg1d[left_idx];
                }
                
                float right_val;
                if (idx_i == (1 << levels[crt_dim]) - 1) {
                    right_val = 0.0f;
                } else {
                    const int right_lev_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 2);
                    const int right_idx_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 3);
                    const int right_idx = par_idx[right_lev_i] + 
                                          ((l >> prefix_sums[crt_dim]) << (right_lev_i + prefix_sums[crt_dim + 1])) + 
                                          (right_idx_i << prefix_sums[crt_dim + 1]) +
                                          (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                    right_val = sg1d[right_idx];
                }
                            
                sg1d[index] -= (left_val + right_val) * 0.5f;
            }
        }
    }
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
   - no conversion indices -> index (opt5)
   - loop interchange (opt7)
*/
__global__ void hierarchize6(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;;                 // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *par_idx = shared + num_dims * blockDim.x/WARP_SIZE + num_levels * local_warp_id;
    int *prefix_sums = shared + (num_dims + num_levels) * blockDim.x/WARP_SIZE + (num_dims + 1) * local_warp_id;
    
    int crt_level = num_levels - 1;

    const int subspace = warp_id;                                       // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);       // subspaces on last level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        int subspace_end = subspace_start + (1 << crt_level);
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {

            if (0 == lane) {
                // compute (l,_) for this subspace
                idx2l(num_dims, num_levels,
                      index, levels);

                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over all possibilities for the crt_dim-th component of levels
            for (; levels[crt_dim] >= 1; --crt_level, subspace_start = par_idx[levels[crt_dim]]) {
                
                if (0 == lane) {
                    // from loop invariant code motion
                    // helps us to reduce the complexity in the innermost loop (subspace update)
                    prefix_sums[num_dims] = 0;
                    prefix_sums[num_dims - 1] = levels[num_dims - 1];
                    for (int l = num_dims - 2; l >= 0; l--)
                        prefix_sums[l] = prefix_sums[l + 1] + levels[l];                   
                }
                // all threads within the warp have an implicit barrier after the branch
                
                // compute indices for the first and last points in the subspace
                // crt_level = num_levels - 1 + levels[crt_dim] - saved_level;
                // subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
                subspace_end = subspace_start + (1 << crt_level);

                // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
                for (index = subspace_start + lane; index < subspace_end; index += WARP_SIZE) {
                    // optimization: no need to convert l to indices, transform indices, and convert back to indices of parent
                    // instead perform transformation directly on l
                    // reduces conversion's complexity from O(num_dims) to O(1)!
                    const int l = index - subspace_start;
                    const int idx_i = (l >> prefix_sums[crt_dim + 1]) & ((1 << levels[crt_dim]) - 1);
                    
                    // if no left parent
                    float left_val;
                    if (0 == idx_i) {
                        left_val = 0.0f;
                    } else {
                        int left_lev_i, left_idx_i;
                        get_left_li(levels[crt_dim], idx_i, &left_lev_i, &left_idx_i);
                        const int left_idx = par_idx[left_lev_i] + 
                                             ((l >> prefix_sums[crt_dim]) << (left_lev_i + prefix_sums[crt_dim + 1])) + 
                                             (left_idx_i << prefix_sums[crt_dim + 1]) + 
                                             (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                        left_val = sg1d[left_idx];
                    }

                    // if no right parent
                    float right_val;
                    if (idx_i == (1 << levels[crt_dim]) - 1) {
                        right_val = 0.0f;
                    } else {
                        int right_lev_i, right_idx_i;
                        get_right_li(levels[crt_dim], idx_i, &right_lev_i, &right_idx_i);
                        const int right_idx = par_idx[right_lev_i] + 
                                              ((l >> prefix_sums[crt_dim]) << (right_lev_i + prefix_sums[crt_dim + 1])) + 
                                              (right_idx_i << prefix_sums[crt_dim + 1]) +
                                              (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                        right_val = sg1d[right_idx];
                    }

                    // update value
                    sg1d[index] -= (left_val + right_val) * 0.5f;
                }

                if (0 == lane)
                    --levels[crt_dim];
            }
        }
    }
}


/* 
   Computes hierarchical coefficients
   - one warp computes one subspace
   Optimizations: 
   - move computation of (l,_) outside the loop over subspace (opt1)
   - reuses indices of the parent subspaces for all points in the subspace (opt2)
   - no conversion indices -> index (opt5)
   - loop interchange (opt7)
   - use lookup table for getting the (l, i) of the parents (opt6)
*/
__global__ void hierarchize7(int num_dims, int num_levels,
                             float *sg1d,
                             int crt_dim)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int local_warp_id = threadIdx.x / WARP_SIZE;;                 // local warp index inside the block
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp

    extern __shared__ int shared[];
    int *levels  = shared + num_dims * local_warp_id;
    int *par_idx = shared + num_dims * blockDim.x/WARP_SIZE + num_levels * local_warp_id;
    int *prefix_sums = shared + (num_dims + num_levels) * blockDim.x/WARP_SIZE + (num_dims + 1) * local_warp_id;
    
    int crt_level = num_levels - 1;

    const int subspace = warp_id;                                       // one warp per subspace
    const int num_subspaces = D_DP_MAT((num_dims - 1),crt_level);       // subspaces on last level
    
    if (subspace < num_subspaces) {
        // compute indices for the first and last points in the subspace
        int subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
        int subspace_end = subspace_start + (1 << crt_level);
        // the index corresponding to this thread
        int index = subspace_start + lane;
        if (index < subspace_end) {

            if (0 == lane) {
                // compute (l,_) for this subspace
                idx2l(num_dims, num_levels,
                      index, levels);

                // compute levels of all parent subspaces
                int saved_level = levels[crt_dim];
                for (int l = 0; l < saved_level; ++l) {
                    levels[crt_dim] = l;
                    const int pos = l2idx(num_dims, num_levels,
                                          levels);
                    par_idx[l] = d_lidx_vect[crt_level - saved_level + l] + (pos << (crt_level - saved_level + l));
                }
                levels[crt_dim] = saved_level;
            }
            // all threads within the warp have an implicit barrier after the branch
            
            // loop over all possibilities for the crt_dim-th component of levels
            for (; levels[crt_dim] >= 1; --crt_level, subspace_start = par_idx[levels[crt_dim]]) {
                
                if (0 == lane) {
                    // from loop invariant code motion
                    // helps us to reduce the complexity in the innermost loop (subspace update)
                    prefix_sums[num_dims] = 0;
                    prefix_sums[num_dims - 1] = levels[num_dims - 1];
                    for (int l = num_dims - 2; l >= 0; l--)
                        prefix_sums[l] = prefix_sums[l + 1] + levels[l];                   
                }
                // all threads within the warp have an implicit barrier after the branch
                
                // compute indices for the first and last points in the subspace
                // crt_level = num_levels - 1 + levels[crt_dim] - saved_level;
                // subspace_start = d_lidx_vect[crt_level] + subspace * (1 << crt_level);
                subspace_end = subspace_start + (1 << crt_level);

                // loop over points in subspace in interval [subspace_start, subspace_end) with step WARP_SIZE
                for (index = subspace_start + lane; index < subspace_end; index += WARP_SIZE) {
                    // optimization: no need to convert l to indices, transform indices, and convert back to indices of parent
                    // instead perform transformation directly on l
                    // reduces conversion's complexity from O(num_dims) to O(1)!
                    const int l = index - subspace_start;
                    const int idx_i = (l >> prefix_sums[crt_dim + 1]) & ((1 << levels[crt_dim]) - 1);
                    
                    // if no left parent
                    float left_val;
                    if (0 == idx_i) {
                        left_val = 0.0f;
                    } else {
                        const int left_lev_i = D_PAR_LILI(levels[crt_dim],idx_i << 2);
                        const int left_idx_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 1);
                        const int left_idx = par_idx[left_lev_i] + 
                                             ((l >> prefix_sums[crt_dim]) << (left_lev_i + prefix_sums[crt_dim + 1])) + 
                                             (left_idx_i << prefix_sums[crt_dim + 1]) + 
                                             (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                        left_val = sg1d[left_idx];
                    }

                    // if no right parent
                    float right_val;
                    if (idx_i == (1 << levels[crt_dim]) - 1) {
                        right_val = 0.0f;
                    } else {
                        const int right_lev_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 2);
                        const int right_idx_i = D_PAR_LILI(levels[crt_dim],(idx_i << 2) + 3);
                        const int right_idx = par_idx[right_lev_i] + 
                                              ((l >> prefix_sums[crt_dim]) << (right_lev_i + prefix_sums[crt_dim + 1])) + 
                                              (right_idx_i << prefix_sums[crt_dim + 1]) +
                                              (l & ((1 << prefix_sums[crt_dim + 1]) - 1));
                        right_val = sg1d[right_idx];
                    }

                    // update value
                    sg1d[index] -= (left_val + right_val) * 0.5f;
                }

                if (0 == lane)
                    --levels[crt_dim];
            }
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - none
*/
__global__ void evaluate0(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat,
                          float *out, int num_evals, int num_evals_per_warp)
{
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;

    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
            for (index = chunk_start + lane, coords = coord_mat + (index * num_dims); 
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                int index01 = 0;
                float val = 0.0f;
                // loop over sets of subspaces of different levels
                for (int i = 0; i < num_levels; ++i) {
                    // loop over subspaces of the same level
                    for (int j = 0; j < D_DP_MAT((num_dims - 1),i); ++j) {
                        
                        __syncthreads();
                        if (0 == threadIdx.x) {
                            // get grid coordinates for the first point in this subspace
                            idx2l(num_dims, num_levels,
                                  d_lidx_vect[i] + j * (1 << i), levels);
                        }
                        __syncthreads();
                        
                        float prod = 1.0f;
                        int index2 = 0;
                        for (int k = 0; k < num_dims; k++) {
                            const float div = (1.0f - 0.0f) / (1 << levels[k]);
                            index2 = index2 * (1 << levels[k]) + (int) ((coords[k] - 0.0f) / div);
                            const float left = (int) ((coords[k] - 0.0f) / div) * div;
                            const float m = (2.0f * (coords[k] - left) - div) / div;
                            prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                        }
                        prod *= sg1d[index01 + index2];
                        val += prod;
                        index01 += 1 << i;
                    }
                }
                out[index] = val;
            }           
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - memory layout optimization: transpose coord_mat (opt6)
*/
__global__ void evaluate1(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat_trans,
                          float *out, int num_evals, int num_evals_per_warp)
{
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;

    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
            for (index = chunk_start + lane, coords = coord_mat_trans + (index / WARP_SIZE) * num_dims * WARP_SIZE + (index % WARP_SIZE);
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                int index01 = 0;
                float val = 0.0f;
                // loop over sets of subspaces of different levels
                for (int i = 0; i < num_levels; ++i) {
                    // loop over subspaces of the same level
                    for (int j = 0; j < D_DP_MAT((num_dims - 1),i); ++j) {
                        
                        __syncthreads();
                        if (0 == threadIdx.x) {
                            // get grid coordinates for the first point in this subspace
                            idx2l(num_dims, num_levels,
                                  d_lidx_vect[i] + j * (1 << i), levels);
                        }
                        __syncthreads();
                        
                        float prod = 1.0f;
                        int index2 = 0;
                        for (int k = 0; k < num_dims; k++) {
                            const int kk = k * WARP_SIZE;
                            const float div = (1.0f - 0.0f) / (1 << levels[k]);
                            index2 = index2 * (1 << levels[k]) + (int) ((coords[kk] - 0.0f) / div);
                            const float left = (int) ((coords[kk] - 0.0f) / div) * div;
                            const float m = (2.0f * (coords[kk] - left) - div) / div;
                            prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                        }
                        prod *= sg1d[index01 + index2];
                        val += prod;
                        index01 += 1 << i;
                    }
                }
                out[index] = val;
            }
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - memory layout optimization: transpose coord_mat (opt6)
   - loop interchange: loop over interpolation points is the innermost (opt1)
*/
__global__ void evaluate2(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat_trans,
                          float *out, int num_evals, int num_evals_per_warp)
{
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;

    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            // loop over sets of subspaces of different levels
            for (int i = 0; i < num_levels; ++i) {
                // loop over subspaces of the same level
                for (int j = 0; j < D_DP_MAT((num_dims - 1),i); ++j) {
                    
                    __syncthreads();
                    if (0 == threadIdx.x) {
                        // get grid coordinates for the first point in this subspace
                        idx2l(num_dims, num_levels,
                              d_lidx_vect[i] + j * (1 << i), levels);
                    }
                    __syncthreads();
                    
                    // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
                    for (index = chunk_start + lane, coords = coord_mat_trans + (index / WARP_SIZE) * num_dims * WARP_SIZE + (index % WARP_SIZE);
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                        float prod = 1.0f;
                        int index2 = 0;
                        for (int k = 0; k < num_dims; k++) {
                            const int kk = k * WARP_SIZE;
                            const float div = (1.0f - 0.0f) / (1 << levels[k]);
                            index2 = index2 * (1 << levels[k]) + (int) ((coords[kk] - 0.0f) / div);
                            const float left = (int) ((coords[kk] - 0.0f) / div) * div;
                            const float m = (2.0f * (coords[kk] - left) - div) / div;
                            prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                        }
                        prod *= sg1d[index2];
                        out[index] += prod;
                    }
                    sg1d += 1 << i;
                }
            }
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - memory layout optimization: transpose coord_mat (opt6)
   - loop interchange: loop over interpolation points is the innermost (opt1)
   - one kernel per subspace (opt5)
*/
__global__ void evaluate3(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat_trans,
                          float *out, int num_evals, int num_evals_per_warp,
                          int crt_level, int crt_subspace)
{
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;

    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    const int warp_id = thread_id / WARP_SIZE;                   // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                // thread index within a warp

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            
            __syncthreads();
            if (0 == threadIdx.x) {
                // get grid coordinates for the first point in this subspace
                idx2l(num_dims, num_levels,
                      d_lidx_vect[crt_level] + crt_subspace * (1 << crt_level), levels);
            }
            __syncthreads();
            
            // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
            for (index = chunk_start + lane, coords = coord_mat_trans + (index / WARP_SIZE) * num_dims * WARP_SIZE + (index % WARP_SIZE);
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                float prod = 1.0f;
                int index2 = 0;
                for (int k = 0; k < num_dims; k++) {
                    const int kk = k * WARP_SIZE;
                    const float div = (1.0f - 0.0f) / (1 << levels[k]);
                    index2 = index2 * (1 << levels[k]) + (int) ((coords[kk] - 0.0f) / div);
                    const float left = (int) ((coords[kk] - 0.0f) / div) * div;
                    const float m = (2.0f * (coords[kk] - left) - div) / div;
                    prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                }
                prod *= sg1d[d_lidx_vect[crt_level] + crt_subspace * (1 << crt_level) + index2];
                out[index] += prod;
            }
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - memory layout optimization: transpose coord_mat (opt6)
   - loop interchange: loop over interpolation points is the innermost (opt1)
   - strength reduction = no divisions in the innermost loop (opt3)
*/
__global__ void evaluate4(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat_trans,
                          float *out, int num_evals, int num_evals_per_warp)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp
    
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;
    float *divs  = (float *)(((char *)shared) + num_dims * sizeof(int));
    float *inv_divs = (float *)(((char *)shared) + num_dims * sizeof(int) + num_dims * sizeof(float));

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            // loop over sets of subspaces of different levels
            for (int i = 0; i < num_levels; ++i) {
                // loop over subspaces of the same level
                for (int j = 0; j < D_DP_MAT((num_dims - 1),i); ++j) {
                    
                    __syncthreads();
                    if (0 == threadIdx.x) {
                        // get grid coordinates for the first point in this subspace
                        idx2l(num_dims, num_levels,
                              d_lidx_vect[i] + j * (1 << i), levels);
                        
                        // for loop invariant code motion
                        for (int k = 0; k < num_dims; k++) {
                            divs[k] = 1.0f / (1 << levels[k]);
                            // for strength reduction
                            inv_divs[k] = 1 << levels[k];
                        }
                    }
                    __syncthreads();

                    // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
                    for (index = chunk_start + lane, coords = coord_mat_trans + (index / WARP_SIZE) * num_dims * WARP_SIZE + (index % WARP_SIZE);
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                        float prod = 1.0f;
                        int index2 = 0;
                        for (int k = 0; k < num_dims; k++) {
                            const int kk = k * WARP_SIZE;
                            // strength reduction by multiplying with inverse
                            const int t = (int) (coords[kk] * inv_divs[k]);
                            // extra ILP here from regular reduction
                            index2 = index2 * (1 << levels[k]) + t;
                            const float left = t * divs[k];
                            const float m = (coords[kk] - left) * (inv_divs[k] + inv_divs[k]) - 1.0f;
                            prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                        }
                        prod *= sg1d[index2];
                        out[index] += prod;
                    }
                    sg1d += 1 << i;
                }
            }
        }
    }
}


/* 
   Evaluates sparse grid at given points
   - split out[] into a number of chunks of size 'num_evals_per_warp', each chunk being processed by one warp
   Optimizations:
   - memory layout optimization: transpose coord_mat (opt6)
   - loop interchange: loop over interpolation points is the innermost (opt1)
   - strength reduction = no divisions in the innermost loop (opt3)
   - loop invariant code motion = prefix sums => less dependencies in innermost loop (opt4)
*/
__global__ void evaluate5(int num_dims, int num_levels,
                          float *sg1d, float *coord_mat_trans,
                          float *out, int num_evals, int num_evals_per_warp)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;        // global thread index
    const int warp_id = thread_id / WARP_SIZE;                          // global warp index
    const int lane = thread_id & (WARP_SIZE - 1);                       // thread index within a warp
    
    float *coords;
    extern __shared__ int shared[];
    int *levels  = shared;
    float *divs  = (float *)(((char *)shared) + num_dims * sizeof(int));
    float *inv_divs = (float *)(((char *)shared) + num_dims * sizeof(int) + num_dims * sizeof(float));
    int *prefix_sums = (int *)(((char *)shared) + num_dims * sizeof(int) + 2 * num_dims * sizeof(float));  

    const int chunk = warp_id;                                   // one warp per chunk
    const int num_chunks = (num_evals + num_evals_per_warp - 1) / num_evals_per_warp; // ceil(num_evals / num_evals_per_warp)

    if (chunk < num_chunks) {
        // compute indices for the first and last points in the chunk
        const int chunk_start = chunk * num_evals_per_warp;
        const int chunk_end_temp = (chunk + 1) * num_evals_per_warp;
        const int chunk_end = chunk_end_temp < num_evals ? chunk_end_temp : num_evals;
        // the index corresponding to this thread
        int index = chunk_start + lane;
        if (index < chunk_end) {
            // loop over sets of subspaces of different levels
            for (int i = 0; i < num_levels; ++i) {
                // loop over subspaces of the same level
                for (int j = 0; j < D_DP_MAT((num_dims - 1),i); ++j) {
                    
                    __syncthreads();
                    if (0 == threadIdx.x) {
                        // get grid coordinates for the first point in this subspace
                        idx2l(num_dims, num_levels,
                              d_lidx_vect[i] + j * (1 << i), levels);
                        
                        // for extra ILP
                        prefix_sums[num_dims] = 0;
                        prefix_sums[num_dims - 1] = levels[num_dims - 1];
                        for (int k = num_dims - 2; k >= 0; k--)
                            prefix_sums[k] = prefix_sums[k + 1] + levels[k];

                        // for loop invariant code motion
                        for (int k = 0; k < num_dims; k++) {
                            divs[k] = 1.0f / (1 << levels[k]);
                            // for strength reduction
                            inv_divs[k] = 1 << levels[k];
                        }
                    }
                    __syncthreads();

                    // loop over interpolation points in interval [chunk_start, chunk_end) with step WARP_SIZE
                    for (index = chunk_start + lane, coords = coord_mat_trans + (index / WARP_SIZE) * num_dims * WARP_SIZE + (index % WARP_SIZE);
                         index < chunk_end; 
                         index += WARP_SIZE, coords += (WARP_SIZE * num_dims)) {
                        float prod = 1.0f;
                        int index2 = 0;
                        for (int k = 0; k < num_dims; k++) {
                            const int kk = k * WARP_SIZE;
                            // strength reduction by multiplying with inverse
                            const int t = (int) (coords[kk] * inv_divs[k]);
                            // extra ILP here from regular reduction
                            index2 += t << prefix_sums[k + 1];
                            const float left = t * divs[k];
                            const float m = (coords[kk] - left) * (inv_divs[k] + inv_divs[k]) - 1.0f;
                            prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
                        }
                        prod *= sg1d[index2];
                        out[index] += prod;
                    }
                    sg1d += 1 << i;
                }
            }
        }
    }
}


//=============================================================
//    main
//=============================================================

const char *help_msg = "Usage: adaptive_sparse_grid_bench_cuda <num. dimensions> <refinement level> <num. evals> <hier. optim.> <eval. optim> <num. evals. per warp>\n"
                       "\tHierarchical optimizations:\n"
                       "\t\topt0 - none\n"
                       "\t\topt1 - move computation of (l,_) outside the loop over subspace\n"
                       "\t\topt2 - reuses indices of the parent subspaces for all points in the subspace\n"
                       "\t\topt3 - reduces the complexity O(num_dims) -> O(1) of converting indices -> index\n"
                       "\t\topt5 - no conversion indices -> index\n"
                       "\t\topt6 - uses lookup table for getting the (l, i) of the parents\n"
                       "\t\topt7 - loop interchange\n"
                       "\t<hier. optim.> = 0 (opt0)\n"
                       "\t               = 1 (opt1)\n"
                       "\t               = 2 (opt1 + opt2)\n"
                       "\t               = 3 (opt1 + opt2 + opt3)\n"
                       "\t               = 4 (opt1 + opt2 + opt5)\n"
                       "\t               = 5 (opt1 + opt2 + opt5 + opt6)\n"
                       "\t               = 6 (opt1 + opt2 + opt5 + opt7)\n"
                       "\t               = 7 (opt1 + opt2 + opt5 + opt6 + opt7)\n"
                       "\tEvaluation optimizations:\n"
                       "\t\topt0 - none\n"
                       "\t\topt1 - loop interchange: loop over interpolation points is the innermost\n"
                       "\t\topt3 - strength reduction = no divisions in the innermost loop\n"
                       "\t\topt4 - loop invariant code motion = prefix sums => less dependencies in innermost loop\n"
                       "\t\topt5 - one kernel per subspace\n"
                       "\t\topt6 - memory layout optimization: transpose coord_mat\n"
                       "\t<eval. optim.> = 0 (opt0)\n"
                       "\t               = 1 (opt6)\n"
                       "\t               = 2 (opt6 + opt1)\n"
                       "\t               = 3 (opt6 + opt1 + opt5)\n"
                       "\t               = 4 (opt6 + opt1 + opt3)\n"
                       "\t               = 5 (opt6 + opt1 + opt3 + opt4)\n"
                       "\t<num. evals. per warp> is the number of evaluation points computed by each warp.\n"
                       "\tEach thread computes therefore <num. evals. per warp> / WARP_SIZE points.\n";

int main(int argc, char **argv)
{
    char hostname[256];
    int num_dims, num_levels, num_evals, num_grid_points, num_subspaces, num_evals_per_warp;
    int h_opt, e_opt;

    if (argc != 7) {
        printf("%s", help_msg);
        return -1;
    } else {
        num_dims = atoi(argv[1]);
        num_levels = atoi(argv[2]);
        num_evals = atoi(argv[3]);
        h_opt = atoi(argv[4]);
        e_opt = atoi(argv[5]);
        num_evals_per_warp = atoi(argv[6]);
    }

    /*****************************************/
    /************* Initialization ************/
    /*****************************************/
    
    init_restr_vec(num_dims, num_levels);
    
    init_dp_mat(num_dims, num_levels);

    init_lidx_vect(num_dims, num_levels, &num_grid_points, &num_subspaces);
    
    /* sparse grid values */
    float *h_sg1d = init_sg1d(num_dims, num_levels, num_grid_points);
    float *d_sg1d;
    cudaMalloc(&d_sg1d, num_grid_points * sizeof(float));
    
    init_par_lili(num_levels);
    
    /* evaluation points coordinates*/
    float *h_coord_mat = init_coord_mat(num_dims, num_levels, num_grid_points, num_evals);
    float *d_coord_mat;
    cudaMalloc(&d_coord_mat, num_evals * num_dims * sizeof(float));
    
    /* evaluation points coordinates transposed*/
    float *h_coord_mat_trans = init_coord_mat_trans(num_dims, num_evals, h_coord_mat);
    float *d_coord_mat_trans;
    const int size_coord_mat_trans = (((num_evals + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE) * num_dims * sizeof(float);
    cudaMalloc(&d_coord_mat_trans, size_coord_mat_trans);
    

    /*****************************************/
    /************* System info ***************/
    /*****************************************/
    
    gethostname(hostname, 256);
    printf("==============================================\n");
    printf("# host name: %s\n", hostname);
    printf("# num_levels: %d, num_dims: %d, num_evals: %d\n", num_levels, num_dims, num_evals);
    printf("# num. of gridpoints: %d\n", num_grid_points);
    double hi_flops = HI_FLOPS(num_dims, num_grid_points);
    printf("# num. of floating point ops. hierarchization: %.10lf GFlop\n", hi_flops);
    double ev_flops = EV_FLOPS(num_dims, num_subspaces, num_evals);
	printf("# num. of floating point ops. evaluation: %.10lf GFlop\n", ev_flops);
    printf("\n");

    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*****************************************/
    /************ Hierarchization ************/
    /*****************************************/

#define NUM_SUBSPACES   H_DP_MAT((num_dims - 1),j)
#define NUM_THREADS     256
#define NUM_WARPS       (NUM_THREADS / WARP_SIZE)
#define NUM_BLOCKS      (NUM_SUBSPACES / NUM_WARPS + (NUM_SUBSPACES % NUM_WARPS == 0 ? 0 : 1))

    cudaEventRecord(start, 0);

    cudaMemcpyToSymbol("d_dp_mat", h_dp_mat, num_dims * num_levels * sizeof(int));
    cudaMemcpyToSymbol("d_lidx_vect", h_lidx_vect, num_levels * sizeof(int));
    cudaMemcpy(d_sg1d, h_sg1d, num_grid_points * sizeof(float), cudaMemcpyHostToDevice);

    switch (h_opt) {
    case 0: 
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize0<<<NUM_SUBSPACES, NUM_THREADS, num_dims * (NUM_WARPS + NUM_THREADS) * sizeof(int)>>>
                    (num_dims, num_levels, num_grid_points,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break;
    case 1: 
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize1<<<NUM_SUBSPACES, NUM_THREADS, num_dims * (NUM_WARPS + NUM_THREADS) * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break;    
    case 2: 
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize2<<<NUM_SUBSPACES, NUM_THREADS, (num_dims * (NUM_WARPS + NUM_THREADS) + num_levels * NUM_WARPS) * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break;
    case 3: 
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize3<<<NUM_SUBSPACES, NUM_THREADS, (2 * num_dims * NUM_THREADS + num_levels * NUM_WARPS) * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break;
    case 4: 
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize4<<<NUM_SUBSPACES, NUM_THREADS, (num_dims + num_levels + (num_dims + 1)) * NUM_WARPS * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break;    
    case 5:
        cudaMemcpyToSymbol("d_par_lili", h_par_lili, 4 * ((1 << num_levels) - 1) * sizeof(int));
        for (int i = 0; i < num_dims; ++i)
            for (int j = num_levels - 1; j >= 0; --j) {
                hierarchize5<<<NUM_SUBSPACES, NUM_THREADS, (num_dims + num_levels + (num_dims + 1)) * NUM_WARPS * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d,
                     i, j);
                cudaDeviceSynchronize();
            }
        break; 
    case 6:
        for (int i = 0; i < num_dims; ++i) {
            const int j = num_levels - 1;
            hierarchize6<<<NUM_SUBSPACES, NUM_THREADS, (num_dims + num_levels + (num_dims + 1)) * NUM_WARPS * sizeof(int)>>>
                (num_dims, num_levels,
                 d_sg1d,
                 i);
            cudaDeviceSynchronize();
        }
        break;
    case 7:
        cudaMemcpyToSymbol("d_par_lili", h_par_lili, 4 * ((1 << num_levels) - 1) * sizeof(int));
        for (int i = 0; i < num_dims; ++i) {
            const int j = num_levels - 1;
            hierarchize7<<<NUM_SUBSPACES, NUM_THREADS, (num_dims + num_levels + (num_dims + 1)) * NUM_WARPS * sizeof(int)>>>
                (num_dims, num_levels,
                 d_sg1d,
                 i);
            cudaDeviceSynchronize();
        }
        break;
    default:
        printf("%s", help_msg);
        return -1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("exec. time hierarchization: %.10lf (ms)\n", elapsed_time);
    printf("GFLOPS rate hierarchization: %.10lf\n", hi_flops / elapsed_time * 1000.0);
    printf("\n");
    

    /*****************************************/
    /*************** Evaluation **************/
    /*****************************************/

    float *d_out;
    const int out_size = num_evals * sizeof(float);
    cudaMalloc(&d_out, out_size);
    cudaMemset(d_out, 0, out_size);
    float *h_out = (float *) malloc(out_size);
    
    const int num_threads = 256;
    const int num_blocks  =  (num_evals * WARP_SIZE) / (num_evals_per_warp * num_threads) + 
                            ((num_evals * WARP_SIZE) % (num_evals_per_warp * num_threads) != 0 ? 1 : 0);
    
    cudaEventRecord(start, 0);
    switch (e_opt) {
    case 0:
        cudaMemcpy(d_coord_mat, h_coord_mat, num_evals * num_dims * sizeof(float), cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate0, cudaFuncCachePreferL1);
        evaluate0<<<num_blocks, num_threads, num_dims * sizeof(int)>>>
            (num_dims, num_levels,
             d_sg1d, d_coord_mat,
             d_out, num_evals, num_evals_per_warp);
        break;
    case 1:
        cudaMemcpy(d_coord_mat_trans, h_coord_mat_trans, size_coord_mat_trans, cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate1, cudaFuncCachePreferL1);
        evaluate1<<<num_blocks, num_threads, num_dims * sizeof(int)>>>
            (num_dims, num_levels,
             d_sg1d, d_coord_mat_trans,
             d_out, num_evals, num_evals_per_warp);
        break;
    case 2:
        cudaMemcpy(d_coord_mat_trans, h_coord_mat_trans, size_coord_mat_trans, cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate2, cudaFuncCachePreferL1);
        evaluate2<<<num_blocks, num_threads, num_dims * sizeof(int)>>>
            (num_dims, num_levels,
             d_sg1d, d_coord_mat_trans,
             d_out, num_evals, num_evals_per_warp);
        break;
    case 3:
        cudaMemcpy(d_coord_mat_trans, h_coord_mat_trans, size_coord_mat_trans, cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate3, cudaFuncCachePreferL1);
        // loop over sets of subspaces of different levels
        for (int i = 0; i < num_levels; ++i) {
            // loop over subspaces of the same level
            for (int j = 0; j < H_DP_MAT((num_dims - 1),i); ++j) {
                evaluate3<<<num_blocks, num_threads, num_dims * sizeof(int)>>>
                    (num_dims, num_levels,
                     d_sg1d, d_coord_mat_trans,
                     d_out, num_evals, num_evals_per_warp,
                     i, j);
                cudaDeviceSynchronize();
            }
        }
        break;
    case 4:
        cudaMemcpy(d_coord_mat_trans, h_coord_mat_trans, size_coord_mat_trans, cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate4, cudaFuncCachePreferL1);
        evaluate4<<<num_blocks, num_threads, num_dims * sizeof(int) + 2 * num_dims * sizeof(float)>>>
            (num_dims, num_levels,
             d_sg1d, d_coord_mat_trans,
             d_out, num_evals, num_evals_per_warp);
        break;
    case 5:
        cudaMemcpy(d_coord_mat_trans, h_coord_mat_trans, size_coord_mat_trans, cudaMemcpyHostToDevice);
        cudaFuncSetCacheConfig(evaluate5, cudaFuncCachePreferL1);
        evaluate5<<<num_blocks, num_threads, num_dims * sizeof(int) + 2 * num_dims * sizeof(float) + (num_dims + 1) * sizeof(int)>>>
            (num_dims, num_levels,
             d_sg1d, d_coord_mat_trans,
             d_out, num_evals, num_evals_per_warp);
        break;
    default:
        printf("%s", help_msg);
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("exec. time evaluation: %.10lf (ms)\n", elapsed_time);
    printf("GFLOPS rate evaluation: %.10lf\n", ev_flops / elapsed_time * 1000.0);
    printf("\n");

    /*****************************************/
    /********** Correctness test *************/
    /*****************************************/

    for (int i = 0; i < num_evals; i++)
        assert(fabs(h_out[i] - fct(num_dims, &h_coord_mat[i * num_dims])) < 0.000001f);
    printf("Correctness test passed\n");
    printf("==============================================\n");

    cudaFree(d_sg1d);
    cudaFree(d_coord_mat);
    cudaFree(d_coord_mat_trans);
    cudaFree(d_out);
    
    free(h_sg1d);
    free(h_coord_mat);
    free(h_coord_mat_trans);
    free(h_out);

    return 0;
}