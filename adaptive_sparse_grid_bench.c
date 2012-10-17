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


/* global variables */
int num_levels, num_dims, num_grid_points, num_evals;
int **dyn_count_mat;		/* dyn_count_mat[i][j] contains the number of vectors with i components 
				   whose norm is j and the constraints from restr_vec are applied */
int *ljmp_vec;			/* ljmp_vec[i] contains the number fo points on level < i */
int *restr_vec;			/* restrictions, the i-th value in a level vector must be <= restr_vec[i]*/
float *sg1d;


//=============================================================
//    helper functions
//=============================================================

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_usec / 1000000.0) + tv.tv_sec;
}


//=============================================================
//    helper functions
//=============================================================

float fct(float *coords)
{
    int i;
    float res = 0;
    for (i = 0; i < num_dims; i++)
	res += coords[i] * coords[i] * (1.0f - coords[i]) * (1.0f - coords[i]);
    return res;
}


void init_dyn_count_mat()
{
    int i, j, k;

    // allocate memory
    dyn_count_mat = (int **) malloc(num_dims * sizeof(int *));
    for (i = 0; i < num_dims; ++i)
        dyn_count_mat[i] = (int *) malloc(num_levels * sizeof(int));
    
    // initialize
    for (i = 0; i < num_dims; ++i)
        for (j = 0; j < num_levels; ++j)
            dyn_count_mat[i][j] = 0;
        
    // dynamic programming
    for (j = 0; j <= MIN(num_levels - 1, restr_vec[0]); ++j)
        dyn_count_mat[0][j] = 1;
    for (i = 0; i < num_dims; ++i)
        dyn_count_mat[i][0] = 1;
    
    for (i = 1; i < num_dims; ++i)
        for (j = 1; j < num_levels; ++j)
            for (k = 0; k <= MIN(j, restr_vec[i]); ++k)
                dyn_count_mat[i][j] += dyn_count_mat[i - 1][j - k];
}


void init_ljmp_vec()
{
    int i, num_grid_points_before;
    
    ljmp_vec = (int *) malloc(num_levels * sizeof(int));
    num_grid_points_before = 0;
    for (i = 0; i < num_levels; ++i) {
        ljmp_vec[i] = num_grid_points_before;
        num_grid_points_before += (dyn_count_mat[num_dims - 1][i] << i);
    }
    num_grid_points = num_grid_points_before;
}


void idx2gp(int index, int *levels, int *indices);

void init_sparse_grid_points()
{
    int i, j;
    int levels[num_dims], indices[num_dims];
    float coords[num_dims];

    sg1d = (float *) malloc(num_grid_points * sizeof(float));
    
    // fill sparse grid with function values
    for (i = 0; i < num_grid_points; ++i) {
	idx2gp(i, levels, indices);
	for (j = 0; j < num_dims; ++j)
	    coords[j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
	sg1d[i] = fct(coords);
    }
}


/* generate random evaluation points */
void generate_rand_points(int num, float *coord_mat)
{
    int i, j;
    int levels[num_dims], indices[num_dims];
    int rand_idx;
    
    for (i = 0; i < num; ++i) {
	rand_idx = random() % num_grid_points;
	idx2gp(rand_idx, levels, indices);
	for (j = 0; j < num_dims; ++j)
	    coord_mat[i * num_dims + j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
    }
}


/* returns number of vectors with 'd' components, whose norm is 'sum' */
int dyn_count_valid_levels(int d, int sum) {
    if ((d < 1 || d > num_dims) || (sum < 0 || sum > num_levels - 1))
        return 0;
    return dyn_count_mat[d - 1][sum];
}


/* returns the position of a vector 'levels' inside a group with the same norm 'sum' */
int position(int *levels, int sum) {
    int idx = 0;
    int i, j;
    for (i = num_dims - 1; i >= 0; --i) {
        for (j = 0; j <= levels[i] - 1; ++j)
            idx += dyn_count_valid_levels(i, sum - j);
        sum -= levels[i];
    }
    return idx;
}


/* initialization function */
void init()
{
    init_dyn_count_mat();
    init_ljmp_vec();
    init_sparse_grid_points();
}

//=============================================================
//    sparse grid operations
//=============================================================

/* conversion function: grid point to index */
int gp2idx(int *levels, int *indices)
{
    int index1, index2, index3, i, crt_level;

    index1 = indices[0];
    crt_level = levels[0];
    for (i = 1; i < num_dims; i++) {
        index1 = (index1 << levels[i]) + indices[i];
        crt_level += levels[i];
    }

    index2 = position(levels, crt_level) << crt_level;

    index3 = ljmp_vec[crt_level];

    return index1 + index2 + index3;
}


/* conversion function: index to grid point */
void idx2gp(int index, int *levels, int *indices)
{
    int i, index1, index3, position;
    int crt_level = 0;

    /* determine the group level */
    while (ljmp_vec[crt_level] <= index && crt_level < num_levels)
	++crt_level;

    index3 = ljmp_vec[--crt_level]; 		/* the original index3 */
    index -= index3;

    position = index >> crt_level; 		/* position of the vector inside its group */
    index1 = index % (1 << crt_level); 		/* the original index1 */

    index = position;

    for (i = num_dims - 2; i >=0; --i) {
	int isum = 0;				/* maximum number of vectors having i components */
	int j = crt_level;
	while (index >= isum + dyn_count_mat[i][j]) {
	    isum += dyn_count_mat[i][j];
	    --j;
	}
	levels[i + 1] = crt_level - j;
	indices[i + 1] = index1 % (1 << levels[i + 1]);
	index1 >>= levels[i + 1];
	crt_level = j;
	index -= isum;
    }
    levels[0] = crt_level;
    indices[0] = index1 % (1 << levels[0]);
}


/* gets value of left parent in dimension crt_dim */
float getLeftParentVal(int *levels, int *indices, int crt_dim)
{
    int plevel, pindex, saved_index, saved_level;
    float val;

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

    indices[crt_dim] = pindex;
    levels[crt_dim] = plevel;
    val = sg1d[gp2idx(levels, indices)];

    indices[crt_dim] = saved_index;
    levels[crt_dim] = saved_level;
    
    return val;
}


/* gets value of right parent in dimension crt_dim */
float getRightParentVal(int *levels, int *indices, int crt_dim)
{
    int plevel, pindex, saved_index, saved_level;
    float val; 

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

    indices[crt_dim] = pindex;
    levels[crt_dim] = plevel;
    val = sg1d[gp2idx(levels, indices)];

    // restore index and level for current dimension
    indices[crt_dim] = saved_index;
    levels[crt_dim] = saved_level;
    
    return val;
}


/* computes hierarchical coefficients (no optimizations) */
void hierarchize0()
{
    int i, j;
    int levels[num_dims], indices[num_dims];

    printf("%s\n", __func__);
    // loop over dimensions
    for (i = 0; i < num_dims; ++i) {
	// loop over each grid point
	for (j = num_grid_points - 1; j >= 0; --j) {
	    idx2gp(j, levels, indices);
	    sg1d[j] -= (getLeftParentVal(levels, indices, i) + getRightParentVal(levels, indices, i)) * 0.5f;
	}
    }
}


/* computes hierarchical coefficients (caches the level within a subspace) */
void hierarchize1()
{
    int i, j, k, t, u;
    int levels[num_dims], indices[num_dims];
    int index, subspace_index;

    printf("%s\n", __func__);
    // loop over dimensions
    for (i = 0; i < num_dims; ++i) {
	// loop over sets of subspaces (bottom - up)
	for (j = num_levels - 1; j >= 0; --j) {
	    index = ljmp_vec[j];
	    // loop over subspaces of the same level
	    for (k = 0; k < dyn_count_mat[num_dims - 1][j]; ++k) {
		// get grid coordinates for the first point in this subspace
		idx2gp(ljmp_vec[j] + k * (1 << j), levels, indices);
		// loop over points in subspace
		for (t = 0; t < (1 << j); ++t) {
		    sg1d[index] -= (getLeftParentVal(levels, indices, i) + 
				    getRightParentVal(levels, indices, i)) * 0.5f;

		    // compute indices for the next point in subspace
		    subspace_index = t + 1;
		    for (u = num_dims - 2; u >= 0; --u) {
			indices[u + 1] = subspace_index % (1 << levels[u + 1]);
			subspace_index >>= levels[u + 1];
		    }
		    indices[0] = subspace_index % (1 << levels[0]);

		    ++index;
		}
	    }
	}
    }
}


/* computes hierarchical coefficients (caches the level within a subspace 
   and efficiently computes next index) */
void hierarchize2()
{
    int i, j, k, t, u;
    int levels[num_dims], indices[num_dims];
    int index;

    printf("%s\n", __func__);
    // loop over dimensions
    for (i = 0; i < num_dims; ++i) {
	// loop over sets of subspaces (bottom - up)
	for (j = num_levels - 1; j >= 0; --j) {
	    index = ljmp_vec[j];
	    // loop over subspaces of the same level
	    for (k = 0; k < dyn_count_mat[num_dims - 1][j]; ++k) {
		// get grid coordinates for the first point in this subspace
		idx2gp(ljmp_vec[j] + k * (1 << j), levels, indices);
		// loop over points in subspace
		for (t = 0; t < (1 << j); ++t) {
		    sg1d[index] -= (getLeftParentVal(levels, indices, i) + 
				    getRightParentVal(levels, indices, i)) * 0.5f;

		    // compute indices for the next point in subspace
		    for (u = num_dims - 1; u >= 0; --u) {
			if (indices[u] == (1 << levels[u]) - 1) {
			    indices[u] = 0;
			} else {
			    ++indices[u];
			    break;
			}
		    }

		    ++index;
		}
	    }
	}
    }
}


/* evaluates sparse grid at given points */
void evaluate0(float *coord_mat, float *out)
{
    int i, j, c, k;
    int levels[num_dims], indices[num_dims];
    int index01, index2;
    float left, prod, div, m;

    for (i = 0; i < num_evals; ++i)
	out[i] = 0;

    // loop over interpolation points
    for (c = 0; c < num_evals; ++c, coord_mat += num_dims) {
	index01 = 0;
	// loop over sets of subspaces of different levels
	for (i = 0; i < num_levels; ++i) {
	    // loop over subspaces of the same level
	    for (j = 0; j < dyn_count_mat[num_dims - 1][i]; ++j) {
		// get grid coordinates for the first point in this subspace
		idx2gp(ljmp_vec[i] + j * (1 << i), levels, indices);
		prod = 1.0f;
		index2 = 0;
		for (k = 0; k < num_dims; k++) {
		    div = (1.0f - 0.0f) / (1 << levels[k]);
		    index2 = index2 * (1 << levels[k]) + (int) ((coord_mat[k] - 0.0f) / div);
		    left = (int) ((coord_mat[k] - 0.0f) / div) * div;
		    m = (2.0f * (coord_mat[k] - left) - div) / div;
		    prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
		}
		prod *= sg1d[index01 + index2];
		out[c] += prod;

		index01 += 1 << i;
	    }
	}
    }
}


/* evaluates sparse grid at given points - loop interchange*/
void evaluate1(float *coord_mat, float *out)
{
    int i, j, c, k;
    int levels[num_dims], indices[num_dims];
    int index2;
    float left, prod, div, *coords, m;
    float *sg = sg1d;

    for (i = 0; i < num_evals; ++i)
	out[i] = 0;

    // loop over sets of subspaces of different levels
    for (i = 0; i < num_levels; ++i) {
	// loop over subspaces of the same level
	for (j = 0; j < dyn_count_mat[num_dims - 1][i]; ++j) {
	    coords = coord_mat;
	    // get grid coordinates for the first point in this subspace
	    idx2gp(ljmp_vec[i] + j * (1 << i), levels, indices);
	    // loop over interpolation points
	    for (c = 0; c < num_evals; ++c, coords += num_dims) {
		prod = 1.0f;
		index2 = 0;
		for (k = 0; k < num_dims; k++) {
		    div = (1.0f - 0.0f) / (1 << levels[k]);
		    index2 = index2 * (1 << levels[k]) + (int) ((coords[k] - 0.0f) / div);
		    left = (int) ((coords[k] - 0.0f) / div) * div;
		    m = (2.0f * (coords[k] - left) - div) / div;
		    prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
		}

		prod *= sg[index2];
		out[c] += prod;
	    }

	    sg += 1 << i;
	}
    }
}



//=============================================================
//    main
//=============================================================

int main(int argc, char **argv)
{
    char hostname[256];
    double et;
    float *coord_mat, *out;
    int i, hierarchization_mode, evaluation_mode;

    if (argc != 6) {
	printf("Usage: adaptive_sparse_grid_bench <num. dimensions> <refinement level> <num. evals> <hierarchization mode: [0|1|2]> <evaluation mode: [0|1]>\n");
	return -1;
    } else {
	num_dims = atoi(argv[1]);
	num_levels = atoi(argv[2]);
	num_evals = atoi(argv[3]);
	hierarchization_mode = atoi(argv[4]);
	evaluation_mode = atoi(argv[5]);
    }

    // TODO: initialize restrictions vector
    restr_vec = (int *) malloc(num_dims * sizeof(int));
    for (i = 0; i < num_dims; ++i)
	restr_vec[i] = num_levels - 1;

    init();

    coord_mat = (float *) malloc(num_evals * num_dims * sizeof(float));
    out = (float *) malloc(num_evals * sizeof(float));

    generate_rand_points(num_evals, coord_mat);

    // info
    gethostname(hostname, 256);
    printf("==============================================\n");
    printf("# host name: %s\n", hostname);
    printf("# num_levels: %d, num_dims: %d, num_evals: %d\n", num_levels, num_dims, num_evals);
    printf("# num. of gridpoints: %d\n", num_grid_points);
    printf("\n");

    // execution time hierarchization
    et = get_time();
    switch (hierarchization_mode) {
	case 0: hierarchize0(); break;
	case 1: hierarchize1(); break;
	case 2: hierarchize2(); break;
	default: printf("Wrong hierarchization mode!\n"); exit(0);
    }
    et = get_time() - et;
    printf("exec. time hierarchization: %.10lf\n", et);
    printf("\n");
    
    // execution time evaluation
    et = get_time();
    switch (evaluation_mode) {
	case 0: evaluate0(coord_mat, out); break;
	case 1: evaluate1(coord_mat, out); break;
	default: printf("Wrong evaluation mode!\n"); exit(0);
    }
    et = get_time() - et;
    printf("exec. time evaluation: %.10lf\n", et);
    printf("\n");

    // corectness test
    for (i = 0; i < num_evals; i++)
	assert(fabs(out[i] - fct(&coord_mat[i * num_dims])) < 0.000001f);
    printf("Correctness test passed\n");
    printf("==============================================\n");

    return 0;
}

