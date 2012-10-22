/**********************************************************************************
 *
 * The Sparse Grid Benchmark
 * Copyright (c) 2009 Alin Murarasu
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * For any other enquiries send an email to Alin Murarasu, murarasu@in.tum.de.
 *
 * When publishing work that is based on this program please cite:
 * A. Murarasu, J. Weidendorfer, G. Buse, D. Butnaru, and D. Pflueger:
 * "Compact Data Structure and Scalable Algorithms for the Sparse Grid Technique"
 * PPoPP, Feb. 2011
 *
 *********************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include <xmmintrin.h>
#include <emmintrin.h>

#include <omp.h>


//=============================================================
//    utils
//=============================================================

#define MIN(x, y)	((x < y)? x: y)

#define HI_FLOPS(num_dims, num_grid_points) \
	(((num_dims) * 3.0 * (double) (num_grid_points)) / 1000000000.0)

#define EV_FLOPS(num_dims, num_subspaces, num_evals) \
	(((num_evals) * (double) (num_subspaces) * (11 * (num_dims) + 2)) / 1000000000.0)
	
double get_time()
{
	struct timeval tv;	
	gettimeofday(&tv, NULL);
	return (tv.tv_usec / 1000000.0) + tv.tv_sec;
}

int combi(int n, int k)
{
	int i, c = 1;
	for (i = k + 1; i <= n; i++) {
		c *= i;
		c /= i - k;
	}
	return c;
}


//=============================================================
//    sparse grid operations
//=============================================================

// global variables
int num_levels, num_dims, num_evals, num_subspaces;
int num_grid_points;
float *sg1d;
int **par_lili;

// vector of constraints (0 <= levels[i] <= limits[i])
int *limits;
int **dp_mat;
int *lidx_vect;

float fct(float *coords)
{
	int i;
	float res = 0;
	for (i = 0; i < num_dims; i++)
		res += coords[i] * coords[i] * (1.0f - coords[i]) * (1.0f - coords[i]);
	return res;
}

int idx2gp(int, int *, int *);
int get_left_li(int, int, int *, int *);
int get_right_li(int, int, int *, int *);

// initialization function
void init()
{
	int i, j, f;
	int levels[num_dims], indices[num_dims];
	float coords[num_dims];
	
	// allocate memory
	dp_mat = (int **) malloc(num_dims * sizeof(int *));
	for (i = 0; i < num_dims; i++)
		dp_mat[i] = (int *) malloc(num_levels * sizeof(int));
	lidx_vect = (int *) malloc(num_levels * sizeof(int));
	
	// base case
	for (j = 0; j <= limits[0]; j++)
		dp_mat[0][j] = 1;
	for (i = 1; i < num_dims; i++)
		dp_mat[i][0] = 1;

	// loop over dimensions
	for (i = 1; i < num_dims; i++) {
		// loop over sums
		for (j = 1; j <= limits[i]; j++) {
			dp_mat[i][j] = dp_mat[i][j - 1] + dp_mat[i - 1][j];	
		}
		
		for (j = limits[i] + 1; j < num_levels; j++) {
			dp_mat[i][j] = dp_mat[i][j - 1] - dp_mat[i - 1][j - 1 - limits[i]] + dp_mat[i - 1][j];
		}
	}
	
	// optimization (memoization)
	num_grid_points = 0;
	num_subspaces = 0;
	f = 1;
	for (i = 0; i < num_levels; i++) {
		lidx_vect[i] = num_grid_points;
		num_grid_points += dp_mat[num_dims - 1][i] * f;
		num_subspaces += dp_mat[num_dims - 1][i];
		f *= 2;
	}
	
	// allocate memory
	sg1d = (float *) malloc(num_grid_points * sizeof(float));
	
	// fill sparse grid with random values
	for (i = 0; i < num_grid_points; i++) {
		// sg1d[i] = random();
		idx2gp(i, levels, indices);
		for (j = 0; j < num_dims; j++)
			coords[j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
		sg1d[i] = fct(coords);
	}
	
	par_lili = (int **) malloc(num_levels * sizeof(int *));
	for (i = 0; i < num_levels; i++) {
		par_lili[i] = (int *) malloc((1 << i) * 4 * sizeof(int));
		if (i == 0) {
			par_lili[i][0] = -1;
			par_lili[i][1] = 0;
			par_lili[i][2] = -1;
			par_lili[i][3] = 1;
			continue;
		}
		
		for (j = 0; j < 1 << i; j++) {
			if (j == 0) {
				par_lili[i][0] = -1;
				par_lili[i][1] = 0;
				get_right_li(i, j, &par_lili[i][2], &par_lili[i][3]);
				continue;
			}
	
			if (j == (1 << i) - 1) {
				get_left_li(i, j, &par_lili[i][j << 2], &par_lili[i][(j << 2) + 1]);
				par_lili[i][(j << 2) + 2] = -1;
				par_lili[i][(j << 2) + 3] = 1;	
				continue;
			}
			
			get_left_li(i, j, &par_lili[i][j << 2], &par_lili[i][(j << 2) + 1]);
			get_right_li(i, j, &par_lili[i][(j << 2) + 2], &par_lili[i][(j << 2) + 3]);					
		}
	}
}

// conversion function: grid point to index
int gp2idx(int *levels, int *indices)
{
	int index1, index2, index3, i, j, sum;

	index1 = indices[0];
	for (i = 1; i < num_dims; i++)
		index1 = (index1 << levels[i]) + indices[i];
		
	index2 = 0;
	sum = levels[0];
	for (i = 1; i < num_dims; i++) {
		sum += levels[i];
		for (j = 0; j < levels[i]; j++)
			index2 += dp_mat[i - 1][sum - j];
	}	
	index2 <<= sum;
	
	index3 = lidx_vect[sum];
	
	return index1 + index2 + index3;	
}

// conversion function: index to grid point
int idx2gp(int index, int *levels, int *indices)
{
	int i, sum, level, rest;

	i = 0;
	while ((i < num_levels) && (index >= lidx_vect[i]))
		i++;
	
	sum = i - 1;
	index -= lidx_vect[sum];
	rest = index & ((1 << sum) - 1);
	index >>= sum;
	
	for (i = num_dims - 1; i >= 1; i--) {
		// levels
		level = 0;
		while (index >= dp_mat[i - 1][sum - level]) {
			index -= dp_mat[i - 1][sum - level];
			level++;
		}
		levels[i] = level;
		sum -= level;
		
		// indices (this part is here as a result of loop fusion)
		indices[i] = rest & ((1 << level) - 1);
		rest >>= level;
	}
	
	level = sum;
	levels[0] = level;
	indices[0] = rest & ((1 << level) - 1);
	
	return 0;
}

int idx2l(int index, int sum, int *levels)
{
	int i, level;

	for (i = num_dims - 1; i >= 1; i--) {
		// levels
		level = 0;
		while (index >= dp_mat[i - 1][sum - level]) {
			index -= dp_mat[i - 1][sum - level];
			level++;
		}
		levels[i] = level;
		sum -= level;
	}
	
	level = sum;
	levels[0] = level;

	return 0;
}

int l2idx(int *levels)
{
	int i, j, sum, index;

	index = 0;
	sum = levels[0];
	for (i = 1; i < num_dims; i++) {
		sum += levels[i];
		for (j = 0; j < levels[i]; j++)
			index += dp_mat[i - 1][sum - j];
	}	
	
	return index;
}

// gets value of left parent in dimension crt_dim
float get_left_parent_val(int *levels, int *indices, int crt_dim)
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

	// restore index and level for current dimension	
	indices[crt_dim] = saved_index;
	levels[crt_dim] = saved_level;
	
	return val;
}

// gets value of right parent in dimension crt_dim
float get_right_parent_val(int *levels, int *indices, int crt_dim)
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

// gets value of left parent in dimension crt_dim
float get_left_parent_val_fast(int *levels, int *indices, int crt_dim, int *par_idx)
{
	int i, plevel, pindex, saved_index, saved_level, index1;
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
	
	index1 = indices[0];
	for (i = 1; i < num_dims; i++)
		index1 = (index1 << levels[i]) + indices[i];
	
	val = sg1d[par_idx[plevel] + index1];

	// restore index and level for current dimension
	indices[crt_dim] = saved_index;
	levels[crt_dim] = saved_level;
	
	return val;
}

// gets value of right parent in dimension crt_dim
float get_right_parent_val_fast(int *levels, int *indices, int crt_dim, int *par_idx)
{
	int plevel, pindex, saved_index, saved_level, index1, i;
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
	
	index1 = indices[0];
	for (i = 1; i < num_dims; i++)
		index1 = (index1 << levels[i]) + indices[i];
	
	val = sg1d[par_idx[plevel] + index1];

	// restore index and level for current dimension
	indices[crt_dim] = saved_index;
	levels[crt_dim] = saved_level;
	
	return val;
}

// gets value of left parent in dimension crt_dim
int get_left_parent_val_fast_x4(int *levels, int *indices, int crt_dim, int *par_idx, float *left_vals)
{
	int i, plevel[4], pindex[4], index1[4] __attribute__((aligned(16))), zero_flags[4];

	for (i = 0; i < 4; i++) {
		if (indices[(crt_dim << 2) + i] == 0) {
			zero_flags[i] = 1;

			left_vals[i] = 0.0f;
		} else {
			zero_flags[i] = 0;
			
			plevel[i] = levels[crt_dim] - 1;
			pindex[i] = indices[(crt_dim << 2) + i];
			// while even
			while ((pindex[i] & 1) == 0) {
				pindex[i] >>= 1;
				plevel[i]--;
			}
			pindex[i] >>= 1;
		}
	}
	
//	index1[0] = indices[0];
//	index1[1] = indices[1];
//	index1[2] = indices[2];
//	index1[3] = indices[3];
//	for (i = 1; i < num_dims; i++) {
//		index1[0] = (index1[0] << levels[i << 2]) + indices[i << 2];
//		index1[1] = (index1[1] << levels[(i << 2) + 1]) + indices[(i << 2) + 1];
//		index1[2] = (index1[2] << levels[(i << 2) + 2]) + indices[(i << 2) + 2];
//		index1[3] = (index1[3] << levels[(i << 2) + 3]) + indices[(i << 2) + 3];	
//	}

	__m128i index1_xmm = _mm_set1_epi32(0);
	__m128i indices_xmm;
	for (i = 0; i < crt_dim; i++) {
		indices_xmm = _mm_load_si128(&indices[i << 2]);
		index1_xmm = _mm_add_epi32(_mm_slli_epi32(index1_xmm, levels[i]), indices_xmm);
	}
	_mm_store_si128(index1, index1_xmm);	

	index1[0] = (index1[0] << plevel[0]) + pindex[0];
	index1[1] = (index1[1] << plevel[1]) + pindex[1];
	index1[2] = (index1[2] << plevel[2]) + pindex[2];
	index1[3] = (index1[3] << plevel[3]) + pindex[3];	
	i++;
	
	index1_xmm = _mm_load_si128(index1);
	for ( ; i < num_dims; i++) {
		indices_xmm = _mm_load_si128(&indices[i << 2]);
		index1_xmm = _mm_add_epi32(_mm_slli_epi32(index1_xmm, levels[i]), indices_xmm);
	}

	_mm_store_si128(index1, index1_xmm);
	
	// restore index and level for current dimension
	for (i = 0; i < 4; i++) {
		if (zero_flags[i] == 0) {
			left_vals[i] = sg1d[par_idx[plevel[i]] + index1[i]];
		}
	}

	return 0;
}

// gets value of right parent in dimension crt_dim
int get_right_parent_val_fast_x4(int *levels, int *indices, int crt_dim, int *par_idx, float *right_vals)
{	
	int i, plevel[4], pindex[4], index1[4] __attribute__((aligned(16))), zero_flags[4];

	for (i = 0; i < 4; i++) {
		if (indices[(crt_dim << 2) + i] == (1 << levels[crt_dim]) - 1) {
			zero_flags[i] = 1;

			right_vals[i] = 0.0f;
		} else {
			zero_flags[i] = 0;
			
			plevel[i] = levels[crt_dim] - 1;
			pindex[i] = indices[(crt_dim << 2) + i] + 1;
			// while even
			while ((pindex[i] & 1) == 0) {
				pindex[i] >>= 1;
				plevel[i]--;
			}
			pindex[i] >>= 1;
		}
	}
	
//	index1[0] = indices[0];
//	index1[1] = indices[1];
//	index1[2] = indices[2];
//	index1[3] = indices[3];
//	for (i = 1; i < num_dims; i++) {
//		index1[0] = (index1[0] << levels[i << 2]) + indices[i << 2];
//		index1[1] = (index1[1] << levels[(i << 2) + 1]) + indices[(i << 2) + 1];
//		index1[2] = (index1[2] << levels[(i << 2) + 2]) + indices[(i << 2) + 2];
//		index1[3] = (index1[3] << levels[(i << 2) + 3]) + indices[(i << 2) + 3];	
//	}

	__m128i index1_xmm = _mm_set1_epi32(0);
	__m128i indices_xmm;
	for (i = 0; i < crt_dim; i++) {
		indices_xmm = _mm_load_si128(&indices[i << 2]);
		index1_xmm = _mm_add_epi32(_mm_slli_epi32(index1_xmm, levels[i]), indices_xmm);
	}
	_mm_store_si128(index1, index1_xmm);	

	index1[0] = (index1[0] << plevel[0]) + pindex[0];
	index1[1] = (index1[1] << plevel[1]) + pindex[1];
	index1[2] = (index1[2] << plevel[2]) + pindex[2];
	index1[3] = (index1[3] << plevel[3]) + pindex[3];	
	i++;
	
	index1_xmm = _mm_load_si128(index1);
	for ( ; i < num_dims; i++) {
		indices_xmm = _mm_load_si128(&indices[i << 2]);
		index1_xmm = _mm_add_epi32(_mm_slli_epi32(index1_xmm, levels[i]), indices_xmm);
	}

	_mm_store_si128(index1, index1_xmm);
	
	// restore index and level for current dimension
	for (i = 0; i < 4; i++) {
		if (zero_flags[i] == 0) {
			right_vals[i] = sg1d[par_idx[plevel[i]] + index1[i]];
		}
	}
	
	return 0;
}

// computes hierarchical coefficients (no optimizations)
// this is the reference version
int hierarchize0()
{
	int i, j, levels[num_dims], indices[num_dims];

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over grid points
		for (j = num_grid_points - 1; j >= 0; j--) {
			idx2gp(j, levels, indices);						
			sg1d[j] -= (get_left_parent_val(levels, indices, i) + get_right_parent_val(levels, indices, i)) * 0.5f;
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// reuses the levels of each subspace for all points in that subspace (opt1)
int hierarchize1()
{
	int i, j, k, l, m, levels[num_dims], indices[num_dims], index, rest;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {
			// use precomputed index
			index = lidx_vect[j];
			
			// loop over subspaces of same level
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				// convert index of subspace to levels
				idx2l(k, j, levels);

				// loop over points in subspace
				for (l = 0; l < (1 << j); l++) {
					rest = l;
					// convert index in regular grid to indices
					for (m = num_dims - 1; m >= 0; m--) {						
						indices[m] = rest & ((1 << levels[m]) - 1);
						rest >>= levels[m];						
					}					
					
					sg1d[index] -= (get_left_parent_val(levels, indices, i) + get_right_parent_val(levels, indices, i)) * 0.5f;
					index++;
				}
			}
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// reuses indices of the parent subspaces for all points in the subspace (opt2)
// opt1 + opt2
int hierarchize2()
{
	int i, j, k, l, m, rest, levels[num_dims], indices[num_dims], index, par_idx[num_levels], saved_level;				

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {			
			
			// loop over subspaces of same level
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				
				index = lidx_vect[j] + (k << j);
			
				// convert index of subspace to levels
				idx2l(k, j, levels);

				// compute levels of all parent subspaces
				saved_level = levels[i];
				for (l = 0; l < saved_level; l++) {
					levels[i] = l;
					par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
				}
				levels[i] = saved_level;

				// loop over points in subspace
				for (l = 0; l < (1 << j); l++) {
					rest = l;
					// convert index in regular grid to indice
					for (m = num_dims - 1; m >= 0; m--) {						
						indices[m] = rest & ((1 << levels[m]) - 1);
						rest >>= levels[m];						
					}

					sg1d[index] -= (get_left_parent_val_fast(levels, indices, i, par_idx) + get_right_parent_val_fast(levels, indices, i, par_idx)) * 0.5f;
					index++;
				}
			}
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// reduces the complexity O(num_dims) -> O(1) of converting indices -> index (opt3)
// opt1 + opt2 + opt3
int hierarchize3()
{
	int i, j, k, l, m, levels[num_dims], indices[num_dims], index, par_idx[num_levels], saved_level;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {			
			
			// loop over subspaces of same level
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				
				index = lidx_vect[j] + (k << j);
			
				// convert index of subspace to levels
				idx2l(k, j, levels);

				// compute levels of all parent subspaces
				saved_level = levels[i];
				for (l = 0; l < saved_level; l++) {
					levels[i] = l;
					par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
				}
				levels[i] = saved_level;

				for (m = num_dims - 1; m >= 0; m--)
					indices[m] = 0;

					// loop over points in subspace
				for (l = 0; l < (1 << j); l++) {
					sg1d[index] -= (get_left_parent_val_fast(levels, indices, i, par_idx) + get_right_parent_val_fast(levels, indices, i, par_idx)) * 0.5f;
					index++;
				
					// compute indices of next point in subspace
					for (m = num_dims - 1; m >= 0; m--) {
						if (indices[m] == (1 << levels[m]) - 1) {
							indices[m] = 0;
						} else {
							indices[m]++;
							break;
						}
					}				
				}
			}
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// vectorization using sse intrinsics (opt4)
// opt1 + opt2 + opt4
int hierarchize4()
{
	int i, j, k, l, m, rest[4] __attribute__((aligned(16))), levels[num_dims], indices[num_dims * 4] __attribute__((aligned(16))), index, par_idx[num_levels], saved_level;
	float left_vals[4] __attribute__((aligned(16))), right_vals[4] __attribute__((aligned(16)));

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {
			index = lidx_vect[j];
			
			if ((1 << j) < 4) {
				// loop over subspaces of same level
				for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
					// convert index of subspace to levels
					idx2l(k, j, levels);

					// compute levels of all parent subspaces
					saved_level = levels[i];
					for (l = 0; l < saved_level; l++) {
						levels[i] = l;
						par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
					}
					levels[i] = saved_level;				

					for (m = num_dims - 1; m >= 0; m--)
						indices[m] = 0;
						
					// loop over points in subspace
					for (l = 0; l < (1 << j); l++) {
						sg1d[index] -= (get_left_parent_val_fast(levels, indices, i, par_idx) + get_right_parent_val_fast(levels, indices, i, par_idx)) * 0.5f;
						index++;
						
						// compute indices of next point in subspace
						for (m = num_dims - 1; m >= 0; m--) {
							if (indices[m] == (1 << levels[m]) - 1) {
								indices[m] = 0;
							} else {
								indices[m]++;
								break;
							}
						}
					}						
				}
			} else {
				// loop over subspaces of same level
				for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
					// convert index of subspace to levels
					idx2l(k, j, levels);

					// compute levels of all parent subspaces
					saved_level = levels[i];
					for (l = 0; l < saved_level; l++) {
						levels[i] = l;
						par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
					}
					levels[i] = saved_level;				

					for (m = num_dims - 1; m >= 0; m--)
						indices[m] = 0;
						
					// loop over points in subspace
					for (l = 0; l < (1 << j); l += 4, index += 4) {
						rest[0] = l;
						rest[1] = l + 1;
						rest[2] = l + 2;
						rest[3] = l + 3;
						
						int t0, t1;

						__m128i rest_xmm = _mm_load_si128(rest);
						__m128i indices_xmm;
						// convert index in regular grid to indices
						for (m = num_dims - 1; m >= 0; m--) {
							t0 = levels[m];
							t1 = (1 << t0) - 1;
													
//			    			indices[m << 2] = rest[0] & ((1 << levels[m << 2]) - 1);
//			    			indices[(m << 2) + 1] = rest[1] & ((1 << levels[m << 2]) - 1);
//			    			indices[(m << 2) + 2] = rest[2] & ((1 << levels[m << 2]) - 1);
//			    			indices[(m << 2) + 3] = rest[3] & ((1 << levels[m << 2]) - 1);
							
							indices_xmm = _mm_and_si128(rest_xmm, _mm_set1_epi32(t1));
							_mm_store_si128(&indices[m << 2], indices_xmm);

//                         rest[0] >>= levels[m << 2];
//                         rest[1] >>= levels[m << 2];
//                         rest[2] >>= levels[m << 2];
//                         rest[3] >>= levels[m << 2];
						
						   rest_xmm = _mm_srli_epi32(rest_xmm, t0);                     
						}					
					
						get_left_parent_val_fast_x4(levels, indices, i, par_idx, left_vals);
						get_right_parent_val_fast_x4(levels, indices, i, par_idx, right_vals);
							
//						sg1d[index] -= (left_vals[0] + right_vals[0]) * 0.5f;
//						sg1d[index + 1] -= (left_vals[1] + right_vals[1]) * 0.5f;
//						sg1d[index + 2] -= (left_vals[2] + right_vals[2]) * 0.5f;
//						sg1d[index + 3] -= (left_vals[3] + right_vals[3]) * 0.5f;
					
						__m128 sg1d_xmm = _mm_loadu_ps(&sg1d[index]);
						__m128 left_xmm = _mm_load_ps(left_vals);
						__m128 right_xmm = _mm_load_ps(right_vals);
						sg1d_xmm = _mm_sub_ps(sg1d_xmm, _mm_mul_ps(_mm_add_ps(left_xmm, right_xmm), _mm_set1_ps(0.5f)));
						_mm_storeu_ps(&sg1d[index], sg1d_xmm);
					}
				}
			}
		}
	}
	
	return 0;
}

int get_left_li(int l, int i, int *ll, int *li)
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

int get_right_li(int l, int i, int *rl, int *ri)
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

// computes hierarchical coefficients
// no conversion indices -> index (opt5)
// opt1 + opt2 + opt5
int hierarchize5()
{
	int i, j, k, l, levels[num_dims], index, par_idx[num_levels], saved_level, prefix_sums[num_dims + 1];
	int idx_i, left_lev_i, left_idx_i, right_lev_i, right_idx_i, left_idx, right_idx;
	float left_val, right_val;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {			
			
			// loop over subspaces of same level
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				
				index = lidx_vect[j] + (k << j);
			
				// convert index of subspace to levels
				idx2l(k, j, levels);
				
				prefix_sums[num_dims] = 0;
				prefix_sums[num_dims - 1] = levels[num_dims - 1];
				for (l = num_dims - 2; l >= 0; l--)
					prefix_sums[l] = prefix_sums[l + 1] + levels[l];
				
				// compute levels of all parent subspaces
				saved_level = levels[i];
				for (l = 0; l < saved_level; l++) {
					levels[i] = l;
					par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
				}
				levels[i] = saved_level;

				// loop over points in subspace
				for (l = 0; l < 1 << j; l++) {
					idx_i = (l >> prefix_sums[i + 1]) & ((1 << levels[i]) - 1);

					if (idx_i == 0) {
						left_val = 0.0f;
					} else {
						get_left_li(levels[i], idx_i, &left_lev_i, &left_idx_i);
						left_idx = par_idx[left_lev_i] + 
						           ((l >> prefix_sums[i]) << (left_lev_i + prefix_sums[i + 1])) + 
						           (left_idx_i << prefix_sums[i + 1]) + 
						           (l & ((1 << prefix_sums[i + 1]) - 1));
						left_val = sg1d[left_idx];
					}
					
					if (idx_i == (1 << levels[i]) - 1) {
						right_val = 0.0f;
					} else {
						get_right_li(levels[i], idx_i, &right_lev_i, &right_idx_i);
						right_idx = par_idx[right_lev_i] + 
						            ((l >> prefix_sums[i]) << (right_lev_i + prefix_sums[i + 1])) + 
						            (right_idx_i << prefix_sums[i + 1]) +
						            (l & ((1 << prefix_sums[i + 1]) - 1));
						right_val = sg1d[right_idx];
					}
								
					sg1d[index] -= (left_val + right_val) * 0.5f;
					index++;
				}
			}
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// uses lookup table for getting the (l, i) of the parents (opt6)
// opt1 + opt2 + opt5 + opt6
int hierarchize6()
{
	int i, j, k, l, levels[num_dims], index, par_idx[num_levels], saved_level, prefix_sums[num_dims + 1];
	int left_lev_i, left_idx_i, right_lev_i, right_idx_i, idx_i, left_idx, right_idx;
	float left_val, right_val;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {			
			
			// loop over subspaces of same level
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				
				index = lidx_vect[j] + (k << j);
			
				// convert index of subspace to levels
				idx2l(k, j, levels);
				
				// from loop invariant code motion
				prefix_sums[num_dims] = 0;
				prefix_sums[num_dims - 1] = levels[num_dims - 1];
				for (l = num_dims - 2; l >= 0; l--)
					prefix_sums[l] = prefix_sums[l + 1] + levels[l];
				
				// from loop invariant code motion
				// compute levels of all parent subspaces
				saved_level = levels[i];
				for (l = 0; l < saved_level; l++) {
					levels[i] = l;
					par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
				}
				levels[i] = saved_level;

				// loop over points in subspace
				// for unroll, group together same instructions, switch (2 * idx_i_1 + idx_i_0) ...
				for (l = 0; l < 1 << j; l++) {
					//  optimization: no need to convert l to indices, transform indices, and convert back to indices of parent
					// instead perform transformation directly on l
					// reduces conversion's complexity from O(num_dims) to O(1)! 
					idx_i = (l >> prefix_sums[i + 1]) & ((1 << levels[i]) - 1);

					if (idx_i == 0) {
						left_val = 0.0f;
					} else {
						left_lev_i = par_lili[levels[i]][idx_i << 2];
						left_idx_i = par_lili[levels[i]][(idx_i << 2) + 1];						
						left_idx = par_idx[left_lev_i] + 
						          ((l >> prefix_sums[i]) << (left_lev_i + prefix_sums[i + 1])) + 
						          (left_idx_i << prefix_sums[i + 1]) + 
						          (l & ((1 << prefix_sums[i + 1]) - 1));
						left_val = sg1d[left_idx];
					}
					
					if (idx_i == (1 << levels[i]) - 1) {
						right_val = 0.0f;
					} else {
						right_lev_i = par_lili[levels[i]][(idx_i << 2) + 2];
						right_idx_i = par_lili[levels[i]][(idx_i << 2) + 3];					
						right_idx = par_idx[right_lev_i] + 
						            ((l >> prefix_sums[i]) << (right_lev_i + prefix_sums[i + 1])) + 
						            (right_idx_i << prefix_sums[i + 1]) +
						            (l & ((1 << prefix_sums[i + 1]) - 1));
						right_val = sg1d[right_idx];
					}
								
					sg1d[index] -= (left_val + right_val) * 0.5f;
					index++;				
				}
			}
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
// loop interchange (opt7)
// opt1 + opt2 + opt5 + opt7
int hierarchize7()
{
	int i, k, l, levels[num_dims], index, par_idx[num_levels], saved_level, prefix_sums[num_dims + 1];
	int left_lev_i, left_idx_i, right_lev_i, right_idx_i, idx_i, left_idx, right_idx;
	float left_val, right_val;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over subspaces on the last level (bottom)
		for (k = 0; k < dp_mat[num_dims - 1][num_levels - 1]; k++) {
			// index is the location in the 1d array of the current subspace
			index = lidx_vect[num_levels - 1] + (k << (num_levels - 1));

			// convert index of subspace to levels
			idx2l(k, num_levels - 1, levels);

			// from loop invariant code motion
			// compute levels of all parent subspaces
			saved_level = levels[i];
			for (l = 0; l < saved_level; l++) {
				levels[i] = l;
				par_idx[l] = lidx_vect[num_levels - 1 - saved_level + l] + (l2idx(levels) << (num_levels - 1 - saved_level + l));
			}

			// loop over all possibilities for the i-th component of levels
			for (levels[i] = saved_level; levels[i] >= 1; levels[i]--, index = par_idx[levels[i]]) {
				// from loop invariant code motion
				// helps us to reduce the complexity in the innermost loop (subspace update)
				prefix_sums[num_dims] = 0;
				prefix_sums[num_dims - 1] = levels[num_dims - 1];
				for (l = num_dims - 2; l >= 0; l--)
					prefix_sums[l] = prefix_sums[l + 1] + levels[l];

				// loop over points in subspace (this is the update of the current subspace)
				// for unroll, group together same instructions, switch (2 * idx_i_1 + idx_i_0) ...
				for (l = 0; l < 1 << (num_levels - 1 + levels[i] - saved_level); l++) {
					//  optimization: no need to convert l to indices, transform indices, and convert back to indices of parent
					// instead perform transformation directly on l
					// reduces conversion's complexity from O(num_dims) to O(1)!
					idx_i = (l >> prefix_sums[i + 1]) & ((1 << levels[i]) - 1);

					// if no left parent
					if (idx_i == 0) {
						left_val = 0.0f;
					} else {
						// left_lev_i = par_lili[levels[i]][idx_i << 2];
						// left_idx_i = par_lili[levels[i]][(idx_i << 2) + 1];
						// or
						get_left_li(levels[i], idx_i, &left_lev_i, &left_idx_i);

						left_idx = par_idx[left_lev_i] +
							      ((l >> prefix_sums[i]) << (left_lev_i + prefix_sums[i + 1])) +
							      (left_idx_i << prefix_sums[i + 1]) +
							      (l & ((1 << prefix_sums[i + 1]) - 1));
						left_val = sg1d[left_idx];
					}

					// if no right parent
					if (idx_i == (1 << levels[i]) - 1) {
						right_val = 0.0f;
					} else {
						// right_lev_i = par_lili[levels[i]][(idx_i << 2) + 2];
						// right_idx_i = par_lili[levels[i]][(idx_i << 2) + 3];
						// or
						get_right_li(levels[i], idx_i, &right_lev_i, &right_idx_i);

						right_idx = par_idx[right_lev_i] +
							        ((l >> prefix_sums[i]) << (right_lev_i + prefix_sums[i + 1])) +
							        (right_idx_i << prefix_sums[i + 1]) +
							        (l & ((1 << prefix_sums[i + 1]) - 1));
						right_val = sg1d[right_idx];
					}

					// update value
					sg1d[index + l] -= (left_val + right_val) * 0.5f;
				}
			}
		}
	}

	return 0;
}

// computes hierarchical coefficients
// loop interchange (opt7)
// opt1 + opt2 + opt6 + opt7
int hierarchize8()
{
	int i, k, l, levels[num_dims], index, par_idx[num_levels], saved_level, prefix_sums[num_dims + 1];
	int left_lev_i, left_idx_i, right_lev_i, right_idx_i, idx_i, left_idx, right_idx;
	float left_val, right_val;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over subspaces on the last level (bottom)
		for (k = 0; k < dp_mat[num_dims - 1][num_levels - 1]; k++) {
			// index is the location in the 1d array of the current subspace
			index = lidx_vect[num_levels - 1] + (k << (num_levels - 1));

			// convert index of subspace to levels
			idx2l(k, num_levels - 1, levels);

			// from loop invariant code motion
			// compute levels of all parent subspaces
			saved_level = levels[i];
			for (l = 0; l < saved_level; l++) {
				levels[i] = l;
				par_idx[l] = lidx_vect[num_levels - 1 - saved_level + l] + (l2idx(levels) << (num_levels - 1 - saved_level + l));
			}

			// loop over all possibilities for the i-th component of levels
			for (levels[i] = saved_level; levels[i] >= 1; levels[i]--, index = par_idx[levels[i]]) {
				// from loop invariant code motion
				// helps us to reduce the complexity in the innermost loop (subspace update)
				prefix_sums[num_dims] = 0;
				prefix_sums[num_dims - 1] = levels[num_dims - 1];
				for (l = num_dims - 2; l >= 0; l--)
					prefix_sums[l] = prefix_sums[l + 1] + levels[l];

				// loop over points in subspace (this is the update of the current subspace)
				// for unroll, group together same instructions, switch (2 * idx_i_1 + idx_i_0) ...
				for (l = 0; l < 1 << (num_levels - 1 + levels[i] - saved_level); l++) {
					//  optimization: no need to convert l to indices, transform indices, and convert back to indices of parent
					// instead perform transformation directly on l
					// reduces conversion's complexity from O(num_dims) to O(1)!
					idx_i = (l >> prefix_sums[i + 1]) & ((1 << levels[i]) - 1);

					// if no left parent
					if (idx_i == 0) {
						left_val = 0.0f;
					} else {
						left_lev_i = par_lili[levels[i]][idx_i << 2];
						left_idx_i = par_lili[levels[i]][(idx_i << 2) + 1];
						// or
						// get_left_li(levels[i], idx_i, &left_lev_i, &left_idx_i);

						left_idx = par_idx[left_lev_i] +
							      ((l >> prefix_sums[i]) << (left_lev_i + prefix_sums[i + 1])) +
							      (left_idx_i << prefix_sums[i + 1]) +
							      (l & ((1 << prefix_sums[i + 1]) - 1));
						left_val = sg1d[left_idx];
					}

					// if no right parent
					if (idx_i == (1 << levels[i]) - 1) {
						right_val = 0.0f;
					} else {
						right_lev_i = par_lili[levels[i]][(idx_i << 2) + 2];
						right_idx_i = par_lili[levels[i]][(idx_i << 2) + 3];
						// or
						// get_right_li(levels[i], idx_i, &right_lev_i, &right_idx_i);

						right_idx = par_idx[right_lev_i] +
							        ((l >> prefix_sums[i]) << (right_lev_i + prefix_sums[i + 1])) +
							        (right_idx_i << prefix_sums[i + 1]) +
							        (l & ((1 << prefix_sums[i + 1]) - 1));
						right_val = sg1d[right_idx];
					}

					// update value
					sg1d[index + l] -= (left_val + right_val) * 0.5f;
				}
			}
		}
	}

	return 0;
}

// computes hierarchical coefficients
// opt1 + opt2 + opt3
int omp_hierarchize3()
{
	#pragma omp parallel
	{
	int i, j, k;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {			
			
			// loop over subspaces of same level
			#pragma omp for 
			for (k = 0; k < dp_mat[num_dims - 1][j]; k++) {
				int l, m, levels[num_dims], indices[num_dims], index, par_idx[num_levels], saved_level;				
				
				index = lidx_vect[j] + (k << j);
			
				// convert index of subspace to levels
				idx2l(k, j, levels);

				// compute levels of all parent subspaces
				saved_level = levels[i];
				for (l = 0; l < saved_level; l++) {
					levels[i] = l;
					par_idx[l] = lidx_vect[j - saved_level + l] + (l2idx(levels) << (j - saved_level + l));
				}
				levels[i] = saved_level;				
					
				for (m = num_dims - 1; m >= 0; m--)
					indices[m] = 0;

				// loop over points in subspace
				for (l = 0; l < (1 << j); l++) {
					sg1d[index] -= (get_left_parent_val_fast(levels, indices, i, par_idx) + get_right_parent_val_fast(levels, indices, i, par_idx)) * 0.5f;
					index++;
					
					// compute indices of next point in subspace
					for (m = num_dims - 1; m >= 0; m--) {
						if (indices[m] == (1 << levels[m]) - 1) {
							indices[m] = 0;
						} else {
							indices[m]++;
							break;
						}
					}
				}
			}
		}
	}
	}
	
	return 0;
}

// evaluates sparse grid at given points (no optimizations)
// this is the reference version
int evaluate0(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, i, j, levels[num_dims], index01, index2, c;
	float left, prod, div, m;

	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;
		
	for (c = 0; c < num_evals; c++, coord_mat += num_dims) {
		index01 = 0;
		// loop over sets of subspaces of different levels
		for (i = 0; i < num_levels; i++) {		
			// loop over subspaces of same level
			for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
				idx2l(j, i, levels);
				
				// loop over interpolation points
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
	
	return 0;
}

// evaluates sparse grid at given points
// loop interchange for improved locality (opt1)
int evaluate1(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, i, j, levels[num_dims], index2, c;
	float left, prod, div, *coords, m;

	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;

	// loop over sets of subspaces of different levels
	for (i = 0; i < num_levels; i++) {
		// loop over subspaces of same level
		for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
			idx2l(j, i, levels);

			coords = coord_mat;
			// loop over interpolation points
			for (c = 0; c < num_evals; c++, coords += num_dims) {
				prod = 1.0f;
				index2 = 0;
				for (k = 0; k < num_dims; k++) {
					div = (1.0f - 0.0f) / (1 << levels[k]);
					index2 = index2 * (1 << levels[k]) + (int) ((coords[k] - 0.0f) / div);
					left = (int) ((coords[k] - 0.0f) / div) * div;
					m = (2.0f * (coords[k] - left) - div) / div;
					prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
				}

				prod *= sg1d[index2];
				out[c] += prod;
			}

			sg1d += 1 << i;
		}
	}
	
	return 0;
}

// evaluates sparse grid at given points
// unroll-and-jam, data layout change, and vectorization using sse intrinsics (opt2)
int evaluate2(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, j, i, levels[num_dims], index2, c, tran_n;
	float left, prod, div, *coords, m;
	float tran_coord_mat[num_dims * (num_evals >> 2)][4] __attribute__((aligned(16)));
	float prod_x4[4] __attribute__((aligned(16)));
	int index2_x4[4] __attribute__((aligned(16)));
	
	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;
		
	tran_n = num_dims * (num_evals >> 2);
	for (i = 0; i < tran_n; i += num_dims) {
		for (j = 0; j < (num_dims >> 2) << 2; j += 4) {
			__m128 t0 = _mm_loadu_ps(&coord_mat[(i << 2) + j]);
			__m128 t1 = _mm_loadu_ps(&coord_mat[(i << 2) + num_dims + j]);
			__m128 t2 = _mm_loadu_ps(&coord_mat[(i << 2) + 2 * num_dims + j]);
			__m128 t3 = _mm_loadu_ps(&coord_mat[(i << 2) + 3 * num_dims + j]);

			// __m128 t4 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1, 0, 1, 0));
			// __m128 t5 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 2, 3, 2));
			// __m128 t6 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(1, 0, 1, 0));
			// __m128 t7 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3, 2, 3, 2));

			// t0 = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(2, 0, 2, 0));
			// t1 = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 1, 3, 1));
			// t2 = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(2, 0, 2, 0));
			// t3 = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 1, 3, 1));

			_MM_TRANSPOSE4_PS(t0, t1, t2, t3);			
			
			_mm_store_ps(tran_coord_mat[i + j], t0);
			_mm_store_ps(tran_coord_mat[i + j + 1], t1);
			_mm_store_ps(tran_coord_mat[i + j + 2], t2);
			_mm_store_ps(tran_coord_mat[i + j + 3], t3);			
		}
		
		for (j = (num_dims >> 2) << 2; j < num_dims; j++) {
			tran_coord_mat[i + j][0] = coord_mat[(i << 2) + j];
			tran_coord_mat[i + j][1] = coord_mat[(i << 2) + num_dims + j];
			tran_coord_mat[i + j][2] = coord_mat[(i << 2) + 2 * num_dims + j];
			tran_coord_mat[i + j][3] = coord_mat[(i << 2) + 3 * num_dims + j];
		}	
	}

	for (i = 0; i < num_levels; i++) {
		// loop over subspaces of same level
		for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
			idx2l(j, i, levels);
			
			coords = tran_coord_mat[0];

	    	for (c = 0; c < tran_n / num_dims; c++) {
	    		__m128 prod_xmm = _mm_set1_ps(1.0f);
			    __m128 index2_xmm = _mm_set1_ps(0.0f);
			    __m128 crt_index_xmm;
			    __m128i crt_index_int_xmm;
			    
				for (k = 0; k < num_dims; k++, coords += 4) {
					div = 1.0f / (1 << levels[k]);
					__m128 div_xmm = _mm_set1_ps(div);
					__m128 coord_xmm = _mm_load_ps(coords);
					
					crt_index_xmm = _mm_div_ps(coord_xmm, div_xmm);
					crt_index_int_xmm = _mm_cvttps_epi32(crt_index_xmm);
					crt_index_xmm = _mm_cvtepi32_ps(crt_index_int_xmm);

					__m128 left_xmm = _mm_mul_ps(crt_index_xmm, div_xmm);
					
					index2_xmm = _mm_add_ps(_mm_mul_ps(index2_xmm, _mm_set1_ps(1 << levels[k])), crt_index_xmm);
					
					__m128 right_xmm = _mm_add_ps(left_xmm, div_xmm);
					__m128 t0 = _mm_set1_ps(2.0f);
					__m128 t1 = _mm_set1_ps(1.0f);
					__m128 m_xmm = _mm_div_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(coord_xmm, coord_xmm), left_xmm), right_xmm), div_xmm);
					__m128 f_xmm = _mm_cmpgt_ps(m_xmm, _mm_setzero_ps());
					__m128 hat_xmm = _mm_add_ps(t1, _mm_mul_ps(m_xmm, _mm_sub_ps(t1, _mm_and_ps(f_xmm, t0))));
					
					prod_xmm = _mm_mul_ps(prod_xmm, hat_xmm);
				}
				_mm_store_ps(prod_x4, prod_xmm);
				crt_index_int_xmm = _mm_cvttps_epi32(index2_xmm);
				_mm_store_si128((__m128i *) index2_x4, crt_index_int_xmm);
				
			    prod_x4[0] *= sg1d[index2_x4[0]];
			    prod_x4[1] *= sg1d[index2_x4[1]];
			    prod_x4[2] *= sg1d[index2_x4[2]];
			    prod_x4[3] *= sg1d[index2_x4[3]];
			    
			    out[c << 2] += prod_x4[0];
			    out[(c << 2) + 1] += prod_x4[1];
			    out[(c << 2) + 2] += prod_x4[2];
			    out[(c << 2) + 3] += prod_x4[3];
	        
				// __m128 out_xmm = _mm_loadu_ps(&out[c << 2]);
				// prod_xmm = _mm_load_ps(prod_x4);
				// out_xmm = _mm_add_ps(out_xmm, prod_xmm);
				// _mm_storeu_ps(&out[c << 2], out_xmm);
			}
	        
        	for (c = (num_evals >> 2) << 2; c < num_evals; c++) {
				coords = &coord_mat[c * num_dims];
				        	
		        prod = 1.0f;
		        index2 = 0;
				for (k = 0; k < num_dims; k++) {
					div = (1.0f - 0.0f) / (1 << levels[k]);
					index2 = index2 * (1 << levels[k]) + (int) ((coords[k] - 0.0f) / div);
					left = (int) ((coords[k] - 0.0f) / div) * div;
					m = (2.0f * (coords[k] - left) - div) / div;
					prod *= 1.0f + m * (1.0f - ((m > 0.0f) << 1));
				}

		        prod *= sg1d[index2];
		        out[c] += prod;
            }
	        
			sg1d += 1 << i;
	    }
	}
	
	return 0;
}

// evaluates sparse grid at given points
// has strength reduction = no divisions in the innermost loop (opt3)
// opt1 + opt3
int evaluate3(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, i, j, levels[num_dims], idx01, idx2, t, c;
	float left, prod, m, divs[num_dims], inv_divs[num_dims];

	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;

	idx01 = 0;
	// loop over sets of subspaces of different levels
	for (i = 0; i < num_levels; i++) {
		// loop over subspaces of same level
		for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
			idx2l(j, i, levels);
			
			// for loop invariant code motion
			for (k = 0; k < num_dims; k++) {
				divs[k] = 1.0f / (1 << levels[k]);
				// for strength reduction
				inv_divs[k] = 1 << levels[k];
			}

			// loop over interpolation points
			for (c = 0; c < num_evals; c++) {
				prod = 1.0f;
				idx2 = 0;
				for (k = 0; k < num_dims; k++) {
					// strength reduction by multiplying with inverse
					t = (int) (coord_mat[c * num_dims + k] * inv_divs[k]);
					// extra ILP here from regular reduction
					idx2 = idx2 * (1 << levels[k]) + t;
										
					left = t * divs[k];
					// m = ((coord_mat[c * num_dims + k] - left) + (coord_mat[c * num_dims + k] - left) - divs[k]) * inv_divs[k];
					m = (coord_mat[c * num_dims + k] - left) * (inv_divs[k] + inv_divs[k]) - 1.0f;
					prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
				}

				prod *= sg1d[idx01 + idx2];
				out[c] += prod;
			}

			idx01 += 1 << i;
		}
	}
	
	return 0;
}

// evaluates sparse grid at given points
// loop invariant code motion = prefix sums => less dependencies in innermost loop (opt4)
// after this, the innermost loop has only reductions (+ and *) as dependencies between iterations
// opt1 + opt3 + opt4
int evaluate4(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, i, j, levels[num_dims], idx01, idx2, t, c, prefix_sums[num_dims + 1];
	float left, prod, m, divs[num_dims], inv_divs[num_dims];

	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;

	idx01 = 0;
	// loop over sets of subspaces of different levels
	for (i = 0; i < num_levels; i++) {
		// loop over subspaces of same level
		for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
			idx2l(j, i, levels);
			
			// for extra ILP
			prefix_sums[num_dims] = 0;
			prefix_sums[num_dims - 1] = levels[num_dims - 1];
			for (k = num_dims - 2; k >= 0; k--)
				prefix_sums[k] = prefix_sums[k + 1] + levels[k];

			// for loop invariant code motion
			for (k = 0; k < num_dims; k++) {
				divs[k] = 1.0f / (1 << levels[k]);
				// for strength reduction
				inv_divs[k] = 1 << levels[k];
			}

			// loop over interpolation points
			for (c = 0; c < num_evals; c++) {
				prod = 1.0f;
				idx2 = 0;
				for (k = 0; k < num_dims; k++) {
					// strength reduction by multiplying with inverse
					t = (int) (coord_mat[c * num_dims + k] * inv_divs[k]);
					// extra ILP here from regular reduction
					idx2 += t << prefix_sums[k + 1];
					left = t * divs[k];
					// m = ((coord_mat[c * num_dims + k] - left) + (coord_mat[c * num_dims + k] - left) - divs[k]) * inv_divs[k];
					m = (coord_mat[c * num_dims + k] - left) * (inv_divs[k] + inv_divs[k]) - 1.0f;
					prod *= 1.0f + m * (1.0f - ((m >= 0.0f) << 1));
				}

				prod *= sg1d[idx01 + idx2];
				out[c] += prod;
			}

			idx01 += 1 << i;
		}
	}
	
	return 0;
}

// evaluates sparse grid at given points
// opt1 + opt2 + opt3 + opt4
int evaluate5(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	int k, j, i, levels[num_dims], index2, c, tran_n;
	float left, prod, div, *coords, m;
	float tran_coord_mat[num_dims * (num_evals >> 2)][4] __attribute__((aligned(16)));
	float prod_x4[4] __attribute__((aligned(16)));
	int index2_x4[4] __attribute__((aligned(16)));
	float divs[num_dims], inv_divs[num_dims];
	
	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0.0f;
		
	tran_n = num_dims * (num_evals >> 2);
	for (i = 0; i < tran_n; i += num_dims) {
		for (j = 0; j < (num_dims >> 2) << 2; j += 4) {
			__m128 t0 = _mm_loadu_ps(&coord_mat[(i << 2) + j]);
			__m128 t1 = _mm_loadu_ps(&coord_mat[(i << 2) + num_dims + j]);
			__m128 t2 = _mm_loadu_ps(&coord_mat[(i << 2) + 2 * num_dims + j]);
			__m128 t3 = _mm_loadu_ps(&coord_mat[(i << 2) + 3 * num_dims + j]);

			// __m128 t4 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1, 0, 1, 0));
			// __m128 t5 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 2, 3, 2));
			// __m128 t6 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(1, 0, 1, 0));
			// __m128 t7 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3, 2, 3, 2));

			// t0 = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(2, 0, 2, 0));
			// t1 = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 1, 3, 1));
			// t2 = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(2, 0, 2, 0));
			// t3 = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 1, 3, 1));

			_MM_TRANSPOSE4_PS(t0, t1, t2, t3);			
			
			_mm_store_ps(tran_coord_mat[i + j], t0);
			_mm_store_ps(tran_coord_mat[i + j + 1], t1);
			_mm_store_ps(tran_coord_mat[i + j + 2], t2);
			_mm_store_ps(tran_coord_mat[i + j + 3], t3);			
		}
		
		for (j = (num_dims >> 2) << 2; j < num_dims; j++) {
			tran_coord_mat[i + j][0] = coord_mat[(i << 2) + j];
			tran_coord_mat[i + j][1] = coord_mat[(i << 2) + num_dims + j];
			tran_coord_mat[i + j][2] = coord_mat[(i << 2) + 2 * num_dims + j];
			tran_coord_mat[i + j][3] = coord_mat[(i << 2) + 3 * num_dims + j];
		}	
	}

	for (i = 0; i < num_levels; i++) {
		// loop over subspaces of same level
		for (j = 0; j < dp_mat[num_dims - 1][i]; j++) {
			idx2l(j, i, levels);
			
			coords = tran_coord_mat[0];

			// for loop invariant code motion
			for (k = 0; k < num_dims; k++) {
				divs[k] = 1.0f / (1 << levels[k]);
				// for strength reduction
				inv_divs[k] = 1 << levels[k];
			}

	    	for (c = 0; c < tran_n / num_dims; c++) {
	    		__m128 prod_xmm = _mm_set1_ps(1.0f);
			    __m128 index2_xmm = _mm_set1_ps(0.0f);
			    __m128 crt_index_xmm;
			    __m128i crt_index_int_xmm;
			    
				for (k = 0; k < num_dims; k++, coords += 4) {
					__m128 div_xmm = _mm_set1_ps(divs[k]);
					__m128 inv_div_xmm = _mm_set1_ps(inv_divs[k]);
					__m128 coord_xmm = _mm_load_ps(coords);
					
					crt_index_xmm = _mm_mul_ps(coord_xmm, inv_div_xmm);
					crt_index_int_xmm = _mm_cvttps_epi32(crt_index_xmm);
					crt_index_xmm = _mm_cvtepi32_ps(crt_index_int_xmm);

					__m128 left_xmm = _mm_mul_ps(crt_index_xmm, div_xmm);
					
					index2_xmm = _mm_add_ps(_mm_mul_ps(index2_xmm, _mm_set1_ps(1 << levels[k])), crt_index_xmm);
					
					__m128 right_xmm = _mm_add_ps(left_xmm, div_xmm);
					__m128 t0 = _mm_set1_ps(2.0f);
					__m128 t1 = _mm_set1_ps(1.0f);
					__m128 m_xmm = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(coord_xmm, coord_xmm), left_xmm), right_xmm), inv_div_xmm);
					__m128 f_xmm = _mm_cmpgt_ps(m_xmm, _mm_setzero_ps());
					__m128 hat_xmm = _mm_add_ps(t1, _mm_mul_ps(m_xmm, _mm_sub_ps(t1, _mm_and_ps(f_xmm, t0))));
					
					prod_xmm = _mm_mul_ps(prod_xmm, hat_xmm);
				}
				_mm_store_ps(prod_x4, prod_xmm);
				crt_index_int_xmm = _mm_cvttps_epi32(index2_xmm);
				_mm_store_si128((__m128i *) index2_x4, crt_index_int_xmm);
				
			    prod_x4[0] *= sg1d[index2_x4[0]];
			    prod_x4[1] *= sg1d[index2_x4[1]];
			    prod_x4[2] *= sg1d[index2_x4[2]];
			    prod_x4[3] *= sg1d[index2_x4[3]];
			    
			    out[c << 2] += prod_x4[0];
			    out[(c << 2) + 1] += prod_x4[1];
			    out[(c << 2) + 2] += prod_x4[2];
			    out[(c << 2) + 3] += prod_x4[3];
	        
				// __m128 out_xmm = _mm_loadu_ps(&out[c << 2]);
				// prod_xmm = _mm_load_ps(prod_x4);
				// out_xmm = _mm_add_ps(out_xmm, prod_xmm);
				// _mm_storeu_ps(&out[c << 2], out_xmm);
			}
	        
        	for (c = (num_evals >> 2) << 2; c < num_evals; c++) {
				coords = &coord_mat[c * num_dims];
				        	
		        prod = 1.0f;
		        index2 = 0;
				for (k = 0; k < num_dims; k++) {
					index2 = index2 * (1 << levels[k]) + (int) ((coords[k] - 0.0f) * inv_divs[k]);
					left = (int) ((coords[k] - 0.0f) * inv_divs[k]) * divs[k];
					m = (2.0f * (coords[k] - left) - divs[k]) * inv_divs[k];
					prod *= 1.0f + m * (1.0f - ((m > 0.0f) << 1));
				}

		        prod *= sg1d[index2];
		        out[c] += prod;
            }
	        
			sg1d += 1 << i;
	    }
	}
	
	return 0;
}

// opt1 + opt2
int omp_evaluate2(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		
		int net = num_evals / num_threads + ((tid < num_evals % num_threads)? 1: 0);
		int tindex = tid * (num_evals / num_threads) + ((tid < num_evals % num_threads)? tid: num_evals % num_threads);
		
		evaluate2(sg1d, coord_mat + tindex * num_dims, out + tindex, net);
	}

	return 0;
}

// opt1 + opt2 + opt3
int omp_evaluate5(float *sg1d, float *coord_mat, float *out, int num_evals)
{
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		
		int net = num_evals / num_threads + ((tid < num_evals % num_threads)? 1: 0);
		int tindex = tid * (num_evals / num_threads) + ((tid < num_evals % num_threads)? tid: num_evals % num_threads);
		
		evaluate5(sg1d, coord_mat + tindex * num_dims, out + tindex, net);
	}

	return 0;
}

void generate_rand_points(int num, float *coord_mat)
{
	int i, j;
	int levels[num_dims], indices[num_dims];
	int rand_idx;
	
	for (i = 0; i < num; i++) {
		rand_idx = random() % num_grid_points;
		idx2gp(rand_idx, levels, indices);
		for (j = 0; j < num_dims; j++)
			coord_mat[i * num_dims + j] = indices[j] * (1.0f - 0.0f) / (1 << levels[j]) + (1.0f - 0.0f) / (1 << (levels[j] + 1));
	}
}


//=============================================================
//    main
//=============================================================

int main(int argc, char **argv)
{
	char hostname[256];
	double hi_flops, ev_flops;
	double et;
	float *coord_mat, *ref_sg1d, *f1d, *ref_out, *out;
	int i, j, num_procs;

	if (argc != 4) {
		printf("Usage: sparse_grid_bench <num. dimensions> <refinement level> <num. evals>\n");
		return -1;
	} else {
		num_dims = atoi(argv[1]);
		num_levels = atoi(argv[2]);
		num_evals = atoi(argv[3]);
	}
	
	limits = (int *) malloc(num_dims * sizeof(int));
	
	for (i = 0; i < num_dims; i++)
		limits[i] = num_levels - 1;
		
	num_procs = omp_get_num_procs();

	init();

	coord_mat = (float *) malloc(num_procs * num_evals * num_dims * sizeof(float));
	ref_out = (float *) malloc(num_evals * sizeof(float));
	out = (float *) malloc(num_procs * num_evals * sizeof(float));
	f1d = (float *) malloc(num_grid_points * sizeof(float));
	ref_sg1d = (float *) malloc(num_grid_points * sizeof(float));
	
	memcpy(f1d, sg1d, num_grid_points * sizeof(float));
	
	generate_rand_points(num_procs * num_evals, coord_mat);
	
	// info
	gethostname(hostname, 256);
	printf("# host name: %s\n", hostname);
	printf("# num_levels: %d, num_dims: %d, num_evals: %d\n", num_levels, num_dims, num_evals);
	printf("# num. of gridpoints: %d\n", num_grid_points);
	hi_flops = HI_FLOPS(num_dims, num_grid_points);
	printf("# num. of floating point ops. hierarchization: %.10lf GFlop\n", hi_flops);
	ev_flops = EV_FLOPS(num_dims, num_subspaces, num_evals);
	printf("# num. of floating point ops. evaluation: %.10lf GFlop\n", ev_flops);
	printf("\n");

	// execution time and gflops hierarchization
	et = get_time();
	hierarchize0();
	et = get_time() - et;
	printf("exec. time hierarchization v0: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v0: %.10lf\n", hi_flops / et);
	printf("\n");
	
	memcpy(ref_sg1d, sg1d, num_grid_points * sizeof(float));

	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));
	
	et = get_time();
	hierarchize1();
	et = get_time() - et;
	printf("exec. time hierarchization v1: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v1: %.10lf\n", hi_flops / et);
	printf("\n");
		
	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}
		
	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

	et = get_time();
	hierarchize2();
	et = get_time() - et;
	printf("exec. time hierarchization v2: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v2: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}
	
	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

	et = get_time();
	hierarchize3();
	et = get_time() - et;
	printf("exec. time hierarchization v3: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v3: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}
	
	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

	et = get_time();
	hierarchize4();
	et = get_time() - et;
	printf("exec. time hierarchization v4: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v4: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}
	
	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

	et = get_time();
	hierarchize5();
	et = get_time() - et;
	printf("exec. time hierarchization v5: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v5: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}

	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

	et = get_time();
	hierarchize6();
	et = get_time() - et;
	printf("exec. time hierarchization v6: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v6: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}
	
	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));	
	
	et = get_time();
	hierarchize7();
	et = get_time() - et;
	printf("exec. time hierarchization v7: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v7: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}	

	// execution time and gflops hierarchization
	memcpy(sg1d, f1d, num_grid_points * sizeof(float));		
	
	et = get_time();
	hierarchize8();
	et = get_time() - et;
	printf("exec. time hierarchization v8: %.10lf\n", et);
	printf("GFLOPS rate hierarchization v8: %.10lf\n", hi_flops / et);
	printf("\n");

	for (i = 0; i < num_grid_points; i++) {
		assert(sg1d[i] == ref_sg1d[i]);
	}		

	// for (i = 1; i <= num_procs; i++) {
		// omp_set_num_threads(i);
		
		// execution time and gflops hierarchization
		memcpy(sg1d, f1d, num_grid_points * sizeof(float));	

		et = get_time();
		omp_hierarchize3();
		et = get_time() - et;
		// printf("num. threads: %d\n", i);
		printf("exec. time hierarchization v3_omp: %.10lf\n", et);
		printf("GFLOPS rate hierarchization v3_omp: %.10lf\n", hi_flops / et);
		printf("\n");

		for (j = 0; j < num_grid_points; j++) {
			assert(sg1d[j] == ref_sg1d[j]);
		}
	// }
	
	// execution time and gflops evaluation
	et = get_time();
	evaluate0(sg1d, coord_mat, ref_out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v0: %.10lf\n", et);
	printf("GFLOPS rate evaluation v0: %.10lf\n", ev_flops / et);
	printf("\n");

	et = get_time();
	evaluate1(sg1d, coord_mat, out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v1: %.10lf\n", et);
	printf("GFLOPS rate evaluation v1: %.10lf\n", ev_flops / et);
	printf("\n");

	for (i = 0; i < num_evals; i++) {
		assert(out[i] == ref_out[i]);
	}

	et = get_time();
	evaluate2(sg1d, coord_mat, out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v2: %.10lf\n", et);
	printf("GFLOPS rate evaluation v2: %.10lf\n", ev_flops / et);
	printf("\n");

	for (i = 0; i < num_evals; i++) {
		assert(out[i] == ref_out[i]);
	}

	et = get_time();
	evaluate3(sg1d, coord_mat, out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v3: %.10lf\n", et);
	printf("GFLOPS rate evaluation v3: %.10lf\n", ev_flops / et);
	printf("\n");

	for (i = 0; i < num_evals; i++) {
		assert(out[i] == ref_out[i]);
	}	
	
	et = get_time();
	evaluate4(sg1d, coord_mat, out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v4: %.10lf\n", et);
	printf("GFLOPS rate evaluation v4: %.10lf\n", ev_flops / et);
	printf("\n");

	for (i = 0; i < num_evals; i++) {
		assert(out[i] == ref_out[i]);
	}	

	et = get_time();
	evaluate5(sg1d, coord_mat, out, num_evals);
	et = get_time() - et;
	printf("exec. time evaluation v5: %.10lf\n", et);
	printf("GFLOPS rate evaluation v5: %.10lf\n", ev_flops / et);
	printf("\n");

	for (i = 0; i < num_evals; i++) {
		assert(out[i] == ref_out[i]);
	}	

	// for (i = 1; i <= num_procs; i++) {
		// omp_set_num_threads(i);
		
		et = get_time();
		omp_evaluate2(sg1d, coord_mat, out, num_evals);
		et = get_time() - et;
		// printf("num. threads: %d\n", i);
		printf("exec. time evaluation v2_omp: %.10lf\n", et);
		printf("GFLOPS rate evaluation v2_omp: %.10lf\n", ev_flops / et);
		printf("\n");

		for (j = 0; j < num_evals; j++) {
			assert(out[j] == ref_out[j]);
		}
	// }
	
	// for (i = 1; i <= num_procs; i++) {
		// omp_set_num_threads(i);
		
		et = get_time();
		omp_evaluate5(sg1d, coord_mat, out, num_evals);
		et = get_time() - et;
		// printf("num. threads: %d\n", i);
		printf("exec. time evaluation v5_omp: %.10lf\n", et);
		printf("GFLOPS rate evaluation v5_omp: %.10lf\n", ev_flops / et);
		printf("\n");

		for (j = 0; j < num_evals; j++) {
			assert(out[j] == ref_out[j]);
		}
	// }

	// corectness test
	for (i = 0; i < num_evals; i++) {
		assert(fabs(out[i] - fct(&coord_mat[i * num_dims])) < 0.000001f);
	}
	printf("Correctness test passed\n");
		
	return 0;
}

