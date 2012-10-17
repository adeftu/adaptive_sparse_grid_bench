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


//=============================================================
//    utils
//=============================================================

#define HI_FLOPS(num_dims, num_grid_points) \
	(((num_dims) * 3 * (num_grid_points)) / 1000000000.0)

#define EV_FLOPS(num_dims, num_levels, num_evals) \
	(((num_evals) * (double) combi((num_levels) + (num_dims) - 1, (num_levels) - 1) * (10 * (num_dims) + 2)) / 1000000000.0)

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
int num_levels, num_dims, num_evals;
int num_grid_points;
int **bjmp_mat, **jmp_mat, *ljmp_vect;
float *sg1d;

float fct(float *coords)
{
	int i;
	float res = 0;
	for (i = 0; i < num_dims; i++)
		res += coords[i] * coords[i] * (1.0f - coords[i]) * (1.0f - coords[i]);
	return res;
}

int idx2gp(int, int *, int *);

// initialization function
void init()
{
	int i, j, f;
	int levels[num_dims], indices[num_dims];
	float coords[num_dims];
	
	// allocate memory
	bjmp_mat = (int **) malloc((num_levels + 1) * sizeof(int *));
	for (i = 0; i < num_levels + 1; i++)
		bjmp_mat[i] = (int *) malloc(num_dims * sizeof(int));
	ljmp_vect = (int *) malloc(num_levels * sizeof(int));

	// initialize
	for (i = 0; i < num_dims; i++)
		bjmp_mat[0][i] = 0;
	
	jmp_mat = &bjmp_mat[1];

	// compute binomial coefficients
	for (i = 0; i < num_dims; i++)
		jmp_mat[0][i] = 1;
	for (i = 1; i < num_levels; i++) {
		jmp_mat[i][0] = 1;
		for (j = 1; j < num_dims; j++)
			jmp_mat[i][j] = jmp_mat[i - 1][j] + jmp_mat[i][j - 1];
	}
	
	// optimization (memoization)
	num_grid_points = 0;
	f = 1;
	for (i = 0; i < num_levels; i++) {
		ljmp_vect[i] = num_grid_points;
		num_grid_points += jmp_mat[i][num_dims - 1] * f;
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
}

// conversion function: grid point to index
int gp2idx(int *levels, int *indices)
{
	int index1, index2, index3, i, sum;

	sum = 0;
	index2 = 0;
	
	index1 = indices[0];
	for (i = 1; i < num_dims; i++)
		index1 = (index1 << levels[i]) + indices[i];

	for (i = 0; i < num_dims - 1; i++) {
		sum += levels[i];
		index2 += bjmp_mat[sum][i + 1];
	}
	sum += levels[i];
	index2 <<= sum;

	index3 = ljmp_vect[sum];
	
	return index1 + index2 + index3;	
}

// conversion function: index to grid point
int idx2gp(int index, int *levels, int *indices)
{
	int i, j, isum, sum, level, dindex, rest;

	isum = 0;
	i = 0;
	while (index >= isum + (jmp_mat[i][num_dims - 1] << i)) {
		isum += jmp_mat[i][num_dims - 1] << i;
		i++;
	}

	sum = i;
	index -= isum;
	rest = index & ((1 << i) - 1);
	index >>= i;

	for (i = num_dims - 2; i >= 0; i--) {
		isum = 0;
		j = 0;
		while (index >= isum + jmp_mat[j][i]) {
			isum += jmp_mat[j][i];
			j++;
		}
		level = sum - j;
		sum = j;
		dindex = rest & ((1 << level) - 1);
		rest >>= level;
		levels[i + 1] = level;
		indices[i + 1] = dindex;
		index -= isum;
	}

	level = sum;
	dindex = rest % (1 << level);
	levels[0] = level;
	indices[0] = dindex;

	return 0;
}

// gets value of left parent in dimension crt_dim
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

// gets value of right parent in dimension crt_dim
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

// computes hierarchical coefficients (no optimizations)
int hierarchize_basic()
{
	int i, j, levels[num_dims], indices[num_dims];

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		for (j = num_grid_points - 1; j >= 0; j--) {
			idx2gp(j, levels, indices);
			sg1d[j] -= (getLeftParentVal(levels, indices, i) + getRightParentVal(levels, indices, i)) * 0.5f;
		}
	}
	
	return 0;
}

// computes hierarchical coefficients
int hierarchize()
{
	int i, j, k, l, m, levels[num_dims], indices[num_dims], index, t0;

	for (i = 0; i < num_dims; i++)
		levels[i] = 0;

	// loop over dimensions
	for (i = 0; i < num_dims; i++) {
		index = num_grid_points - 1;
		
		// loop over sets of subspaces (bottom - up)
		for (j = num_levels - 1; j >= 0; j--) {
			levels[num_dims - 1] = 0;
			levels[0] = j;
	
			// loop over subspaces of same level
			for (k = 0; k < jmp_mat[j][num_dims - 1]; k++) {
				for (m = num_dims - 1; m >= 0; m--)
					indices[m] = 0;

				// loop over points in subspace
				for (l = (1 << j) - 1; l >= 0; l--) {
					sg1d[index] -= (getLeftParentVal(levels, indices, i) + 
					                getRightParentVal(levels, indices, i)) * 0.5f;
					index--;
					
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

				// compute levels of next subspace				
				for (l = 0; l < num_dims - 1; l++) {
					if (levels[l] > 0) {
						levels[l + 1]++;
						t0 = levels[l];
						levels[l] = 0;
						levels[0] = t0 - 1;
						break;
					}
				}
			}
		}
	}
	
	return 0;
}

// evaluates sparse grid at given points
int evaluate(float *coord_mat, float *out)
{
	int k, i, levels[num_dims], index2, t0, c;
	float left, prod, div, *coords, m;

	// initialize
	for (i = 0; i < num_evals; i++)
		out[i] = 0;
	for (i = 0; i < num_levels; i++)
		levels[i] = 0;
		
	// loop over sets of subspaces of different levels
	for (i = 0; i < num_levels; i++) {
		levels[0] = 0;
		levels[num_dims - 1] = i;
		// loop over subspaces of same level
		do {
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

			if (levels[0] == i)
				break;

			k = 1;
			while (levels[k] == 0)
				k++;
			levels[k]--;
			t0 = levels[0];
			levels[0] = 0;
			levels[k - 1] = t0 + 1;
		} while (1);
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
	float *coord_mat, *out;
	int i;

	if (argc != 4) {
		printf("Usage: sparse_grid_bench <num. dimensions> <refinement level> <num. evals>\n");
		return -1;
	} else {
		num_dims = atoi(argv[1]);
		num_levels = atoi(argv[2]);
		num_evals = atoi(argv[3]);
	}
	
	init();

	coord_mat = (float *) malloc(num_evals * num_dims * sizeof(float));
	out = (float *) malloc(num_evals * sizeof(float));
	generate_rand_points(num_evals, coord_mat);
	
	// info
	gethostname(hostname, 256);
	printf("# host name: %s\n", hostname);
	printf("# num_levels: %d, num_dims: %d, num_evals: %d\n", num_levels, num_dims, num_evals);
	printf("# num. of gridpoints: %d\n", num_grid_points);
	hi_flops = HI_FLOPS(num_dims, num_grid_points);
	printf("# num. of floating point ops. hierarchization: %.10lf GFlop\n", hi_flops);
	ev_flops = EV_FLOPS(num_dims, num_levels, num_evals);
	printf("# num. of floating point ops. evaluation: %.10lf GFlop\n", ev_flops);
	printf("\n");

	// execution time and gflops hierarchization
	et = get_time();
	hierarchize();
	et = get_time() - et;
	printf("exec. time hierarchization: %.10lf\n", et);
	printf("GFLOPS rate hierarchization: %.10lf\n", hi_flops / et);
	printf("\n");
	
	// execution time and gflops evaluation
	et = get_time();
	evaluate(coord_mat, out);
	et = get_time() - et;
	printf("exec. time evaluation: %.10lf\n", et);
	printf("GFLOPS rate evaluation: %.10lf\n", ev_flops / et);
	printf("\n");
	
	// corectness test
	for (i = 0; i < num_evals; i++)
		assert(fabs(out[i] - fct(&coord_mat[i * num_dims])) < 0.000001f);
	printf("Correctness test passed\n");
		
	return 0;
}

