#ifndef PERLIN_H
#define PERLIN_H

#include "vec3.h"
#include <curand_kernel.h>
#include "curand.h"
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h> 

class perlin {
public:
    __device__ perlin(){}

     __device__ perlin(curandState* d_rand_state) {

         ranfloat = new double[point_count];
         for (int i = 0; i < point_count; ++i) {
             ranfloat[i] = random_double(d_rand_state, 0, 1);
         }

        perm_x = perlin_generate_perm(d_rand_state);
        perm_y = perlin_generate_perm(d_rand_state);
        perm_z = perlin_generate_perm(d_rand_state);

        printf("%d \n", perm_x[6]);
        printf("init \n");
       
    }

    __device__ ~perlin() {
        //delete[] ranfloat;
        //delete[] perm_x;
        //delete[] perm_y;
        //delete[] perm_z;
    }

    __device__ double noise(const vec3& p) const {
        auto u = (double)p.x() - floor((double)p.x());
        auto v = (double)p.y() - floor((double)p.y());
        auto w = (double)p.z() - floor((double)p.z());

        int i = floor((double)p.x());
        int j = floor((double)p.y());
        int k = floor((double)p.z());
        double c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++) {

                    int erg  = perm_x[(i + di) & 255];
                    int erg1 = perm_y[(j + dj) & 255];
                    int erg2 = perm_z[(k + dk) & 255];

                    int test = 10 ^ 47 ^ 213;
                    int test2 = erg ^ erg1 ^ erg2;

                    c[di][dj][dk] = ranfloat[test2];
                }

            return trilinear_interp(c, u, v, w);
    }

private:
    static const int point_count = 256;
    double* ranfloat;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    __device__ static int* perlin_generate_perm(curandState* d_rand_state) {
        auto p = new int[point_count];

        for (int i = 0; i < point_count; i++) {
            p[i] = i;
        }


        permute(p, point_count, d_rand_state);

        return p;
    }

    __device__ static void permute(int* p, int n, curandState* d_rand_state) {

        for (int i = n - 1; i > 0; i--) {
            int target = (int)random_double(d_rand_state, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ static double trilinear_interp(double c[2][2][2], double u, double v, double w) {
        auto accum = 0.0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    accum += (i * u + (1 - i) * (1 - u)) *
                        (j * v + (1 - j) * (1 - v)) *
                        (k * w + (1 - k) * (1 - w)) * c[i][j][k];
                }

            return accum; 
    }
};

#endif
