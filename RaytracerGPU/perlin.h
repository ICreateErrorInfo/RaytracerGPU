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

         ranvec = new vec3[point_count];
         for (int i = 0; i < point_count; ++i) {
             ranvec[i] = vec3(random_double(d_rand_state, -1, 1), random_double(d_rand_state, -1, 1), random_double(d_rand_state, -1, 1));
         }

        perm_x = perlin_generate_perm(d_rand_state);
        perm_y = perlin_generate_perm(d_rand_state);
        perm_z = perlin_generate_perm(d_rand_state);

        printf("%d \n", perm_x[6]);
        printf("init \n");
       
    }

    __device__ ~perlin() {
        //delete[] ranvec;
        //delete[] perm_x;
        //delete[] perm_y;
        //delete[] perm_z;
    }

    __device__ double noise(const vec3& p) const {
        auto u = p.x() - floor((double)p.x());
        auto v = p.y() - floor((double)p.y());
        auto w = p.z() - floor((double)p.z());
        auto i = static_cast<int>(floor((double)p.x()));
        auto j = static_cast<int>(floor((double)p.y()));
        auto k = static_cast<int>(floor((double)p.z()));
        vec3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = ranvec[
                        perm_x[(i + di) & 255] ^
                            perm_y[(j + dj) & 255] ^
                            perm_z[(k + dk) & 255]
                    ];

        return perlin_interp(c, u, v, w);
    }

private:
    static const int point_count = 256;
    vec3* ranvec;
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

    __device__ static double perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
        auto uu = u * u * (3 - 2 * u);
        auto vv = v * v * (3 - 2 * v);
        auto ww = w * w * (3 - 2 * w);
        auto accum = 0.0;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                        * (j * vv + (1 - j) * (1 - vv))
                        * (k * ww + (1 - k) * (1 - ww))
                        * dot(c[i][j][k], weight_v);
                }

        return accum;
    }
};

#endif
