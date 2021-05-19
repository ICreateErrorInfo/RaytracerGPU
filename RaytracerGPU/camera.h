#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"
#include <corecrt_math_defines.h>
#include <random>

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ double random_double(curandState* local_rand_state, double min, double max) {
        
    double random = curand_uniform(local_rand_state);
    return min + (max - min) * random;
}

class camera {
public:
    __device__ camera(
        vec3 lookfrom,
        vec3 lookat,
        vec3 vup,
        float vfov,
        float aspect,
        float aperture,
        float focus_dist,
        double _time0 = 0,
        double _time1 = 0
    ) { 

        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;

        lens_radius = aperture / 2.0f;
        time0 = _time0;
        time1 = _time1;
    }
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_double(local_rand_state, time0, time1));
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    double time0, time1;
};

#endif