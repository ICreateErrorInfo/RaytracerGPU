#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hitable.h"
#include "material.h"
#include "texture.h"
#include "ray.h"
#include "material.h"
#include "vec3.h"

class constant_medium : public hitable {
public:
    __device__ constant_medium(hitable* b, float f, texture1* a, curandState* local_rand_state) : boundary(b), neg_inv_density(-1 / f), rand_state(local_rand_state) {
        phase_function = new isotropic(a);
    }

    __device__ constant_medium(hitable* b, double d, vec3 c, curandState* local_rand_state)
        : boundary(b),
        neg_inv_density(-1 / d),
        rand_state(local_rand_state)
    {
        phase_function = new isotropic(c);
    }

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
        return boundary->bounding_box(time0, time1, output_box);
    }

public:
    hitable* boundary;
    material* phase_function;
    double neg_inv_density;
    curandState* rand_state;
};

__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && random_double(rand_state, -1, 1) < 0.00001;

    hit_record rec1, rec2;

    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2))
        return false;

    if (debugging) printf("error");

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = r.direction().length();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * log(curand_uniform(rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.point_at_parameter(rec.t);

    if (debugging) {
        printf("error");
    }

    rec.normal = vec3(1, 0, 0);  // arbitrary
    rec.frontface = true;     // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}

#endif