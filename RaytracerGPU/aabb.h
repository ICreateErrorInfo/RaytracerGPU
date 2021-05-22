#ifndef AABB_H
#define AABB_H

#include "ray.h"
#include "vec3.h"
#include <curand_kernel.h>

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(vec3 a, vec3 b) { minimum = a; maximum = b; }

    __device__ vec3 min() const { return minimum; }
    __device__ vec3 max() const { return maximum; }

    __device__ bool hit(ray r, double t_min, double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.direction()[a];
            auto t0 = (min()[a] - r.origin()[a]) * invD;
            auto t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    __device__ static aabb surrounding_box(aabb box0, aabb box1) {
        vec3 small(fmin(box0.min().x(), box1.min().x()),
            fmin(box0.min().y(), box1.min().y()),
            fmin(box0.min().z(), box1.min().z()));

        vec3 big(fmax(box0.max().x(), box1.max().x()),
            fmax(box0.max().y(), box1.max().y()),
            fmax(box0.max().z(), box1.max().z()));

        return aabb(small, big);
    }

    vec3 minimum;
    vec3 maximum;
};

#endif