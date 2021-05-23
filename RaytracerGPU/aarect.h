#ifndef AARECT_H
#define AARECT_H

#include "hitable.h"
#include "ray.h"

class xy_rect : public hitable {
public:
    __device__ xy_rect() {}

    __device__ xy_rect(double _x0, double _x1, double _y0, double _y1, double _k,
        material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
        return true;
    }

public:
    material* mp;
    double x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    auto outward_normal = vec3(0, 0, 1);
    rec.setFaceNormal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}

#endif
