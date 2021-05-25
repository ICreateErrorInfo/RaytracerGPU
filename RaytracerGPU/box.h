#ifndef BOX_H
#define BOX_H

#include "aarect.h"
#include "hitable_list.h"
#include "ray.h"
#include "vec3.h"
#include "constant_medium.h"

class Box : public hitable {
public:
    __device__ Box() {}
    __device__ Box(const vec3& p0, const vec3& p1, material* ptr);

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;

    __device__ virtual bool bounding_box(double t0, double t1, aabb& output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

    vec3 box_min;
    vec3 box_max;
    hitable_list sides;
};

__device__ Box::Box(const vec3& p0, const vec3& p1, material* ptr) {
    box_min = p0;
    box_max = p1;

    sides.add(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(new FlipFace(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr)));
    sides.add(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(new FlipFace(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr)));
    sides.add(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(new FlipFace(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr)));
}

__device__ bool Box::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    return sides.hit(r, t0, t1, rec);
}

#endif