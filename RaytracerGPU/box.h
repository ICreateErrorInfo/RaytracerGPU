#ifndef BOX_H
#define BOX_H

#include "aarect.h"
#include "hitable_list.h"
#include "ray.h"
#include "vec3.h"

class box : public hitable {
public:
    __device__ box() {}
    __device__ box(const vec3& p0, const vec3& p1, material* ptr, hitable** sides, int anfang, int angle, vec3 trans);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

public:
    vec3 box_min;
    vec3 box_max;
    hitable_list* sideList;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* ptr, hitable** sides, int anfang, int angle, vec3 trans) {
    box_min = p0;
    box_max = p1;

    anfang += 1;

    sides[anfang]     = new translate(new rotate_y(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr), angle), trans);
    sides[anfang + 1] = new translate(new rotate_y(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr), angle), trans);
               
    sides[anfang + 2] = new translate(new rotate_y(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr), angle), trans);
    sides[anfang + 3] = new translate(new rotate_y(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr), angle), trans);
               
    sides[anfang + 4] = new translate(new rotate_y(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr), angle), trans);
    sides[anfang + 5] = new translate(new rotate_y(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr), angle), trans);

    sideList = new hitable_list(sides, 6);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return sideList->hit(r, t_min, t_max, rec);
}

#endif