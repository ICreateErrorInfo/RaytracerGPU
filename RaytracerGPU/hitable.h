#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
    bool frontface;

    __device__ void setFaceNormal(ray r, vec3 outward_normal) {
        frontface = dot(r.direction(), outward_normal) < 0;
        normal = frontface ? outward_normal : -outward_normal;
    }
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif