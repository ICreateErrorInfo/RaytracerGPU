#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H


#include "hitable.h"
#include "ray.h"
#include "aabb.h"

class moving_sphere : public hitable {
public:
	__device__ moving_sphere() {};
	__device__ moving_sphere(
		vec3 cen0, vec3 cen1, float _time0, float _time1, float r, material* m)
		: center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), matptr(m)
	{};

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ vec3 center(float time) const;

    __device__ virtual bool bounding_box(
        double _time0, double _time1, aabb& output_box) const override;

	vec3 center0, center1;
	float time0, time1;
	float radius;
	material* matptr;
};

__device__ vec3 moving_sphere::center(float time) const {
	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}
__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    auto a = r.direction().squared_length();
    auto half_b = dot(oc, r.direction());
    auto c = oc.squared_length() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.point_at_parameter(rec.t);
    auto outward_normal = (rec.p - center(r.time())) / radius;
    rec.setFaceNormal(r, outward_normal);
    rec.mat_ptr = matptr;

    return true;
}
__device__ bool moving_sphere::bounding_box(double _time0, double _time1, aabb& output_box) const {
    aabb box0(
        center(_time0) - vec3(radius, radius, radius),
        center(_time0) + vec3(radius, radius, radius));
    aabb box1(
        center(_time1) - vec3(radius, radius, radius),
        center(_time1) + vec3(radius, radius, radius));
    output_box = aabb::surrounding_box(box0, box1);
    return true;
}
#endif