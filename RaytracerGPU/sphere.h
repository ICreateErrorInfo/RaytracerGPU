#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include <corecrt_math_defines.h>
#include "aabb.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
	vec3 center;
	float radius;
	material* mat_ptr;

private:
	__device__ static void get_sphere_uv(const vec3& p, float& u, float& v) {

		auto theta = acos(-p.y());
		auto phi = atan2(-p.z(), p.x()) + M_PI;

		u = phi / (2 * M_PI);
		v = theta / M_PI;
	}
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			get_sphere_uv(rec.normal, rec.u, rec.v);
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			get_sphere_uv(rec.normal, rec.u, rec.v);
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

__device__ bool sphere::bounding_box(double time0, double time1, aabb& output_box) const {
	output_box = aabb(
		center - vec3(radius, radius, radius),
		center + vec3(radius, radius, radius));
	return true;
}

#endif