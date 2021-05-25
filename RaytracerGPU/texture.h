#ifndef TEXTURE_H
#define TEXTURE_H

#include "vec3.h"
#include <curand_kernel.h>
#include "perlin.h"

class texture1 {
public:
    __device__ virtual vec3 value(double u, double v, const vec3& p) const = 0;
};

class solid_color : public texture1{
public:
    __device__ solid_color() {}
    __device__ solid_color(vec3 c) : color_value(c) {}

    __device__ solid_color(double red, double green, double blue)
        : solid_color(vec3(red, green, blue)) {}

    __device__ virtual vec3 value(double u, double v, const vec3& p) const override {
        return color_value;
    }

private:
    vec3 color_value;
};

class checker_texture : public texture1 {
public:
    __device__ checker_texture() {}

    __device__ checker_texture(texture1* _even, texture1* _odd)
        : even(_even), odd(_odd) {}

    __device__ checker_texture(vec3 c1, vec3 c2)
        : even(new solid_color(c1)), odd(new solid_color(c2)) {}

    __device__ virtual vec3 value(double u, double v, const vec3& p) const override {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    texture1* odd;
    texture1* even;
};

class noise_texture : public texture1 {
public:
    __device__ noise_texture(curandState* d_rand_state, double sc)
    {
        noise = perlin(d_rand_state);
        scale = sc;
    }

    __device__ virtual vec3 value(double u, double v, const vec3& p) const override {

        return vec3(1, 1, 1) * noise.turb(scale * p);
    }

public:
    perlin noise;
    double scale;
};

class ImageTexture : public texture1 {
public:
    __device__ ImageTexture() {}
    __device__ ImageTexture(unsigned char* pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
    __device__ virtual vec3 value(double u, double v, const vec3& p) const;

    unsigned char* data;
    int nx, ny;
};

__device__ vec3 ImageTexture::value(double u, double v, const vec3& p) const {
    int i = u * nx;
    int j = (1 - v) * ny - 0.001;
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > nx - 1) i = nx - 1;
    if (j > ny - 1) j = ny - 1;
    float r = int(data[3 * i + 3 * nx * j]) / 255.0;
    float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
    float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
    return vec3(r, g, b);
}

#endif