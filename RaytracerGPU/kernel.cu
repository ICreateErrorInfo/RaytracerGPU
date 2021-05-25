
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <bitset>
#include <fstream>

#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include <curand_kernel.h>
#include "hitable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "aarect.h"
#include "moving_spheres.h"
#include "box.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "C:\Users\Moritz\source\repos\stb-master/stb_image.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r, hitable** world, vec3 background, int depth, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	vec3 cur_emitted = vec3(0,0,0);
	for (int i = 0; i < depth; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_emitted += emitted * cur_attenuation;
				cur_ray = scattered;
			}
			else {
				return cur_emitted + emitted * cur_attenuation;
			}
		}
		else {
			//return cur_emitted;

			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return cur_emitted; // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, int depth, vec3 background, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, background, depth, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		auto center2 = vec3(0, 0, -20) + vec3(0, .5, 0);
		d_list[0] = new moving_sphere(vec3(0, 0, -20), center2, 0, 1, 4, new metal(vec3(1, 0.32, 0.36), 0));
		d_list[1] = new sphere(vec3(0, -1004, -20), 1000, new lambertian(new checker_texture(vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9))));
		d_list[2] = new sphere(vec3(5, -1, -15), 2, new metal(vec3(0.90, 0.76, 0.46), 0.0));
		d_list[3] = new sphere(vec3(5, 0, -25), 3, new metal(vec3(0.65, 0.77, 0.97), 0.0));
		d_list[4] = new sphere(vec3(-5.5, 0, -15), 3, new metal(vec3(0.90, 0.90, 0.90), 0.0));

		*d_world = new hitable_list(d_list, 5);

		vec3 lookfrom(0, 0, 0);
		vec3 lookat(0, 0, -1);
		float dist_to_focus = 20;
		float aperture = .1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			50,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);
	}
}

__global__ void create_world1(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		auto center2 = vec3(0, 0, -20) + vec3(0, .5, 0);
		auto checker = new checker_texture(vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9));
		d_list[0] = new sphere(vec3(0, -10, 0), 10, new lambertian(checker));
		d_list[1] = new sphere(vec3(0, 10, 0), 10, new lambertian(checker));

		*d_world = new hitable_list(d_list, 2);


		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);
	}
}

__global__ void create_world2(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		//curandState* randState;
		//cudaMalloc((void**)&randState, sizeof(curandState));
		//
		//curand_init(1984, 0, 0, &randState[0]);

		auto pertext = new checker_texture(vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9));
		auto pertext1 = new noise_texture(rand_state, 4);
		d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext1));
		d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(pertext1));

		*d_world = new hitable_list(d_list, 2);


		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);

		//cudaFree(randState);
	}
}

__global__ void create_world3(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		auto pertext1 = new noise_texture(rand_state, 4);
		auto difflight = new diffuse_light(vec3(4, 4, 4));
		d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext1));
		d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(pertext1));
		d_list[2] = new xy_rect(3, 5, 1, 3, -2, difflight);

		*d_world = new hitable_list(d_list, 3);


		vec3 lookfrom(26, 3, 6);
		vec3 lookat(0, 2, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);

		//cudaFree(randState);
	}
}

__global__ void Earth(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, ImageTexture** tex) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		d_list[0] = new sphere(vec3(0, 0, 0), 2, new lambertian(*tex));

		*d_world = new hitable_list(d_list, 1);


		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);
	}
}

__global__ void cornell_box(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		auto red   = new lambertian   (vec3(.65, .05, .05));
		auto white = new lambertian   (vec3(.73, .73, .73));
		auto green = new lambertian   (vec3(.12, .45, .15));
		auto light = new diffuse_light(vec3(15, 15, 15));

		d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);
		d_list[1] = new yz_rect(0, 555, 0, 555, 0, red);
		d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);
		d_list[3] = new xz_rect(0, 555, 0, 555, 0, white);
		d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);
		d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);
		
		hitable* box1 = new Box(vec3(0, 0, 0), vec3(165, 330, 165), white);
		box1 = new rotate_y(box1, 15);
		box1 = new translate(box1, vec3(265, 0, 295));
		d_list[6] = box1;

		hitable* box2 = new Box(vec3(0, 0, 0), vec3(165, 165, 165), white);
		box2 = new rotate_y(box2, -18);
		box2 = new translate(box2, vec3(130, 0, 65));
		d_list[7] = box2;

		*d_world = new hitable_list(d_list, 8);


		vec3 lookfrom(278, 278, -800);
		vec3 lookat(278, 278, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			40,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);
	}
}

__global__ void cornell_box_smoke(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		auto red = new lambertian(vec3(.65, .05, .05));
		auto white = new lambertian(vec3(.73, .73, .73));
		auto green = new lambertian(vec3(.12, .45, .15));
		auto light = new diffuse_light(vec3(15, 15, 15));

		d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);
		d_list[1] = new yz_rect(0, 555, 0, 555, 0, red);
		d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);
		d_list[3] = new xz_rect(0, 555, 0, 555, 0, white);
		d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);
		d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);

		hitable* box1 = new Box(vec3(0, 0, 0), vec3(165, 330, 165), white);
		box1 = new rotate_y(box1, 15);
		box1 = new translate(box1, vec3(265, 0, 295));
		d_list[6] = new constant_medium(box1, 0.01, vec3(0, 0, 0), rand_state);

		hitable* box2 = new Box(vec3(0, 0, 0), vec3(165, 165, 165), white);
		box2 = new rotate_y(box2, -18);
		box2 = new translate(box2, vec3(130, 0, 65));
		d_list[7] = new constant_medium(box2, 0.01, vec3(1, 1, 1), rand_state);

		*d_world = new hitable_list(d_list, 8);


		vec3 lookfrom(278, 278, -800);
		vec3 lookat(278, 278, 0);
		float dist_to_focus = 10;
		float aperture = 0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			40,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,
			1.0);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	for (int i = 0; i < 5; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

__global__ void texture_init(unsigned char* tex_data, int nx, int ny, ImageTexture** tex) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*tex = new ImageTexture(tex_data, nx, ny);
	}
}


int main()
{
	int nx = 600;
	int ny = 600;
	int ns = 5000;
	int tx = 8;
	int ty = 8;
	int depth = 50;
	vec3 background = vec3(0.70, 0.80, 1.00);

	std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	int tex_x, tex_y, tex_n;
	unsigned char* tex_data_host = stbi_load("C:/Users/Moritz/source/repos/RaytracerGPU/RaytracerGPU/earthmap.jpg", &tex_x, &tex_y, &tex_n, 0);

	unsigned char* tex_data;
	checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
	checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

	ImageTexture** texture;
	checkCudaErrors(cudaMalloc((void**)&texture, sizeof(ImageTexture*)));
	texture_init << <1, 1 >> > (tex_data, tex_x, tex_y, texture);

	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	// make our world of hitables & the camera
	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);

	switch (7)
	{
	case 1:
		create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny);
		break;

	case 2:
		create_world1 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny);
		break;

	case 3:
		create_world2 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state);
		break;
	case 4:
		create_world3 << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state);
		break;
	case 5:
		cornell_box << <1, 1 >> > (d_list, d_world, d_camera, nx, ny);
		break;
	case 6:
		cornell_box_smoke << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state);
		break;
	case 7:
		Earth << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, texture);
		break;
	}

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();
	// Render our buffer
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, depth, background, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output FB as Image
	std::ofstream myfile("Image1.ppm", std::ios::out | std::ios::binary);

	myfile << "P6\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());

			myfile.write(reinterpret_cast<const char*>(&ir), sizeof(char));
			myfile.write(reinterpret_cast<const char*>(&ig), sizeof(char));
			myfile.write(reinterpret_cast<const char*>(&ib), sizeof(char));
		}
	}

	// clean up
	myfile.close();
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());

	cudaDeviceReset();
}