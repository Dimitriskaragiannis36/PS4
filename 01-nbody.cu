#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "timer.h"  
#include "files.h"	

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

__global__ void bodyForce(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop για μεγαλύτερη ασφάλεια και απόδοση
    for (int idx = i; idx < n; idx += stride) {
        // Caching σε registers για αποφυγή πολλαπλών αναγνώσεων από την Global Memory
        float pi_x = p[idx].x;
        float pi_y = p[idx].y;
        float pi_z = p[idx].z;
        
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - pi_x;
            float dy = p[j].y - pi_y;
            float dz = p[j].z - pi_z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr); 
            float invDist3 = invDist * invDist * invDist;	

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        p[idx].vx += dt * Fx;
        p[idx].vy += dt * Fy;
        p[idx].vz += dt * Fz;
    }
}

__global__ void integratePosition(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < n; idx += stride) {
        p[idx].x += p[idx].vx * dt;
        p[idx].y += p[idx].vy * dt;
        p[idx].z += p[idx].vz * dt;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2<<11; 
    if (argc > 1) nBodies = 2<<atoi(argv[1]); 

    const char * initialized_values;
    const char * solution_values;

    if (nBodies == 2<<11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; 
    const int nIters = 10;  

    int bytes = nBodies * sizeof(Body);
    Body *h_p = (Body*)malloc(bytes); 
    read_values_from_file(initialized_values, (float*)h_p, bytes); 

    Body *d_p;
    cudaMalloc(&d_p, bytes); 
    cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice); 

    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize; 
    double totalTime = 0.0;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer(); 

        bodyForce<<<gridSize, blockSize>>>(d_p, dt, nBodies); 
        // ΑΦΑΙΡΕΘΗΚΕ: cudaDeviceSynchronize(); - Τα kernels στο ίδιο stream εκτελούνται σειριακά

        integratePosition<<<gridSize, blockSize>>>(d_p, dt, nBodies); 
        cudaDeviceSynchronize(); // Μόνο ένα synchronize στο τέλος για τη σωστή μέτρηση χρόνου

        double tElapsed = GetTimer() / 1000.0;     
        totalTime += tElapsed;
    }

    cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost); 
    write_values_to_file(solution_values, (float*)h_p, bytes);  

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    free(h_p);        
    cudaFree(d_p);    
    return 0;
}