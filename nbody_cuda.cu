#include <math.h>
#include<cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SOFTENING 1e-9f

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a)<(b))?(a):(b))

typedef struct { float x, y, z, vx, vy, vz; } Body;
void randomizeBodies(float *data, int n) {
  /* Function to initialize bodies randomly
  */
  double sum = 0;
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    sum += data[i];
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {
  /* Function to calculate body force for each body
  */

  // The tid for this block
  int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid0; i < n; i += blockDim.x * gridDim.x) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      // Calculating force on each particle
      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    // Finding velocity in each direction
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int particle_positions_to_csv(FILE *datafile, int iter, Body *p_h, int nBodies) {
  /* Sending files to csv for plot generation
  */
  for (int i = 0 ; i < nBodies; i++) 
    fprintf(datafile, "%i, %f, %f, %f\n", iter, p_h[i].x, p_h[i].y, p_h[i].z);
  return 0;
}

__global__ void position(Body *p, float dt, int n)
{
  /* Function to find position of each particle
  */
  int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid0; i < n; i += blockDim.x * gridDim.x) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {

  cudaEvent_t                /* CUDA timers */
  start_device,
  stop_device;  
  float time_device;
  cudaEventCreate(&start_device);
  cudaEventCreate(&stop_device);


  FILE *datafile;  
  int nBodies = 1000;
  int nthreads_per_block = 1;

  // Taking number of bodies and threads per block as input from user
  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nthreads_per_block = atoi(argv[2]);

  double nblocks = min(nBodies/nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

  const float dt = 0.01f; // time step
  const int nIters = 1000;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf_h = (float*)malloc(bytes);
  float *buf;
  cudaMalloc((void **) &buf, (bytes));
  Body *p_h = (Body*)buf_h;
  Body *p = (Body*)buf;
  
  randomizeBodies(buf_h, 6*nBodies); // Init pos / vel data
  cudaMemcpy(buf,buf_h,bytes,cudaMemcpyHostToDevice);


  double totalTime = 0.0;
  int to_print = 1;

  datafile = fopen("nbody.csv","w");

  /* ------------------------------*/
  /*     MAIN LOOP                 */
  /* ------------------------------*/
  for (int iter = 1; iter <= nIters; iter++) {
    
    if (iter % to_print == 0)
    {
      cudaMemcpy(p_h,p,bytes,cudaMemcpyDeviceToHost);
      particle_positions_to_csv(datafile, iter/to_print, p_h, nBodies);
    }

    cudaEventRecord( start_device, 0 );  

    bodyForce<<<nblocks, nthreads_per_block>>>(p, dt, nBodies);           // compute interbody forces
    position<<<nblocks, nthreads_per_block>>>(p, dt, nBodies); 
    

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );
    if (iter > 1)                       // First iter is warm up
      totalTime = totalTime + (time_device / 1000); 
  }
  
  fclose(datafile);
  double avgTime = totalTime / (double)(nIters-1); 

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);
  cudaFree(buf);
  free(buf_h);
}