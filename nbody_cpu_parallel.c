#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <omp.h>
#include <time.h>

#define SOFTENING 1e-9f

struct timeval timerStart;

void StartTimer(){
  gettimeofday(&timerStart, NULL);
}

double GetTimer(){
  struct timeval timerStop, timerElapsed;
  gettimeofday(&timerStop, NULL);
  timersub(&timerStop, &timerStart, &timerElapsed);

  return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

typedef struct { float x, y, z, vx, vy, vz; } Body;
void randomizeBodies(float *data, int n) {
  /* Function to initialize bodies randomly
  */
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  /* Function to calculate body force for each body
  */
  #ifdef PARALLEL
  #pragma omp parallel for default(none) shared(p, n, dt) schedule(static)
  #endif
  for (int i = 0; i < n; i++) { 
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

int particle_positions_to_csv(FILE *datafile, int iter, Body *p, int nBodies) {
  /* Sending files to csv for plot generation
  */
  for (int i = 0 ; i < nBodies; i++) 
    fprintf(datafile, "%i, %f, %f, %f\n", iter, p[i].x, p[i].y, p[i].z);
  return 0;
}

int main(const int argc, const char** argv) {

  FILE *datafile;  
  int nBodies = 10000;
  int nthreads = 1;

  // Taking number of bodies and threads as arguments
  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nthreads = atoi(argv[2]);

  omp_set_num_threads(nthreads);

  const float dt = 0.01f; // time step
  const int nIters = 1000;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;
  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  double totalTime = 0.0;
  double t1, t2;

  int to_print = 1;

  datafile = fopen("nbody.csv","w");

  /* ------------------------------*/
  /*     MAIN LOOP                 */
  /* ------------------------------*/
  for (int iter = 1; iter <= nIters; iter++) {
    
    // Printing only for some specific iterations
    if (iter % to_print == 0)
      particle_positions_to_csv(datafile, iter/to_print, p, nBodies);

    t1 = omp_get_wtime();

    bodyForce(p, dt, nBodies);           // compute interbody forces

    #ifdef PARALLEL
    #pragma omp parallel for default(none) shared(p, nBodies, dt) schedule(static)
    #endif
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    t2 = omp_get_wtime();
    if (iter > 1)                       // First iter is warm up
      totalTime = totalTime + t2 - t1; 
  }
  
  fclose(datafile);
  double avgTime = totalTime / (double)(nIters-1); 

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);
  free(buf);
}