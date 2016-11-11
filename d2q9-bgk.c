/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include<omp.h>
#include "mpi.h"
//#include<fenv.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define BLOCKSIZE       16  //Not used
#define NUMTHREADS      15
#define MPI_PROCS       1
#define MASTER          0
#define PAR                 //Comment this out and set NUMTHREADS to 1 for serial

//Vector size
#define VECSIZE 4

MPI_Datatype MPI_ROW_OF_OBSTACLES;
MPI_Datatype MPI_ROW_OF_CELLS;

/* struct to hold the parameter values */
struct __declspec(align(32)) t_param
{
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
  double free_cells_inv;
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */

};

typedef struct t_param t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr,  double** av_vels_ptr, int rank, int size,
               int* ny_local, int* displs);
void preprocess_obstacles(int* obstacles,const t_param params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles);
//int propagate(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr);
//int rebound(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
//int collision(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
double timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells,
                int* restrict obstacles, int start, int end);
double timestep_row(const t_param params, t_speed* cells0, t_speed* cells1, t_speed* tmp_cells0,
              t_speed* tmp_cells1, int* obstacles0, int* obstacles1, int ii, int tid);


int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr,  double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  //t_speed* cells1     = NULL;
  t_speed* tmp_cells = NULL;
  //t_speed* tmp_cells1 = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  //int*     obstacles1 = NULL;
  double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  omp_set_num_threads(NUMTHREADS);
  
  /************** MPI Part ********************/
  int size=1, rank=0;
#if MPI_PROCS>1
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
#endif

  int ny_local[MPI_PROCS];
  int displs[MPI_PROCS];
  /* **************************************** */
  
  //feenableexcept(FE_INVALID | FE_OVERFLOW);
  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels,
             rank, size, ny_local, displs);

  /***************************OpenMP*******************************/
  int omp_ny_local[NUMTHREADS];
  int omp_displs[NUMTHREADS];
  int omp_orig_ny_local = ny_local[rank]/NUMTHREADS;
  int omp_left = ny_local[rank]%NUMTHREADS;
  int one_for_last_thread = 0;
  int one_less_for_second_to_last = 0;
  //if it is less than 3 then it is 2 given that the smallest
  //size is 128x128 and max rank size is 64.
  if(omp_orig_ny_local<3 && omp_left){
    omp_left--;
    one_for_last_thread = 1;
  }
  else if(omp_orig_ny_local<3 && !left){
    one_for_last_thread = 1;
    one_less_for_second_to_last = 1;
  }
  //we need to make sure that the last thread gets at least 3 rows
  //so that accelerate_flow will not affect other rows. 
  for(int tid=0;tid<NUMTHREADS;tid++){
    if(tid<NUMTHREADS-2)
        omp_ny_local[tid] = omp_orig_ny_local;
    else if(tid == NUMTHREADS-2)
        omp_ny_local[tid] = omp_orig_ny_local - one_less_for_second_to_last;
    else if(tid == NUMTHREADS-1)
        omp_ny_local[tid] = omp_orig_ny_local + one_for_last_thread;
    if(tid<omp_left) omp_ny_local[tid]++;
    if(tid == MASTER)
        omp_displs[tid] = 1; //start from 1 to accommodate the halo rows
    else
        omp_displs[tid] = omp_displs[tid-1] + omp_ny_local[tid-1];
  }

 /* ************************************************************* */

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
#ifdef PAR
#pragma omp parallel firstprivate(tmp_cells,cells)
#endif
{
  int tid = omp_get_thread_num();
  int start = omp_displs[tid];
  int end = start + omp_ny_local[tid];
  for (unsigned int tt = 0; tt < params.maxIters;tt++)
  {
    #ifdef PAR
    #pragma omp barrier
    #endif
    if(tid==NUMTHREADS-1){
      accelerate_flow(params, cells, obstacles);
    }
    //#ifdef PAR
    //#pragma omp barrier
    //#endif
    double local = timestep(params, cells, tmp_cells, obstacles, start, end) * params.free_cells_inv;

    #ifdef PAR
    #pragma omp atomic
    #endif
    av_vels[tt] += local;

    t_speed* tmp = cells;
    cells = tmp_cells;
    tmp_cells = tmp;

#ifdef DEBUG
#ifdef PAR
#pragma omp single nowait
#endif
    {
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
    }
#endif
  }
}
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

#if MPI_PROCS>1
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}

inline int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel * 0.111111111111111111111111f;
  double w2 = params.density * params.accel * 0.0277777777777777777777778f;

  /* modify the 2nd row of the grid */
  int ii = params.ny - 2;
  //int tid = omp_get_thread_num();
  //int start = tid * (params.nx/NUMTHREADS);
  //int end   = (tid+1) * (params.nx/NUMTHREADS);
  for (unsigned int jj = 0; jj < params.nx; jj+=VECSIZE)
  {
      #pragma vector aligned
      for(int k=0;k<VECSIZE;k++){
        if (!obstacles[ii * params.nx + jj+k]
         && cells[ii*params.nx+jj+k].speeds[3]-w1>0.0
         && cells[ii*params.nx+jj+k].speeds[6]-w2>0.0
         && cells[ii*params.nx+jj+k].speeds[7]-w2>0.0){
         
                        /* increase 'east-side' densities */
                        cells[ii * params.nx + jj+k].speeds[1] += w1;
                        cells[ii * params.nx + jj+k].speeds[5] += w2;
                        cells[ii * params.nx + jj+k].speeds[8] += w2;
                        /* decrease 'west-side' densities */
                        cells[ii * params.nx + jj+k].speeds[3] -= w1;
                        cells[ii * params.nx + jj+k].speeds[6] -= w2;
                        cells[ii * params.nx + jj+k].speeds[7] -= w2;
                    
                
            
        }
      }
    }

  return EXIT_SUCCESS;
}


//double sqrt13(double n)
//{
//    double result;
//
//    __asm__(
//        "fsqrt\n\t"
//        : "=t"(result) : "0"(n)
//    );
//
//    return result;
//}

inline double timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells,
                       int* restrict obstacles, int start, int end)
{
  //static const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  static const double ic_sq = 3.0;
  //static const double ic_sq_sq = 9.0;
  static const double w0 = 4.0 / 9.0;  /* weighting factor */
  static const double w1 = 1.0 / 9.0;  /* weighting factor */
  static const double w2 = 1.0 / 36.0; /* weighting factor */
  double tot_u = 0.0;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  int start = tid * (params.ny/NUMTHREADS);
  int end   = (tid+1) * (params.ny/NUMTHREADS);
  for (unsigned int ii = start; ii < end; ii++)
  {
    int y_n = ii + 1;
    if(y_n == params.ny) y_n = 0;
    int y_s = 0;
    if (ii == 0) 
        y_s = params.ny - 1;
    else
        y_s = ii - 1;
    for(unsigned int jj = 0; jj < params.nx; jj+=VECSIZE){
        /* determine indices of axis-direction neighbours
        ** respecting periodic boundary conditions (wrap around) */
        double tmp[VECSIZE*NSPEEDS] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            int x = jj+k;
            int x_e = x + 1;
            if(x_e >= params.nx) x_e -= params.nx;
            int x_w = (x == 0) ? (params.nx - 1) : (x-1);
            tmp[VECSIZE*0+k] = cells[ii * params.nx + x].speeds[0];
            tmp[VECSIZE*1+k] = cells[ii * params.nx + x_w].speeds[1];
            tmp[VECSIZE*2+k] = cells[y_s * params.nx + x].speeds[2];
            tmp[VECSIZE*3+k] = cells[ii * params.nx + x_e].speeds[3];
            tmp[VECSIZE*4+k] = cells[y_n * params.nx + x].speeds[4];
            tmp[VECSIZE*5+k] = cells[y_s * params.nx + x_w].speeds[5];
            tmp[VECSIZE*6+k] = cells[y_s * params.nx + x_e].speeds[6];
            tmp[VECSIZE*7+k] = cells[y_n * params.nx + x_e].speeds[7];
            tmp[VECSIZE*8+k] = cells[y_n * params.nx + x_w].speeds[8];
            
        }

        double densvec[VECSIZE] __attribute__((aligned(32)));

        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            densvec[k] = tmp[VECSIZE*0+k];
            densvec[k] += tmp[VECSIZE*1+k];
            densvec[k] += tmp[VECSIZE*2+k];
            densvec[k] += tmp[VECSIZE*3+k];
            densvec[k] += tmp[VECSIZE*4+k];
            densvec[k] += tmp[VECSIZE*5+k];
            densvec[k] += tmp[VECSIZE*6+k];
            densvec[k] += tmp[VECSIZE*7+k];
            densvec[k] += tmp[VECSIZE*8+k];
        }

        double densinv[VECSIZE] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            densinv[k] = 1.0/densvec[k];
        }

        double u_x[VECSIZE] __attribute__((aligned(32)));
        double u_y[VECSIZE] __attribute__((aligned(32)));

        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            u_x[k] = tmp[VECSIZE*1+k] + tmp[VECSIZE*5+k];
            u_x[k] += tmp[VECSIZE*8+k];
            u_x[k] -= tmp[VECSIZE*3+k];
            u_x[k] -= tmp[VECSIZE*6+k];
            u_x[k] -= tmp[VECSIZE*7+k];
            //u_x[k] *= densinv[k];
            u_y[k] = tmp[VECSIZE*2+k] + tmp[VECSIZE*5+k];
            u_y[k] += tmp[VECSIZE*6+k];
            u_y[k] -= tmp[VECSIZE*4+k];
            u_y[k] -= tmp[VECSIZE*7+k];
            u_y[k] -= tmp[VECSIZE*8+k];
            //u_y[k] *= densinv[k];
        }

        double u_sq[VECSIZE] __attribute__((aligned(32)));

        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            u_sq[k] = u_x[k]*u_x[k] + u_y[k]*u_y[k];
        }

        double uvec[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            uvec[VECSIZE*1+k] =   u_x[k];
            uvec[VECSIZE*2+k] =            u_y[k];
            uvec[VECSIZE*3+k] = - u_x[k];
            uvec[VECSIZE*4+k] =          - u_y[k];
            uvec[VECSIZE*5+k] =   u_x[k] + u_y[k];
            uvec[VECSIZE*6+k] = - u_x[k] + u_y[k];
            uvec[VECSIZE*7+k] = - u_x[k] - u_y[k];
            uvec[VECSIZE*8+k] =   u_x[k] - u_y[k];
        }

        double ic_sqtimesu[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            ic_sqtimesu[VECSIZE*1+k] = uvec[VECSIZE*1+k]*ic_sq;
            ic_sqtimesu[VECSIZE*2+k] = uvec[VECSIZE*2+k]*ic_sq;
            ic_sqtimesu[VECSIZE*3+k] = uvec[VECSIZE*3+k]*ic_sq;
            ic_sqtimesu[VECSIZE*4+k] = uvec[VECSIZE*4+k]*ic_sq;
            ic_sqtimesu[VECSIZE*5+k] = uvec[VECSIZE*5+k]*ic_sq;
            ic_sqtimesu[VECSIZE*6+k] = uvec[VECSIZE*6+k]*ic_sq;
            ic_sqtimesu[VECSIZE*7+k] = uvec[VECSIZE*7+k]*ic_sq;
            ic_sqtimesu[VECSIZE*8+k] = uvec[VECSIZE*8+k]*ic_sq;
        }

        double ic_sqtimesu_sq[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            ic_sqtimesu_sq[VECSIZE*1+k] = ic_sqtimesu[VECSIZE*1+k] * uvec[VECSIZE*1+k];
            ic_sqtimesu_sq[VECSIZE*2+k] = ic_sqtimesu[VECSIZE*2+k] * uvec[VECSIZE*2+k];
            ic_sqtimesu_sq[VECSIZE*3+k] = ic_sqtimesu[VECSIZE*3+k] * uvec[VECSIZE*3+k];
            ic_sqtimesu_sq[VECSIZE*4+k] = ic_sqtimesu[VECSIZE*4+k] * uvec[VECSIZE*4+k];
            ic_sqtimesu_sq[VECSIZE*5+k] = ic_sqtimesu[VECSIZE*5+k] * uvec[VECSIZE*5+k];
            ic_sqtimesu_sq[VECSIZE*6+k] = ic_sqtimesu[VECSIZE*6+k] * uvec[VECSIZE*6+k];
            ic_sqtimesu_sq[VECSIZE*7+k] = ic_sqtimesu[VECSIZE*7+k] * uvec[VECSIZE*7+k];
            ic_sqtimesu_sq[VECSIZE*8+k] = ic_sqtimesu[VECSIZE*8+k] * uvec[VECSIZE*8+k];
        }

        double d_equ[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++)
        {
            d_equ[VECSIZE*0+k] = w0 * (densvec[k] - 0.5*densinv[k]*ic_sq*u_sq[k]);
            d_equ[VECSIZE*1+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*1+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*1+k]-u_sq[k]) );
            d_equ[VECSIZE*2+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*2+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*2+k]-u_sq[k]) );
            d_equ[VECSIZE*3+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*3+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*3+k]-u_sq[k]) );
            d_equ[VECSIZE*4+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*4+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*4+k]-u_sq[k]) );
            d_equ[VECSIZE*5+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*5+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*5+k]-u_sq[k]) );
            d_equ[VECSIZE*6+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*6+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*6+k]-u_sq[k]) );
            d_equ[VECSIZE*7+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*7+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*7+k]-u_sq[k]) );
            d_equ[VECSIZE*8+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*8+k] + 0.5 * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*8+k]-u_sq[k]) );
        }

        int obst=0;
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            obst+=obstacles[ii*params.nx+jj+k];
        }

        if(!obst){
            #pragma vector aligned
            for(int k=0;k<VECSIZE;k++){
                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                tot_u += sqrt(u_sq[k]) * densinv[k];
            }
        }
        else{

        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            if(!obstacles[ii * params.nx +jj +k]){
                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                tot_u += sqrt(u_sq[k]) * densinv[k]; 
            }
            else{
                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k];
                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*1+k];
                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*2+k];
                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*3+k];
                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*4+k];
                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*5+k];
                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*6+k];
                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*7+k];
                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*8+k];
                
            }
        }
        }
    }
  }

  return tot_u;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii*params.nx+jj])
      {
        /* local density total */
        double local_density = 0.0;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii*params.nx+jj].speeds[kk];
        }
        /* x-component of velocity */
        t_speed* cell = &cells[ii*params.nx+jj];
        double u_x = (cell->speeds[1]
                      + cell->speeds[5]
                      + cell->speeds[8]
                      - (cell->speeds[3]
                         + cell->speeds[6]
                         + cell->speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (cell->speeds[2]
                      + cell->speeds[5]
                      + cell->speeds[6]
                      - (cell->speeds[4]
                         + cell->speeds[7]
                         + cell->speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
      }
    }
  }

  return tot_u * params.free_cells_inv;
}

int initialise(char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr, int rank, int size,
               int* ny_local, int* displs)
{
  char   message[1024];  /* message buffer */
  FILE* fp;
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */


  /* open the parameter file */
  fp = fopen(paramfile,"r");

  if (fp == NULL)
  {
      sprintf(message, "could not open input parameter file: %s", paramfile);
      die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  int numOfFreeCells = params->nx*params->ny;

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
/* ******************************MPI********************************* */
    MPI_Type_contiguous(params->nx,MPI_INT,&MPI_ROW_OF_OBSTACLES);
    MPI_Type_commit(&MPI_ROW_OF_OBSTACLES);


  
    int orig_ny_local = params->ny/size;
    int left = params->ny%size;
    int one_for_last_rank = 0;
    int one_less_for_second_to_last = 0;
    //if it is less than 3 then it is 2 given that the smallest
    //size is 128x128 and max rank size is 64.
    if(orig_ny_local<3 && left){
        left--;
        one_for_last_rank = 1;
    }
    else if(orig_ny_local<3 && !left){
        one_for_last_rank = 1;
        one_less_for_second_to_last = 1;
    }
    //we need to make sure that the last rank gets at least 3 rows
    //so that accelerate_flow will not affect other rows. 
    for(int proc=0;proc<size;proc++){
        if(proc<size-2)
            ny_local[proc] = orig_ny_local;
        else if(proc == size-2)
            ny_local[proc] = orig_ny_local - one_less_for_second_to_last;
        else if(proc == size-1)
            ny_local[proc] = orig_ny_local + one_for_last_rank;
        if(proc<left) ny_local[proc]++;
        if(proc == MASTER)
            displs[proc] = 1;
        else
            displs[proc] = displs[proc-1] + ny_local[proc-1];
    }
/*************************************************************************/
    
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((ny_local[rank]+2) * params->nx));

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((ny_local[rank]+2) * params->nx));

    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = (int*)malloc(sizeof(int) * ((ny_local[rank]+2) * params->nx));//+2 not needed but makes things easier

    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;
 
  for (unsigned int ii = 1; ii < ny_local[rank]+1; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
    {
      t_speed* cell = &((*cells_ptr)[ii*params->nx+jj]);
      /* centre */
      cell->speeds[0] = w0;
      /* axis directions */
      cell->speeds[1] = w1;
      cell->speeds[2] = w1;
      cell->speeds[3] = w1;
      cell->speeds[4] = w1;
      /* diagonals */
      cell->speeds[5] = w2;
      cell->speeds[6] = w2;
      cell->speeds[7] = w2;
      cell->speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (unsigned int ii = 1; ii < ny_local[rank]+1; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii*params->nx+jj] = 0;
    }
  }
  int* obstacles_all = NULL;
  /* open the obstacle data file */
  
  if(rank==MASTER)
  {
    obstacles_all = (int*)malloc( sizeof(int) * params->nx * params->ny );
    for(unsigned int ii=0;ii<params->ny;ii++){
        for(unsigned int jj=0;jj<params->nx;jj++){
            obstacles_all[ii*params->nx+jj] = 0;
        }
    }

    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        if(obstacles_all[yy*params->nx+xx]==0)
            numOfFreeCells--;
        obstacles_all[yy*params->nx+xx]=blocked;
        
    }
    params->free_cells_inv = 1.0/numOfFreeCells;

    /* and close the file */
    fclose(fp);

    *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);
    
  }
  
  int obstlinesize = sizeof(int) * params->nx;

#if MPI_PROCS>1
  MPI_Scatterv(obstacles_all, ny_local, displs, ,
               *obstacles_ptr+obstlinesize, ny_local[rank], ,
               MASTER, MPI_COMM_WORLD);
#else
  int* temp = *obstacles_ptr;
  *obstacles_ptr = obstacles_all;
  obstacles_all = temp;
#endif
  if(obstacles_all) free(obstacles_all);
  return EXIT_SUCCESS;
}


int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr,  double** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);  
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0;  /* accumulator */

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      for (unsigned int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii*params.nx+jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii*params.nx+jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;
        t_speed* cell = &cells[ii*params.nx+jj];

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cell->speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cell->speeds[1]
               + cell->speeds[5]
               + cell->speeds[8]
               - (cell->speeds[3]
                  + cell->speeds[6]
                  + cell->speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cell->speeds[2]
               + cell->speeds[5]
               + cell->speeds[6]
               - (cell->speeds[4]
                  + cell->speeds[7]
                  + cell->speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii*params.nx+jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
