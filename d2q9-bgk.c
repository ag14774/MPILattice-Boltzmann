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
//#include<fenv.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define BLOCKSIZE       16  //Not used
#define NUMTHREADS      16

//Vector size
#define VECSIZE 4

//nyhalf = ny/2
#define getcelladdr(ii,jj,arr1,arr2,nyhalf,nx) ((ii<nyhalf) ? (&(arr1[ii*nx+jj])) : (&(arr2[(ii-nyhalf)*nx+jj])))
#define getcellval(ii,jj,arr1,arr2,nyhalf,nx) ((ii<nyhalf) ? (arr1[ii*nx+jj]) : (arr2[(ii-nyhalf)*nx+jj]))
#define getcellspeed(ii,jj,sp,arr1,arr2,nyhalf,nx) ((ii<nyhalf) ? (arr1[ii*nx+jj].speeds[sp]) : (arr2[(ii-nyhalf)*nx+jj].speeds[sp]))


/* struct to hold the parameter values */
struct __declspec(align(32)) t_param
{
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
  double free_cells_inv;
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    nyhalf;        /*  to prevent it from redoing the division */
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
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr0, t_speed** cells_ptr1, t_speed** tmp_cells_ptr0,
               t_speed** tmp_cells_ptr1, int** obstacles_ptr0, int** obstacles_ptr1, double** av_vels_ptr);
void preprocess_obstacles(int* obstacles,const t_param params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* restrict cells1, int* restrict obstacles1);
//int propagate(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr);
//int rebound(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
//int collision(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
double timestep(const t_param params, t_speed* restrict cells0, t_speed* restrict cells1, t_speed* restrict tmp_cells0,
              t_speed* restrict tmp_cells1, int* restrict obstacles0, int* restrict obstacles1, int tid);
double timestep_row(const t_param params, t_speed* cells0, t_speed* cells1, t_speed* tmp_cells0,
              t_speed* tmp_cells1, int* obstacles0, int* obstacles1, int ii, int tid);


int write_values(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr0, t_speed** cells_ptr1, t_speed** tmp_cells_ptr0,
             t_speed** tmp_cells_ptr1, int** obstacles_ptr0, int** obstacles_ptr1,  double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells0, t_speed* cells1);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1);

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
  t_speed* cells0     = NULL;    /* grid containing fluid densities */
  t_speed* cells1     = NULL;
  t_speed* tmp_cells0 = NULL;
  t_speed* tmp_cells1 = NULL;    /* scratch space */
  int*     obstacles0 = NULL;    /* grid indicating which cells are blocked */
  int*     obstacles1 = NULL;
  double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  //omp_set_num_threads(1);
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
  initialise(paramfile, obstaclefile, &params, &cells0, &cells1, &tmp_cells0, &tmp_cells1, &obstacles0, &obstacles1, &av_vels);
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

#pragma omp parallel firstprivate(tmp_cells0,cells0, tmp_cells1, cells1)
{
  int tid = omp_get_thread_num();
  for (unsigned int tt = 0; tt < params.maxIters;tt++)
  {
    #pragma omp barrier
    if(tid==NUMTHREADS-1){
        accelerate_flow(params, cells1, obstacles1);
    }
    #pragma omp barrier
    double local = timestep(params, cells0, cells1, tmp_cells0, tmp_cells1, obstacles0, obstacles1,tid);
    //local += timestep_row(params, cells0, cells1, tmp_cells0, tmp_cells1, obstacles0, obstacles1,0,tid);
    //local += timestep_row(params, cells0, cells1, tmp_cells0, tmp_cells1, obstacles0, obstacles1,params.nyhalf-1,tid);
    //local += timestep_row(params, cells0, cells1, tmp_cells0, tmp_cells1, obstacles0, obstacles1,params.nyhalf,tid);
    //local += timestep_row(params, cells0, cells1, tmp_cells0, tmp_cells1, obstacles0, obstacles1,params.ny-1,tid);

    #pragma omp atomic
    av_vels[tt] += local * params.free_cells_inv;

    t_speed* tmp = cells0;
    cells0 = tmp_cells0;
    tmp_cells0 = tmp;

    tmp = cells1;
    cells1 = tmp_cells1;
    tmp_cells1= tmp;

#ifdef DEBUG
//#pragma omp single
//    {
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells0, cells1));
//    }
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
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells0, cells1, obstacles0, obstacles1));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells0, cells1, obstacles0, obstacles1, av_vels);
  finalise(&params, &cells0, &cells1, &tmp_cells0, &tmp_cells1, &obstacles0, &obstacles1, &av_vels);

  return EXIT_SUCCESS;
}

inline int accelerate_flow(const t_param params, t_speed* restrict cells1, int* restrict obstacles1)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel * 0.111111111111111111111111f;
  double w2 = params.density * params.accel * 0.0277777777777777777777778f;

  /* modify the 2nd row of the grid */
  int ii = params.nyhalf - 2;
  //int tid = omp_get_thread_num();
  //int start = tid * (params.nx/NUMTHREADS);
  //int end   = (tid+1) * (params.nx/NUMTHREADS);
  //#pragma omp for
  for (unsigned int jj = 0; jj < params.nx; jj+=VECSIZE)
  {
      #pragma vector aligned
      for(int k=0;k<VECSIZE;k++){
        if (!obstacles1[ii * params.nx + jj+k]
         && cells1[ii*params.nx+jj+k].speeds[3]-w1>0.0
         && cells1[ii*params.nx+jj+k].speeds[6]-w2>0.0
         && cells1[ii*params.nx+jj+k].speeds[7]-w2>0.0){

                        /* increase 'east-side' densities */
                        cells1[ii * params.nx + jj+k].speeds[1] += w1;
                        cells1[ii * params.nx + jj+k].speeds[5] += w2;
                        cells1[ii * params.nx + jj+k].speeds[8] += w2;
                        /* decrease 'west-side' densities */
                        cells1[ii * params.nx + jj+k].speeds[3] -= w1;
                        cells1[ii * params.nx + jj+k].speeds[6] -= w2;
                        cells1[ii * params.nx + jj+k].speeds[7] -= w2;



        }
      }
    }

  return EXIT_SUCCESS;
}

//inline int accelerate_flow(const t_param params, t_speed* restrict cells1, int* restrict obstacles1)
//{
//  /* compute weighting factors */
//  double w1 = params.density * params.accel * 0.111111111111111111111111f;
//  double w2 = params.density * params.accel * 0.0277777777777777777777778f;

  /* modify the 2nd row of the grid */
//  int ii = params.nyhalf - 2;
  //int tid = omp_get_thread_num();
  //int start = tid * (params.nx/NUMTHREADS);
  //int end   = (tid+1) * (params.nx/NUMTHREADS);
  //#pragma omp for
//  for (unsigned int jj = 0; jj < params.nx; jj+=VECSIZE)
//  {
//    int obst=0;
//    #pragma vector aligned
//    for(int k=0;k<VECSIZE;k++)
//        obst+=obstacles1[ii*params.nx+jj+k];
    /* if the cell is not occupied and
    ** we don't send a negative density */
//    if(!obst){
//      #pragma vector aligned
//      for(int k=0;k<VECSIZE;k++){
//        double res1 = cells1[ii * params.nx + jj+k].speeds[3] - w1;
//        if(res1>0.0){
//            double res2 = cells1[ii * params.nx + jj+k].speeds[6] - w2;
//            if(res2>0.0){
//               double res3 = cells1[ii * params.nx + jj+k].speeds[7] - w2;
//               if(res3>0.0){
                  /* increase 'east-side' densities */
//                  cells1[ii * params.nx + jj+k].speeds[1] += w1;
//                  cells1[ii * params.nx + jj+k].speeds[5] += w2;
//                  cells1[ii * params.nx + jj+k].speeds[8] += w2;
                  /* decrease 'west-side' densities */
//                  cells1[ii * params.nx + jj+k].speeds[3] = res1;
//                  cells1[ii * params.nx + jj+k].speeds[6] = res2;
//                  cells1[ii * params.nx + jj+k].speeds[7] = res3;
//               }
//            }
//        }
//     }
//    }
//    else{
//      #pragma vector aligned
//      for(int k=0;k<VECSIZE;k++){
//        if (!obstacles1[ii * params.nx + jj+k]){
//            double res1 = cells1[ii * params.nx + jj+k].speeds[3] - w1;
//            if(res1>0.0){
//                double res2 = cells1[ii * params.nx + jj+k].speeds[6] - w2;
//                if(res2>0.0){
//                    double res3 = cells1[ii * params.nx + jj+k].speeds[7] - w2;
//                    if(res3>0.0){
                        /* increase 'east-side' densities */
//                        cells1[ii * params.nx + jj+k].speeds[1] += w1;
//                        cells1[ii * params.nx + jj+k].speeds[5] += w2;
//                        cells1[ii * params.nx + jj+k].speeds[8] += w2;
                        /* decrease 'west-side' densities */
//                        cells1[ii * params.nx + jj+k].speeds[3] = res1;
//                        cells1[ii * params.nx + jj+k].speeds[6] = res2;
//                        cells1[ii * params.nx + jj+k].speeds[7] = res3;
//                    }
//                }
//            }
//        }
//      }
//    }

//  }
//  return EXIT_SUCCESS;
//}

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

inline double timestep_row(const t_param params, t_speed* restrict cells0, t_speed* restrict cells1, t_speed* restrict tmp_cells0,
              t_speed* restrict tmp_cells1, int* restrict obstacles0, int* restrict obstacles1, int ii, int tid)
{
  //static const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  static const double ic_sq = 3.0;
  //static const double ic_sq_sq = 9.0;
  static const double w0 = 4.0 / 9.0;  /* weighting factor */
  static const double w1 = 1.0 / 9.0;  /* weighting factor */
  static const double w2 = 1.0 / 36.0; /* weighting factor */
  double tot_u = 0.0;

    t_speed* restrict cells = NULL;
    t_speed* restrict tmp_cells = NULL;
    int* restrict obstacles = NULL;
    int qq = 0;
    if(ii<params.nyhalf){
        cells = cells0;
        tmp_cells = tmp_cells0;
        obstacles = obstacles0;
        qq = ii;
    }
    else{
        cells = cells1;
        tmp_cells = tmp_cells1;
        obstacles = obstacles1;
        qq = ii - params.nyhalf;
    }

    int y_n = ii+1;
    if(y_n == params.ny) y_n = 0;
    int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);

    int start = tid * (params.nx/NUMTHREADS);
    int end   = (tid+1) * (params.nx/NUMTHREADS);

    for(unsigned int jj = start; jj < end; jj+=VECSIZE){
        /* determine indices of axis-direction neighbours
        ** respecting periodic boundary conditions (wrap around) */
        double tmp[VECSIZE*NSPEEDS] __attribute__((aligned(32)));
        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            int x = jj+k;
            int x_e = x + 1;
            if(x_e >= params.nx) x_e -= params.nx;
            int x_w = x-1;
            if(x==0) x_w = params.nx-1;
            tmp[VECSIZE*0+k] = getcellspeed(ii,x,0,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*1+k] = getcellspeed(ii,x_w,1,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*2+k] = getcellspeed(y_s,x,2,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*3+k] = getcellspeed(ii,x_e,3,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*4+k] = getcellspeed(y_n,x,4,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*5+k] = getcellspeed(y_s,x_w,5,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*6+k] = getcellspeed(y_s,x_e,6,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*7+k] = getcellspeed(y_n,x_e,7,cells0,cells1,params.nyhalf,params.nx);
            tmp[VECSIZE*8+k] = getcellspeed(y_n,x_w,8,cells0,cells1,params.nyhalf,params.nx);

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
            obst+=obstacles[qq*params.nx+jj+k];
        }

        if(!obst){
            #pragma vector aligned
            for(int k=0;k<VECSIZE;k++){
                tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                tot_u += sqrt(u_sq[k]) * densinv[k];
            }
        }
        else{

          #pragma vector aligned
          for(int k=0;k<VECSIZE;k++){
              if(!obstacles[qq * params.nx +jj +k]){
                  tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                  tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                  tot_u += sqrt(u_sq[k]) * densinv[k];
              }
              else{
                  tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*1+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*2+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*3+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*4+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*5+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*6+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*7+k];
                  tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*8+k];

              }
          }
        }
  }

  return tot_u;
}

// inline double timestep_row(const t_param params, t_speed* cells0, t_speed* cells1, t_speed* tmp_cells0,
//               t_speed* tmp_cells1, int* obstacles0, int* obstacles1, int ii, int tid)
// {
//   static const double c_sq = 1.0 / 3.0; /* square of speed of sound */
//   static const double twooverthree = 2.0/3.0;
//   static const double two_c_sq_sq = 2.0 / 9.0;
//   static const double w0 = 4.0 / 81.0 * 4.5;  /* weighting factor */
//   static const double w1 = 1.0 / 9.0 * 4.5 ;  /* weighting factor */
//   static const double w2 = 1.0 / 36.0 * 4.5; /* weighting factor */
//   double oneminusomega = 1.0 - params.omega;
//   double tot_u = 0.0;
//   //int rows[4] = {0, params.nyhalf-1, params.nyhalf, params.ny-1};
//
//   /* loop over the cells in the grid
//   ** NB the collision step is called after
//   ** the propagate step and so values of interest
//   ** are in the scratch-space grid */
//
//   //int tid = omp_get_thread_num(); //4 threads for each of the 4 remaining special rows
//   //for(int qq=0;qq<4;qq++){
//   //int ii = rows[qq];
//   int y_n = ii+1;
//   if(y_n == params.ny) y_n = 0;
//   int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
//
//   int start = tid * (params.nx/NUMTHREADS);
//   int end   = (tid+1) * (params.nx/NUMTHREADS);
//   //printf("tid:%d, ii:%d, start:%d, end:%d\n",tid,ii,start,end);
//
//   for(unsigned int jj = start; jj < end; jj++){
//     /* determine indices of axis-direction neighbours
//     ** respecting periodic boundary conditions (wrap around) */
//     int x_e = jj + 1;
//     if (x_e == params.nx) x_e = 0;
//     int x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
//     /* propagate densities to neighbouring cells, following
//     ** appropriate directions of travel and writing into
//     ** scratch space grid */
//     t_speed *const tmp_cell = getcelladdr(ii,jj,tmp_cells0,tmp_cells1,params.nyhalf,params.nx);
//     //Reverse the operation such that after each iteration the current cell is fully updated
//     //and hence the loop can be merged with the next step
//     if(0 == getcellval(ii,jj,obstacles0,obstacles1,params.nyhalf,params.nx)){
//         double local_density = tmp_cell->speeds[0] = getcellspeed(ii,jj,0,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[1] = getcellspeed(ii,x_w,1,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[2] = getcellspeed(y_s,jj,2,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[3] = getcellspeed(ii,x_e,3,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[4] = getcellspeed(y_n,jj,4,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[5] = getcellspeed(y_s,x_w,5,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[6] = getcellspeed(y_s,x_e,6,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[7] = getcellspeed(y_n,x_e,7,cells0,cells1,params.nyhalf,params.nx);
//         local_density += tmp_cell->speeds[8] = getcellspeed(y_n,x_w,8,cells0,cells1,params.nyhalf,params.nx);
//         //double local_density = 0.0;
//         /* compute local density total */
//         //for (unsigned int kk = 0; kk < NSPEEDS; kk++)
//         //{
//         //    local_density += tmp_cell->speeds[kk];
//         //}
//
//         /* compute x velocity component. NO DIVISION BY LOCAL DENSITY*/
//         double u_x = tmp_cell->speeds[1]
//                     + tmp_cell->speeds[5]
//                     + tmp_cell->speeds[8]
//                     - tmp_cell->speeds[3]
//                     - tmp_cell->speeds[6]
//                     - tmp_cell->speeds[7];
//         /* compute y velocity component. NO DIVISION BY LOCAL DENSITY */
//         double u_y = tmp_cell->speeds[2]
//                     + tmp_cell->speeds[5]
//                     + tmp_cell->speeds[6]
//                     - tmp_cell->speeds[4]
//                     - tmp_cell->speeds[7]
//                     - tmp_cell->speeds[8];
//
// //EQUATIONS ARE VERY DIFFERENT BUT STILL DO THE SAME THING.
//         const double u_x_sq = u_x * u_x;
//         const double u_y_sq = u_y * u_y;
//         const double u_xy   = u_x + u_y;
//         const double u_xy2  = u_x - u_y;
//         const double ld_sq  = local_density * local_density;
//         const double c_sq_ld_2 = twooverthree * local_density;
//         /* velocity squared */
//         const double u_sq = u_x_sq + u_y_sq;
//         const double ldinv = 1.0/local_density;
//         const double ldinvomega = ldinv*params.omega;
//         /* equilibrium densities */
//         double d_equ[NSPEEDS];
//         /* zero velocity density: weight w0 */
//         d_equ[0] = w0 * (2*ld_sq-3*u_sq) * ldinvomega;
//         /* axis speeds: weight w1 */
//         d_equ[1] = w1 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_x
//                             + u_x_sq - u_sq*c_sq ) * ldinvomega;
//         d_equ[2] = w1 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_y
//                             + u_y_sq - u_sq*c_sq ) * ldinvomega;
//         d_equ[3] = w1 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_x
//                             + u_x_sq - u_sq*c_sq ) * ldinvomega;
//         d_equ[4] = w1 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_y
//                             + u_y_sq - u_sq*c_sq ) * ldinvomega;
//         /* diagonal speeds: weight w2 */
//         d_equ[5] = w2 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_xy
//                             + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
//         d_equ[6] = w2 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_xy2
//                             + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;
//         d_equ[7] = w2 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_xy
//                             + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
//         d_equ[8] = w2 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_xy2
//                             + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;
//
//         //printf("%d : ",ii);
//         /* relaxation step */
//         for (unsigned int kk = 0; kk < NSPEEDS; kk++)
//         {
//             //printf("%lf  ",tmp_cell->speeds[kk]);
//             tmp_cell->speeds[kk] = tmp_cell->speeds[kk]*oneminusomega;
//             tmp_cell->speeds[kk] += d_equ[kk];
//             //local_density += tmp_cell->speeds[kk];
//         }
//
//         tot_u += sqrt(u_x*u_x + u_y*u_y) * ldinv;
//     }
//     else{
//         tmp_cell->speeds[0] = getcellspeed(ii,jj,0,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[3] = getcellspeed(ii,x_w,1,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[4] = getcellspeed(y_s,jj,2,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[1] = getcellspeed(ii,x_e,3,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[2] = getcellspeed(y_n,jj,4,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[7] = getcellspeed(y_s,x_w,5,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[8] = getcellspeed(y_s,x_e,6,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[5] = getcellspeed(y_n,x_e,7,cells0,cells1,params.nyhalf,params.nx);
//         tmp_cell->speeds[6] = getcellspeed(y_n,x_w,8,cells0,cells1,params.nyhalf,params.nx);
//     }
//   }
//   //}
//   return tot_u;
// }

inline double timestep(const t_param params, t_speed* restrict cells0, t_speed* restrict cells1, t_speed* restrict tmp_cells0,
              t_speed* restrict tmp_cells1, int* restrict obstacles0, int* restrict obstacles1, int tid)
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

  //int tid = omp_get_thread_num();
  int start = tid * (params.ny/NUMTHREADS);
  int end   = (tid+1) * (params.ny/NUMTHREADS);
  //#pragma omp for nowait
  for (unsigned int ii = start; ii < end; ii++)
  {
    if (ii==0 || ii==(params.nyhalf-1) || ii==params.nyhalf || ii==(params.ny-1) ) continue; //special cases. handle them elsewhere
    t_speed* restrict cells = NULL;
    t_speed* restrict tmp_cells = NULL;
    int* restrict obstacles = NULL;
    int qq = 0;
    if(ii<params.nyhalf){
        cells = cells0;
        tmp_cells = tmp_cells0;
        obstacles = obstacles0;
        qq = ii;
    }
    else{
        cells = cells1;
        tmp_cells = tmp_cells1;
        obstacles = obstacles1;
        qq = ii - params.nyhalf;
    }
    int y_n = qq + 1;
    int y_s = qq - 1;
    for(unsigned int jj = 0; jj < params.nx; jj+=VECSIZE){
        /* determine indices of axis-direction neighbours
        ** respecting periodic boundary conditions (wrap around) */
        double tmp[VECSIZE*NSPEEDS] __attribute__((aligned(32)));
        if (ii==0 || ii==(params.nyhalf-1) || ii==params.nyhalf || ii==(params.ny-1) ){
          #pragma vector aligned
          for(int k=0;k<VECSIZE;k++){
              int x = jj+k;
              int x_e = x + 1;
              if(x_e >= params.nx) x_e -= params.nx;
              int x_w = x-1;
              if(x==0) x_w = params.nx-1;
              tmp[VECSIZE*0+k] = getcellspeed(ii,x,0,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*1+k] = getcellspeed(ii,x_w,1,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*2+k] = getcellspeed(y_s,x,2,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*3+k] = getcellspeed(ii,x_e,3,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*4+k] = getcellspeed(y_n,x,4,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*5+k] = getcellspeed(y_s,x_w,5,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*6+k] = getcellspeed(y_s,x_e,6,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*7+k] = getcellspeed(y_n,x_e,7,cells0,cells1,params.nyhalf,params.nx);
              tmp[VECSIZE*8+k] = getcellspeed(y_n,x_w,8,cells0,cells1,params.nyhalf,params.nx);
          }
        }
        else{
          #pragma vector aligned
          for(int k=0;k<VECSIZE;k++){
              int x = jj+k;
              int x_e = x + 1;
              if(x_e >= params.nx) x_e -= params.nx;
              int x_w = (x == 0) ? (params.nx - 1) : (x-1);
              tmp[VECSIZE*0+k] = cells[qq * params.nx + x].speeds[0];
              tmp[VECSIZE*1+k] = cells[qq * params.nx + x_w].speeds[1];
              tmp[VECSIZE*2+k] = cells[y_s * params.nx + x].speeds[2];
              tmp[VECSIZE*3+k] = cells[qq * params.nx + x_e].speeds[3];
              tmp[VECSIZE*4+k] = cells[y_n * params.nx + x].speeds[4];
              tmp[VECSIZE*5+k] = cells[y_s * params.nx + x_w].speeds[5];
              tmp[VECSIZE*6+k] = cells[y_s * params.nx + x_e].speeds[6];
              tmp[VECSIZE*7+k] = cells[y_n * params.nx + x_e].speeds[7];
              tmp[VECSIZE*8+k] = cells[y_n * params.nx + x_w].speeds[8];
          }
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
            obst+=obstacles[qq*params.nx+jj+k];
        }

        if(!obst){
            #pragma vector aligned
            for(int k=0;k<VECSIZE;k++){
                tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                tot_u += sqrt(u_sq[k]) * densinv[k];
            }
        }
        else{

        #pragma vector aligned
        for(int k=0;k<VECSIZE;k++){
            if(!obstacles[qq * params.nx +jj +k]){
                tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
                tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
                tot_u += sqrt(u_sq[k]) * densinv[k];
            }
            else{
                tmp_cells[qq * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k];
                tmp_cells[qq * params.nx + jj + k].speeds[3] = tmp[VECSIZE*1+k];
                tmp_cells[qq * params.nx + jj + k].speeds[4] = tmp[VECSIZE*2+k];
                tmp_cells[qq * params.nx + jj + k].speeds[1] = tmp[VECSIZE*3+k];
                tmp_cells[qq * params.nx + jj + k].speeds[2] = tmp[VECSIZE*4+k];
                tmp_cells[qq * params.nx + jj + k].speeds[7] = tmp[VECSIZE*5+k];
                tmp_cells[qq * params.nx + jj + k].speeds[8] = tmp[VECSIZE*6+k];
                tmp_cells[qq * params.nx + jj + k].speeds[5] = tmp[VECSIZE*7+k];
                tmp_cells[qq * params.nx + jj + k].speeds[6] = tmp[VECSIZE*8+k];

            }
        }
        }
    }
  }

  return tot_u;
}

double av_velocity(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1)
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
      if (0 == getcellval(ii,jj,obstacles0,obstacles1,params.nyhalf,params.nx))
      {
        /* local density total */
        double local_density = 0.0;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += getcellspeed(ii,jj,kk,cells0,cells1,params.nyhalf,params.nx);
        }
        /* x-component of velocity */
        t_speed* cell = getcelladdr(ii,jj,cells0,cells1,params.nyhalf,params.nx);
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

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr0, t_speed** cells_ptr1, t_speed** tmp_cells_ptr0,
               t_speed** tmp_cells_ptr1, int** obstacles_ptr0, int** obstacles_ptr1, double** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

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
  params->nyhalf = params->ny/2;

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
  /* Fortunately, blue crystal's compute  */
  #pragma omp parallel
  {
  int tid = omp_get_thread_num();
  if(tid == 0){
    *cells_ptr0 = (t_speed*)malloc(sizeof(t_speed) * (params->nyhalf * params->nx));

    if (*cells_ptr0 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr0 = (t_speed*)malloc(sizeof(t_speed) * (params->nyhalf * params->nx));

    if (*tmp_cells_ptr0 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr0 = (int*)malloc(sizeof(int) * (params->nyhalf * params->nx));

    if (*obstacles_ptr0 == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  }
  if(tid == 8){
    *cells_ptr1 = (t_speed*)malloc(sizeof(t_speed) * (params->nyhalf * params->nx));

    if (*cells_ptr1 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr1 = (t_speed*)malloc(sizeof(t_speed) * (params->nyhalf * params->nx));

    if (*tmp_cells_ptr1 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr1 = (int*)malloc(sizeof(int) * (params->nyhalf * params->nx));

    if (*obstacles_ptr1 == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  }

  }


  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;

  for (unsigned int ii = 0; ii < params->ny; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
    {
      t_speed* cell = getcelladdr(ii,jj,(*cells_ptr0),(*cells_ptr1),params->nyhalf,params->nx);
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
  for (unsigned int ii = 0; ii < params->ny; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
    {
      int* cell = getcelladdr(ii,jj,(*obstacles_ptr0),(*obstacles_ptr1),params->nyhalf,params->nx);
      *cell = 0;
    }
  }

  /* open the obstacle data file */
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
    if(0 == getcellval(yy,xx,(*obstacles_ptr0),(*obstacles_ptr1),params->nyhalf,params->nx))
        numOfFreeCells--;
    int* cell = getcelladdr(yy,xx,(*obstacles_ptr0),(*obstacles_ptr1),params->nyhalf,params->nx);
    *cell = blocked;
  }
  params->free_cells_inv = 1.0/numOfFreeCells;

  /* and close the file */
  fclose(fp);

  //preprocess_obstacles(*obstacles_ptr,*params);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);

  return EXIT_SUCCESS;
}


int finalise(const t_param* params, t_speed** cells_ptr0, t_speed** cells_ptr1, t_speed** tmp_cells_ptr0,
             t_speed** tmp_cells_ptr1, int** obstacles_ptr0, int** obstacles_ptr1,  double** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr0);
  *cells_ptr0 = NULL;

  free(*cells_ptr1);
  *cells_ptr1 = NULL;

  free(*tmp_cells_ptr0);
  *tmp_cells_ptr0 = NULL;

  free(*tmp_cells_ptr1);
  *tmp_cells_ptr1 = NULL;

  free(*obstacles_ptr0);
  *obstacles_ptr0 = NULL;

  free(*obstacles_ptr1);
  *obstacles_ptr1 = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells0, cells1, obstacles0, obstacles1) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells0, t_speed* cells1)
{
  double total = 0.0;  /* accumulator */

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      for (unsigned int kk = 0; kk < NSPEEDS; kk++)
      {
        total += getcellspeed(ii,jj,kk,cells0,cells1,params.nyhalf,params.nx);
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells0, t_speed* cells1, int* obstacles0, int* obstacles1, double* av_vels)
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
      if (1 == getcellval(ii,jj,obstacles0,obstacles1,params.nyhalf,params.nx))
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;
        t_speed* cell = getcelladdr(ii,jj,cells0,cells1,params.nyhalf,params.nx);

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
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, getcellval(ii,jj,obstacles0,obstacles1,params.nyhalf,params.nx));
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
