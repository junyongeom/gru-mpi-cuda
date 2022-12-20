#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

// CUDA call
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

// Defined in main.cpp
extern int mpi_rank, mpi_size;
float *my_random_floats = nullptr;
char *my_output = nullptr;

// static int Mbegin, Mend;

// You can modify the data structure as you want
struct Tensor {

  /* Alloc memory */
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, buf_, n * sizeof(float));
  }

  ~Tensor() {
    if (buf != nullptr)
      free(buf);
  }

  void set_zero() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0;
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf = nullptr;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};

/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

/* Operations */

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
 __global__ void embedding(float *input, float *weight, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n){
    int x = (int)(*input);
    output[tid] = weight[x * n + tid];
  }
}
// void embedding(Tensor *input, Tensor *weight, Tensor *output) {
//   size_t n = weight->shape[1];
//   for (size_t i = 0; i < n; i++) {
//     int x = (int)input->buf[0];
//     output->buf[i] = weight->buf[x * n + i];
//   }
// }

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
__global__  void elemwise_add(float *input1, float *input2, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // size_t sn = input1->num_elem();
  if (tid < n){
    output[tid] = input1[tid] + input2[tid];
  }
}
// void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
//   size_t sn = input1->num_elem();
//   for (size_t i = 0; i < sn; i++) {
//     output->buf[i] = input1->buf[i] + input2->buf[i];
//   }
// }


/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
 __global__ void elemwise_oneminus(float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n){
    float x = input[tid];
    output[tid] = 1.0 - x;
  }
}
// void elemwise_oneminus(Tensor *input, Tensor *output) {
//   size_t n = input->num_elem();
//   for (size_t i = 0; i < n; i++) {
//     float x = input->buf[i];
//     output->buf[i] = 1.0 - x;
//   }
// }

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
__global__ void elemwise_mul(float *input1, float *input2, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n){
    output[tid] = input1[tid] * input2[tid];
  }
}
// void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
//   size_t sn = input1->num_elem();
//   for (size_t i = 0; i < sn; i++) {
//     output->buf[i] = input1->buf[i] * input2->buf[i];
//   }
// }

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
__global__ void elemwise_tanh(float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n){
    float x = input[tid];
    output[tid] = tanhf(x);
  }
}
// void elemwise_tanh(Tensor *input, Tensor *output) {
//   size_t n = input->num_elem();
//   for (size_t i = 0; i < n; i++) {
//     float x = input->buf[i];
//     output->buf[i] = tanhf(x);
//   }
// }

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
 __global__ void elemwise_sigmoid(float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n) {
    float x = input[tid];
    output[tid] = 1.0 / (1.0 + expf(-x));
  }
}
// void elemwise_sigmoid(Tensor *input, Tensor *output) {
//   size_t n = input->num_elem();
//   for (size_t i = 0; i < n; i++) {
//     float x = input->buf[i];
//     output->buf[i] = 1.0 / (1.0 + expf(-x));
//   }
// }

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */ 
 __global__ void matvec(float *input1, float *input2, float *output, int N_, int K_) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < N_){
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1[tid * K_ + j] * input2[j];
    }
    output[tid] = c;
  }
}
// void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
//   size_t N_ = input1->shape[0];
//   size_t K_ = input1->shape[1];
//   for (size_t i = 0; i < N_; i++) {
//     float c = 0.0;
//     for (size_t j = 0; j < K_; j++) {
//       c += input1->buf[i * K_ + j] * input2->buf[j];
//     }
//     output->buf[i] = c;
//   }
// }

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
__global__ void set_zero(float* src){
  src[0] = 0;
}
//  __global__ void softmax_sum(float *input, float *sum, int n){
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(tid < n){
//     // sum[0] += expf(input[tid]);
//     float anysum = atomicAdd(&sum[0], input[tid]);
//   }
//  }

 __global__ void softmax(float *input, float *sum, float* output, int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n){
    float anysum = atomicAdd(&sum[0], expf(input[tid]));
    output[tid] = expf(input[tid]) / sum[0];
  }
 }

// void softmax(Tensor *input, Tensor *output) {
//   size_t n = input->num_elem();
//   float sum = 0.0;
//   for (size_t i = 0; i < n; i++) {
//     float x = input->buf[i];
//     sum += expf(x);
//   }
//   for (size_t i = 0; i < n; i++) {
//     float x = input->buf[i];
//     output->buf[i] = expf(x) / sum;
//   }
// }

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
/*** model parameter gpu upload ***/
/* Network parameters */
// Tensor *character_embedding;
// Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
// Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
// Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
// Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
// Tensor *W_fc, *b_fc;
// Tensor *rfloats;
static float *gcharacter_embedding;
static float *gW_ir0, *gW_iz0, *gW_in0, *gW_ir1, *gW_iz1, *gW_in1;
static float *gW_hr0, *gW_hz0, *gW_hn0, *gW_hr1, *gW_hz1, *gW_hn1;
static float *gb_ir0, *gb_iz0, *gb_in0, *gb_ir1, *gb_iz1, *gb_in1;
static float *gb_hr0, *gb_hz0, *gb_hn0, *gb_hr1, *gb_hz1, *gb_hn1;
static float *gW_fc, *gb_fc;
// static float *grfloats;
static float *a_d, *b_d, *c_d;
static float *gr0, *gz0, *gn0, *gr1, *gz1, *gn1, *gf;
static float *ginput, *gemb_out, *ghidden0, *ghidden1, *gchar_prob, *gsum;

void namegen_initialize(int N, int rng_seed, char *parameter_fname) {
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  printf("(%s) Hello world, rank %d out of %d\n", processor_name, mpi_rank,
         mpi_size);
  
  /* Only the root process reads the parameter */
  // if (mpi_rank == 0) {
    size_t parameter_binary_size = 0;
    float *parameter =
        (float *)read_binary(parameter_fname, &parameter_binary_size);

    /* Network parameters */
    character_embedding =
        new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

    W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
    W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
    W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
    W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
    W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
    W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

    W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
    W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
    W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
    W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
    W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
    W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

    b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
    b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
    b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
    b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
    b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
    b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

    b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
    b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
    b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
    b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
    b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
    b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

    W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
    b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);

    /* input, activations, output, etc. */
    input = new Tensor({1});
    emb_out = new Tensor({EMBEDDING_DIM});

    hidden0 = new Tensor({HIDDEN_DIM});
    hidden1 = new Tensor({HIDDEN_DIM});

    r0 = new Tensor({HIDDEN_DIM});
    r1 = new Tensor({HIDDEN_DIM});
    z0 = new Tensor({HIDDEN_DIM});
    z1 = new Tensor({HIDDEN_DIM});
    n0 = new Tensor({HIDDEN_DIM});
    n1 = new Tensor({HIDDEN_DIM});
    f = new Tensor({NUM_CHAR});

    rtmp00 = new Tensor({HIDDEN_DIM});
    rtmp01 = new Tensor({HIDDEN_DIM});
    rtmp02 = new Tensor({HIDDEN_DIM});
    rtmp03 = new Tensor({HIDDEN_DIM});
    rtmp04 = new Tensor({HIDDEN_DIM});
    rtmp10 = new Tensor({HIDDEN_DIM});
    rtmp11 = new Tensor({HIDDEN_DIM});
    rtmp12 = new Tensor({HIDDEN_DIM});
    rtmp13 = new Tensor({HIDDEN_DIM});
    rtmp14 = new Tensor({HIDDEN_DIM});

    ztmp00 = new Tensor({HIDDEN_DIM});
    ztmp01 = new Tensor({HIDDEN_DIM});
    ztmp02 = new Tensor({HIDDEN_DIM});
    ztmp03 = new Tensor({HIDDEN_DIM});
    ztmp04 = new Tensor({HIDDEN_DIM});
    ztmp10 = new Tensor({HIDDEN_DIM});
    ztmp11 = new Tensor({HIDDEN_DIM});
    ztmp12 = new Tensor({HIDDEN_DIM});
    ztmp13 = new Tensor({HIDDEN_DIM});
    ztmp14 = new Tensor({HIDDEN_DIM});

    ntmp00 = new Tensor({HIDDEN_DIM});
    ntmp01 = new Tensor({HIDDEN_DIM});
    ntmp02 = new Tensor({HIDDEN_DIM});
    ntmp03 = new Tensor({HIDDEN_DIM});
    ntmp04 = new Tensor({HIDDEN_DIM});
    ntmp05 = new Tensor({HIDDEN_DIM});
    ntmp10 = new Tensor({HIDDEN_DIM});
    ntmp11 = new Tensor({HIDDEN_DIM});
    ntmp12 = new Tensor({HIDDEN_DIM});
    ntmp13 = new Tensor({HIDDEN_DIM});
    ntmp14 = new Tensor({HIDDEN_DIM});
    ntmp15 = new Tensor({HIDDEN_DIM});

    htmp00 = new Tensor({HIDDEN_DIM});
    htmp01 = new Tensor({HIDDEN_DIM});
    htmp02 = new Tensor({HIDDEN_DIM});
    htmp10 = new Tensor({HIDDEN_DIM});
    htmp11 = new Tensor({HIDDEN_DIM});
    htmp12 = new Tensor({HIDDEN_DIM});

    rfloats = new Tensor({N * MAX_LEN});
    ftmp0 = new Tensor({NUM_CHAR});
    char_prob = new Tensor({NUM_CHAR});
  // } 

    size_t sn = W_ir0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_ir0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_ir0, W_ir0->buf, sn*sizeof(float), cudaMemcpyHostToDevice));   

    sn = W_iz0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_iz0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_iz0, W_iz0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_in0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_in0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_in0, W_in0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_ir1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_ir1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_ir1, W_ir1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_iz1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_iz1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_iz1, W_iz1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_in1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_in1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_in1, W_in1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hr0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hr0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hr0, W_hr0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hz0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hz0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hz0, W_hz0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hn0->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hn0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hn0, W_hn0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hr1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hr1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hr1, W_hr1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hz1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hz1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hz1, W_hz1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_hn1->num_elem();
    CUDA_CALL(cudaMalloc(&gW_hn1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_hn1, W_hn1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_ir0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_ir0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_ir0, b_ir0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_iz0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_iz0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_iz0, b_iz0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_in0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_in0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_in0, b_in0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_ir1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_ir1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_ir1, b_ir1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_iz1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_iz1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_iz1, b_iz1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_in1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_in1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_in1, b_in1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hr0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hr0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hr0, b_hr0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hz0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hz0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hz0, b_hz0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hn0->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hn0, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hn0, b_hn0->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hr1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hr1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hr1, b_hr1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hz1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hz1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hz1, b_hz1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_hn1->num_elem();
    CUDA_CALL(cudaMalloc(&gb_hn1, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_hn1, b_hn1->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = W_fc->num_elem();
    CUDA_CALL(cudaMalloc(&gW_fc, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gW_fc, W_fc->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_fc->num_elem();
    CUDA_CALL(cudaMalloc(&gb_fc, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_fc, b_fc->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = b_fc->num_elem();
    CUDA_CALL(cudaMalloc(&gb_fc, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gb_fc, b_fc->buf, sn*sizeof(float), cudaMemcpyHostToDevice));

    sn = character_embedding->num_elem();
    CUDA_CALL(cudaMalloc(&gcharacter_embedding, sn*sizeof(float)));
    CUDA_CALL(cudaMemcpy(gcharacter_embedding, character_embedding->buf, sn*sizeof(float), cudaMemcpyHostToDevice)); 

    sn = rtmp00->num_elem();
    CUDA_CALL(cudaMalloc(&a_d, sn*sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d, sn*sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d, sn*sizeof(float)));

    sn = r0->num_elem();
    CUDA_CALL(cudaMalloc(&gr0, sn*sizeof(float)));
    sn = r1->num_elem();
    CUDA_CALL(cudaMalloc(&gr1, sn*sizeof(float)));
    sn = z0->num_elem();
    CUDA_CALL(cudaMalloc(&gz0, sn*sizeof(float)));
    sn = z1->num_elem();
    CUDA_CALL(cudaMalloc(&gz1, sn*sizeof(float)));
    sn = n0->num_elem();
    CUDA_CALL(cudaMalloc(&gn0, sn*sizeof(float)));
    sn = n1->num_elem();
    CUDA_CALL(cudaMalloc(&gn1, sn*sizeof(float)));
    sn = f->num_elem();
    CUDA_CALL(cudaMalloc(&gf, sn*sizeof(float)));
    sn = input->num_elem();
    CUDA_CALL(cudaMalloc(&ginput, sn*sizeof(float)));
    sn = emb_out->num_elem();
    CUDA_CALL(cudaMalloc(&gemb_out, sn*sizeof(float)));
    sn = hidden0->num_elem();
    CUDA_CALL(cudaMalloc(&ghidden0, sn*sizeof(float)));
    sn = hidden1->num_elem();
    CUDA_CALL(cudaMalloc(&ghidden1, sn*sizeof(float)));
    sn = char_prob->num_elem();
    CUDA_CALL(cudaMalloc(&gchar_prob, sn*sizeof(float)));
    CUDA_CALL(cudaMalloc(&gsum, sizeof(float)));

  
    MPI_Barrier(MPI_COMM_WORLD);   
  //   else {
  // }
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {
  int JPP = N / mpi_size; // jops per process
  int my_start = JPP * mpi_rank;
  int my_end = my_start + JPP;
  
  my_random_floats = (float *)malloc(JPP * MAX_LEN * sizeof(float));
  my_output = (char *)malloc(JPP * (MAX_LEN + 1) * sizeof(char));
  
  // MPI_Barrier(MPI_COMM_WORLD); //
  MPI_Scatter(random_floats, JPP*MAX_LEN, MPI_FLOAT, my_random_floats, JPP*MAX_LEN, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if(mpi_rank == 0){
    // memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
    memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));
  }
  memcpy(rfloats->buf, my_random_floats, JPP * MAX_LEN * sizeof(float));
  memset(my_output, 0, JPP * (MAX_LEN + 1) * sizeof(char));

  size_t sn = rtmp00->num_elem();
  int block_size = 512;
  int grid_size = ((sn + block_size) / block_size);
  /* Generate N names */
  for (int n = my_start; n < my_end; n++) {
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[0] = SOS;
    hidden0->set_zero();
    hidden1->set_zero();
  
    sn = hidden0->num_elem();
    CUDA_CALL(cudaMemcpy(ghidden0, hidden0->buf, sn*sizeof(float), cudaMemcpyHostToDevice));
    sn = hidden1->num_elem();
    CUDA_CALL(cudaMemcpy(ghidden1, hidden1->buf, sn*sizeof(float), cudaMemcpyHostToDevice));

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      // embedding(input, character_embedding, emb_out);  
      sn = character_embedding->shape[1];
      grid_size = ((sn + block_size) / block_size);
      // size_t inputsize = input->num_elem();
      // printf("%d\n inputsize\n", inputsize);
      CUDA_CALL(cudaMemcpy(ginput, &input->buf[0], sizeof(float), cudaMemcpyHostToDevice));
      embedding<<<grid_size,block_size>>>(ginput, gcharacter_embedding, gemb_out, sn);
      // CUDA_CALL(cudaDeviceSynchronize());
      /* First layer r */
      // matvec(W_ir0, emb_out, rtmp00);
      size_t N__ = W_ir0->shape[0];
      size_t K__ = W_ir0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_ir0, gemb_out, a_d, N__, K__); // a_d rtmp00
      // matvec(W_hr0, hidden0, rtmp01);
      N__ = W_hr0->shape[0];
      K__ = W_hr0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hr0, ghidden0, b_d, N__, K__); // b_d rtmp01
      // CUDA_CALL(cudaDeviceSynchronize());
      // elemwise_add(rtmp00, b_ir0, rtmp02);  
      sn = rtmp00->num_elem(); 
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_ir0, c_d, sn); // cd=rtmp02
      // elemwise_add(rtmp02, rtmp01, rtmp03);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(c_d, b_d, a_d, sn); //ad=rtmp03
      // elemwise_add(rtmp03, b_hr0, rtmp04);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_hr0, c_d, sn); //cd=rtmp04
      // CUDA_CALL(cudaMemcpy(rtmp04->buf, c_d,  sn*sizeof(float), cudaMemcpyDeviceToHost));
      // elemwise_sigmoid(rtmp04, r0); // 1024 num elem
      elemwise_sigmoid<<<grid_size,block_size>>>(c_d, gr0, sn); 

      /* First layer z */
      // matvec(W_iz0, emb_out, ztmp00);
      N__ = W_iz0->shape[0];
      K__ = W_iz0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_iz0, gemb_out, a_d, N__, K__); // a_d ztmp00
      // matvec(W_hz0, hidden0, ztmp01);
      N__ = W_hz0->shape[0];
      K__ = W_hz0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hz0, ghidden0, b_d, N__, K__); // b_d ztmp01

      // elemwise_add(ztmp00, b_iz0, ztmp02);
      // CUDA_CALL(cudaDeviceSynchronize());
      sn = ztmp00->num_elem();
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_iz0, c_d, sn); // cd ztmp02
      // elemwise_add(ztmp02, ztmp01, ztmp03);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(c_d, b_d, a_d, sn); //ad=ztmp03
      // elemwise_add(ztmp03, b_hz0, ztmp04);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_hz0, c_d, sn); //cd=ztmp04
      // elemwise_sigmoid(ztmp04, z0); //  1024
      elemwise_sigmoid<<<grid_size,block_size>>>(c_d, gz0, sn); 

      /* First layer n */
      // matvec(W_in0, emb_out, ntmp00);
      // CUDA_CALL(cudaDeviceSynchronize());
      N__ = W_in0->shape[0];
      K__ = W_in0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_in0, gemb_out, a_d, N__, K__); // a_d ntmp00
      // elemwise_add(ntmp00, b_in0, ntmp01);
      sn = ntmp00->num_elem();
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_in0, b_d, sn); //bd ntmp01
      // CUDA_CALL(cudaDeviceSynchronize());
      // matvec(W_hn0, hidden0, ntmp02);
      N__ = W_hn0->shape[0];
      K__ = W_hn0->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hn0, ghidden0, a_d, N__, K__); // a_d ntmp02
      // elemwise_add(ntmp02, b_hn0, ntmp03);
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_hn0, c_d, sn); //cd ntmp03
      // elemwise_mul(r0, ntmp03, ntmp04);
      elemwise_mul<<<grid_size,block_size>>>(gr0, c_d, a_d, sn); // a_d ntmp04
      // elemwise_add(ntmp01, ntmp04, ntmp05);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(b_d, a_d, c_d, sn); //cd 05
      // elemwise_tanh(ntmp05, n0);
      elemwise_tanh<<<grid_size,block_size>>>(c_d, gn0, sn);

      /* First layer h (hidden) */
      // elemwise_oneminus(z0, htmp00);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_oneminus<<<grid_size,block_size>>>(gz0, a_d, sn); // a_d htmp00
      // elemwise_mul(htmp00, n0, htmp01);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_mul<<<grid_size,block_size>>>(a_d, gn0, b_d, sn); // b_d htmp01
      // elemwise_mul(z0, hidden0, htmp02);
      elemwise_mul<<<grid_size,block_size>>>(gz0, ghidden0, c_d, sn); // c_d htmp02
      // elemwise_add(htmp01, htmp02, hidden0); // h0 1024
      // printf("hidden %d z0 %d\n", hidden0->num_elem(), z0->num_elem() );
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(b_d, c_d, ghidden0, sn); 
      // CUDA_CALL(cudaMemcpy(hidden0->buf, a_d,  sn*sizeof(float), cudaMemcpyDeviceToHost));
      
      //////

      /* Second layer r */
      // matvec(W_ir1, hidden0, rtmp10);
      N__ = W_ir1->shape[0];
      K__ = W_ir1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_ir1, ghidden0, a_d, N__, K__); // a_d rtmp10
      // matvec(W_hr1, hidden1, rtmp11);
      N__ = W_hr1->shape[0];
      K__ = W_hr1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hr1, ghidden1, b_d, N__, K__); // b_d rtmp11
      // elemwise_add(rtmp10, b_ir1, rtmp12);
      // CUDA_CALL(cudaDeviceSynchronize());
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_ir1, c_d, sn); // c_d rtmp12
      // elemwise_add(rtmp12, rtmp11, rtmp13);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(c_d, b_d, a_d, sn); // a_d rtmp13
      // elemwise_add(rtmp13, b_hr1, rtmp14);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_hr1, c_d, sn); // c_d rtmp14
      // elemwise_sigmoid(rtmp14, r1); //r1 1024
      elemwise_sigmoid<<<grid_size,block_size>>>(c_d, gr1, sn); 
      
      /* Second layer z */
      // matvec(W_iz1, hidden0, ztmp10);
      N__ = W_iz1->shape[0]; 
      K__ = W_iz1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_iz1, ghidden0, a_d, N__, K__); // a_d ztmp10
      // matvec(W_hz1, hidden1, ztmp11);
      N__ = W_hz1->shape[0];
      K__ = W_hz1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hz1, ghidden1, b_d, N__, K__); // b_d ztmp11
      // elemwise_add(ztmp10, b_iz1, ztmp12);
      // CUDA_CALL(cudaDeviceSynchronize());
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_iz1, c_d, sn); // c_d ztmp12
      // elemwise_add(ztmp12, ztmp11, ztmp13);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(c_d, b_d, a_d, sn); // a_d ztmp13
      // elemwise_add(ztmp13, b_hz1, ztmp14);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_hz1, c_d, sn); // c_d ztmp14
      // elemwise_sigmoid(ztmp14, z1);
      elemwise_sigmoid<<<grid_size,block_size>>>(c_d, gz1, sn); 

      /* Second layer n */
      // matvec(W_in1, hidden0, ntmp10); 
      N__ = W_in1->shape[0];
      K__ = W_in1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_in1, ghidden0, a_d, N__, K__); // a_d ntmp10
      // elemwise_add(ntmp10, b_in1, ntmp11);
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(a_d, gb_in1, c_d, sn); // c_d ntmp11
      // matvec(W_hn1, hidden1, ntmp12);
      N__ = W_hn1->shape[0];
      K__ = W_hn1->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_hn1, ghidden1, b_d, N__, K__); // b_d ntmp12
      // elemwise_add(ntmp12, b_hn1, ntmp13);
      // CUDA_CALL(cudaDeviceSynchronize());
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(b_d, gb_hn1, a_d, sn); // a_d ntmp13
      // elemwise_mul(r1, ntmp13, ntmp14);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_mul<<<grid_size,block_size>>>(gr1, a_d, b_d, sn); // b_d ntmp14
      // elemwise_add(ntmp11, ntmp14, ntmp15);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(c_d, b_d, a_d, sn); // a_d ntmp15
      // elemwise_tanh(ntmp15, n1);
      elemwise_tanh<<<grid_size,block_size>>>(a_d, gn1, sn);

      /* Second layer h (hidden) */
      // elemwise_oneminus(z1, htmp10);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_oneminus<<<grid_size,block_size>>>(gz1, a_d, sn); // a_d htmp10
      // elemwise_mul(htmp10, n1, htmp11);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_mul<<<grid_size,block_size>>>(a_d, gn1, b_d, sn); // b_d htmp11
      // elemwise_mul(z1, hidden1, htmp12);
      elemwise_mul<<<grid_size,block_size>>>(gz1, ghidden1, c_d, sn); // c_d htmp12
      // elemwise_add(htmp11, htmp12, hidden1);
      // CUDA_CALL(cudaDeviceSynchronize());
      elemwise_add<<<grid_size,block_size>>>(b_d, c_d, ghidden1, sn); 
    
      /* Fully connected layer */
      // matvec(W_fc, hidden1, ftmp0);
      N__ = W_fc->shape[0];
      K__ = W_fc->shape[1];
      grid_size = ((N__ + block_size) / block_size);
      matvec<<<grid_size,block_size>>>(gW_fc, ghidden1, b_d, N__, K__); // b_d ftmp0
      // elemwise_add(ftmp0, b_fc, f);  // f 256
      sn = ftmp0->num_elem();
      grid_size = ((sn + block_size) / block_size);
      elemwise_add<<<grid_size,block_size>>>(b_d, gb_fc, gf, sn); //       
      // /* Softmax */
      sn = f->num_elem();
      // CUDA_CALL(cudaMemcpy(f->buf, gf,  sn*sizeof(float), cudaMemcpyDeviceToHost));
      // softmax(f, char_prob);   
      set_zero<<<1,1>>>(gsum);
      grid_size = ((sn + block_size) / block_size);
      softmax<<<grid_size,block_size>>>(gf, gsum, gchar_prob, sn);
      CUDA_CALL(cudaMemcpy(char_prob->buf, gchar_prob,  sn*sizeof(float), cudaMemcpyDeviceToHost));
      
      /* Random select */
      int selected_char = random_select(char_prob, rfloats, (n-my_start) * MAX_LEN + l);
      my_output[(n-my_start) * (MAX_LEN + 1) + l ] = selected_char;
      input->buf[0] = selected_char;
      

      if (selected_char == EOS)
        break;
    }
    
  }
  
  // MPI_Barrier(MPI_COMM_WORLD);
  // printf("im rank %d my output %s\n", mpi_rank, my_output);
  MPI_Gather(my_output, JPP * (MAX_LEN+1), MPI_CHAR, output, JPP * (MAX_LEN+1), MPI_CHAR, 0, MPI_COMM_WORLD);
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  // if (mpi_rank == 0) {
    delete character_embedding;
    delete W_ir0;
    delete W_iz0;
    delete W_in0;
    delete W_ir1;
    delete W_iz1;
    delete W_in1;
    delete W_hr0;
    delete W_hz0;
    delete W_hn0;
    delete W_hr1;
    delete W_hz1;
    delete W_hn1;
    delete b_ir0;
    delete b_iz0;
    delete b_in0;
    delete b_ir1;
    delete b_iz1;
    delete b_in1;
    delete b_hr0;
    delete b_hz0;
    delete b_hn0;
    delete b_hr1;
    delete b_hz1;
    delete b_hn1;
    delete W_fc;
    delete b_fc;
    delete rfloats;

    delete input;
    delete emb_out;
    delete hidden0;
    delete hidden1;
    delete r0;
    delete r1;
    delete z0;
    delete z1;
    delete n0;
    delete n1;
    delete f;
    delete char_prob;
    delete rtmp00;
    delete rtmp01;
    delete rtmp02;
    delete rtmp03;
    delete rtmp04;
    delete rtmp10;
    delete rtmp11;
    delete rtmp12;
    delete rtmp13;
    delete rtmp14;
    delete ztmp00;
    delete ztmp01;
    delete ztmp02;
    delete ztmp03;
    delete ztmp04;
    delete ztmp10;
    delete ztmp11;
    delete ztmp12;
    delete ztmp13;
    delete ztmp14;
    delete ntmp00;
    delete ntmp01;
    delete ntmp02;
    delete ntmp03;
    delete ntmp04;
    delete ntmp05;
    delete ntmp10;
    delete ntmp11;
    delete ntmp12;
    delete ntmp13;
    delete ntmp14;
    delete ntmp15;
    delete htmp00;
    delete htmp01;
    delete htmp02;
    delete htmp10;
    delete htmp11;
    delete htmp12;
    delete ftmp0;
    free(my_random_floats);
    free(my_output);

    cudaFree(gcharacter_embedding);
    cudaFree(gW_ir0);
    cudaFree(gW_iz0);
    cudaFree(gW_in0);
    cudaFree(gW_ir1);
    cudaFree(gW_iz1);
    cudaFree(gW_in1);

    cudaFree(gW_hr0);
    cudaFree(gW_hz0);
    cudaFree(gW_hn0);
    cudaFree(gW_hr1);
    cudaFree(gW_hz1);
    cudaFree(gW_hn1);

    cudaFree(gb_ir0);
    cudaFree(gb_iz0);
    cudaFree(gb_in0);
    cudaFree(gb_ir1);
    cudaFree(gb_iz1);
    cudaFree(gb_in1);

    cudaFree(gb_hr0);
    cudaFree(gb_hz0);
    cudaFree(gb_hn0);
    cudaFree(gb_hr1);
    cudaFree(gb_hz1);
    cudaFree(gb_hn1);

    cudaFree(gr0);
    cudaFree(gr1);
    cudaFree(gn0);
    cudaFree(gn1);
    cudaFree(gz0);
    cudaFree(gz1);
    cudaFree(f);

    cudaFree(gW_fc);
    cudaFree(gb_fc);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaFree(ginput);
    cudaFree(gemb_out);
    cudaFree(ghidden0);
    cudaFree(ghidden1);
    cudaFree(gchar_prob);
    cudaFree(gsum);
  // }
}