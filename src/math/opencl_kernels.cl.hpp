// OpenCL Kernels for Matrix Operations
// Optimized for Intel Iris Xe Graphics

const char* OPENCL_KERNELS = R"CL(

// OPTIMIZATION: GPU-side embedding lookup kernel
// Eliminates need to download embeddings to CPU
// Input: token IDs (small upload), embeddings stay on GPU
// Output: embedded sequences ready for transformer
__kernel void embed_sequence(
    __global const float* token_embedding,
    __global const float* position_embedding,
    __global const int* token_ids,
    __global float* output,
    int seq_len,
    int d_model,
    int max_seq_len)
{
    int i = get_global_id(0);  // Token position
    int j = get_global_id(1);  // Embedding dimension
    
    if (i < seq_len && j < d_model) {
        int token_id = token_ids[i];
        int pos_idx = i % max_seq_len;
        
        // Combine token and position embeddings
        output[i * d_model + j] = token_embedding[token_id * d_model + j] 
                                + position_embedding[pos_idx * d_model + j];
    }
}

// Matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M, int K, int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication with local memory (tiled)
__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M, int K, int N)
{
    const int TILE_SIZE = 16;
    
    __local float A_tile[16][16];
    __local float B_tile[16][16];
    
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load tiles into local memory
        int a_col = t * TILE_SIZE + local_col;
        int b_row = t * TILE_SIZE + local_row;
        
        if (row < M && a_col < K)
            A_tile[local_row][local_col] = A[row * K + a_col];
        else
            A_tile[local_row][local_col] = 0.0f;
            
        if (b_row < K && col < N)
            B_tile[local_row][local_col] = B[b_row * N + col];
        else
            B_tile[local_row][local_col] = 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[local_row][k] * B_tile[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matrix addition: C = A + B
__kernel void add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        C[gid] = A[gid] + B[gid];
    }
}

// Matrix subtraction: C = A - B
__kernel void subtract(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        C[gid] = A[gid] - B[gid];
    }
}

// Scalar multiplication: B = A * scalar
__kernel void multiply_scalar(
    __global const float* A,
    __global float* B,
    float scalar,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        B[gid] = A[gid] * scalar;
    }
}

// Hadamard (element-wise) product: C = A ⊙ B
__kernel void hadamard(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        C[gid] = A[gid] * B[gid];
    }
}

// Matrix transpose: B = A^T
__kernel void transpose(
    __global const float* A,
    __global float* B,
    int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// ReLU activation: y = max(0, x)
__kernel void relu(
    __global const float* input,
    __global float* output,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = fmax(0.0f, input[gid]);
    }
}

// Tanh activation: y = tanh(x)
__kernel void tanh_activation(
    __global const float* input,
    __global float* output,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = tanh(input[gid]);
    }
}

// Sigmoid activation: y = 1 / (1 + exp(-x))
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}

// Softmax (row-wise for 2D matrices)
__kernel void softmax(
    __global const float* input,
    __global float* output,
    int rows, int cols)
{
    int row = get_global_id(0);
    
    if (row < rows) {
        __global const float* row_input = input + row * cols;
        __global float* row_output = output + row * cols;
        
        // Find max for numerical stability
        float max_val = row_input[0];
        for (int i = 1; i < cols; i++) {
            max_val = fmax(max_val, row_input[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float exp_val = exp(row_input[i] - max_val);
            row_output[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < cols; i++) {
            row_output[i] /= sum;
        }
    }
}

// Element-wise square root
__kernel void sqrt_op(
    __global const float* input,
    __global float* output,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = sqrt(input[gid]);
    }
}

// Element-wise power
__kernel void pow_op(
    __global const float* input,
    __global float* output,
    float exponent,
    int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        output[gid] = pow(input[gid], exponent);
    }
}

// Parallel reduction for sum (work-group based)
__kernel void sum_reduce(
    __global const float* input,
    __global float* partial_sums,
    __local float* scratch,
    int size)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    // Load into local memory
    scratch[lid] = (gid < size) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int offset = group_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work-group
    if (lid == 0) {
        partial_sums[get_group_id(0)] = scratch[0];
    }
}

)CL";
