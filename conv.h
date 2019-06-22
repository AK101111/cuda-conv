#pragma once
#include <array>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
static constexpr T expand(T hd){
    return hd;
}

template<typename T, typename... Args>
static constexpr T expand(T hd1, T hd2, Args... tl){
    return hd1 + hd2 * expand(tl...);
}

const int X_BLOCK = 16;
const int Y_BLOCK = 16;
const int K_BLOCK = 2;

template<unsigned int N>
struct NdTensor{
    double* _container;
    std::array<size_t, N> _dims;
    size_t _totalSize = 1;

    NdTensor(const std::array<size_t, N>& dims) : _dims(dims){
        _totalSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        auto ret = cudaMallocManaged(&_container, _totalSize * sizeof(double));
        if(ret == cudaErrorMemoryAllocation){
            std::bad_alloc exception;
            throw exception;
        }
    }
    double checkSum(){
        double sum = 0.0;
	for(size_t i=0; i < _totalSize; ++i){
	    sum += _container[i];
        }
        return sum;
    }
    ~NdTensor(){
        cudaFree(_container);
    }
};

__global__
void conv_cuda(double* out,
        const double* filter, int K, int C, int FW, int FH,
        const double* paddedImage, int W, int H){

    int k = threadIdx.x + (blockDim.x * blockIdx.x);
    int x = threadIdx.y + (blockDim.y * blockIdx.y);
    int y = threadIdx.z + (blockDim.z * blockIdx.z);

    auto outIndex = expand(y, H, x, W, k);

    double sum = 0.0;
    
    #pragma unroll 4
    for(int c = 0; c < C; ++c){
        for(int j = 0; j < FH; ++j){
            for(int i=0; i < FW; ++i){
                auto filterIndex = expand(FH - 1 - j, FH, FW - 1 - i, FW, c, C, k);
                auto imageIndex = expand(y + j, W + 2, x + i, H + 2, c);
                sum += (filter[filterIndex] * paddedImage[imageIndex]);
            }
        }            
    }
    out[outIndex] = sum;
}

__global__
void conv_cuda_tiled(double* out,
        const double* filter, int K, int C, int FW, int FH,
        const double* paddedImage, int W, int H){
        
    int k = threadIdx.x + (blockDim.x * blockIdx.x);
    int x = threadIdx.y + (blockDim.y * blockIdx.y);
    int y = threadIdx.z + (blockDim.z * blockIdx.z);

    int tidx = threadIdx.y;
    int tidy = threadIdx.z;

    int X_BOUND = X_BLOCK + FW - 1;
    int Y_BOUND = Y_BLOCK + FH - 1;

    extern __shared__ double tile[];

    for(int c = 0; c < C; ++c){
        tile[ expand(tidy, Y_BOUND, tidx, X_BOUND, c) ] = paddedImage[ expand(y, W + 2, x, H + 2, c) ]; 
        // corner loads
        if(tidx == X_BLOCK - 1 && tidy == Y_BLOCK - 1){
            for(int xx = 0; xx < FW; ++xx){
                for(int yy = 0; yy < FH; ++yy){
                    if(xx == 0 and yy == 0)
                        continue;
                    tile[ expand(tidy + yy, Y_BOUND, tidx + xx, X_BOUND, c) ] = paddedImage[ expand(y + yy, W + 2, x +xx, H + 2, c) ]; 
                }   
            }
        }// edge loads
        else if(tidx == X_BLOCK - 1){
            for(int xx = 1; xx < FW; ++xx){
                tile[ expand(tidy, Y_BOUND, tidx + xx, X_BOUND, c) ] = paddedImage[ expand(y, W + 2, x + xx, H + 2, c) ];
            }
        }else if(tidy == Y_BLOCK - 1){
            for(int yy = 1; yy < FH; ++yy){
                tile[ expand(tidy + yy, Y_BOUND, tidx, X_BOUND, c) ] = paddedImage[ expand(y + yy, W + 2, x, H + 2, c) ];
            }
        }
    }
    __syncthreads();

    auto outIndex = expand(y, H, x, W, k);
    double sum = 0.0;
    #pragma unroll 4
    for(int c = 0; c < C; ++c){
        for(int j = 0; j < FH; ++j){
            for(int i=0; i < FW; ++i){
                auto filterIndex = expand(FH - 1 - j, FH, FW - 1 - i, FW, c, C, k);
                auto imageIndex = expand(tidy + j, Y_BOUND, tidx + i, X_BOUND, c);
                sum += (filter[filterIndex] * tile[imageIndex]);
            }
        }
    }
    out[outIndex] = sum;
}
