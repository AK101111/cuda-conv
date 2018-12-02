#include "conv.h"
#include <cudnn.h>
#include <omp.h>

inline void checkLib(cudnnStatus_t stat){
    if(stat != CUDNN_STATUS_SUCCESS){
        std::cerr << cudnnGetErrorString(stat) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template<typename fdType, cudnnStatus_t (*C)(fdType*), cudnnStatus_t (*D)(fdType)>
struct SmartCudaFd{
    fdType _fd;
    SmartCudaFd(){
        C(&_fd);
    }
    ~SmartCudaFd(){
        D(_fd);
    }
    fdType* operator&() const {
        return &_fd;               
    }
};

void init_tensor_padded(NdTensor<3>& tensor){
    auto C = tensor._dims[0]; auto H = tensor._dims[1];
    auto W = tensor._dims[2];
    
    for(int c = 0; c < C; ++c){
        #pragma omp parallel for
        for(int w = 0; w < W; ++w){
            tensor._container[expand3(w, 0, c, W, H)] = 0.0;
            tensor._container[expand3(w, H-1, c, W, H)] = 0.0;
        }
        #pragma omp parallel for
        for(int h = 0; h < H; ++h){
            tensor._container[expand3(0, h, c, W, H)] = 0.0;
            tensor._container[expand3(W-1, h, c, W, H)] = 0.0;
        }
    }

    for(int c = 0; c < C; ++c){
        #pragma omp parallel for
        for(int h = 1; h < H-1; ++h){
            for(int w = 1; w < W-1; ++w){
                tensor._container[expand3(w, h, c, W, H)] = c*(h+w-2);
            }
        }
    }
}

void init_tensor(NdTensor<3>& tensor){
    auto C = tensor._dims[0]; auto H = tensor._dims[1];
    auto W = tensor._dims[2];
    
    for(int c = 0; c < C; ++c){
        #pragma omp parallel for
        for(int h = 0; h < H; ++h){
            for(int w = 0; w < W; ++w){
                tensor._container[expand3(w, h, c, W, H)] = c*(h+w);
            }
        }
    }
}

void init_conv_filter_transposed(NdTensor<4>& filter){
    auto K = filter._dims[0]; auto C = filter._dims[1];
    auto FW = filter._dims[2]; auto FH = filter._dims[3];

    #pragma omp parallel for
    for(int k = 0; k < K; ++k){
        for(int c = 0; c < C; ++c){
            for(int i = 0; i < FW; ++i){
                for(int j = 0; j < FH; ++j){
                    filter._container[expand4(j, i, c, k, FH, FW, C)] = 
                        (c + k)*(i + j);
                }
            }
        }
    }
}

void cudnn_convolution_simple(NdTensor<3>& imageTensor, 
        NdTensor<4>& filter) {
    auto C = filter._dims[1]; 
    auto H = imageTensor._dims[1];
    auto W = imageTensor._dims[2];
    auto K = filter._dims[0];
    auto FH = filter._dims[2];
    auto FW = filter._dims[3];
    
    SmartCudaFd<cudnnHandle_t, cudnnCreate, cudnnDestroy> cudnn;
    
    SmartCudaFd<cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor> inputd;
    // providing transposed image
    checkLib(cudnnSetTensor4dDescriptor(inputd._fd, CUDNN_TENSOR_NCHW, 
                CUDNN_DATA_DOUBLE, 1, C, W, H));
    
    SmartCudaFd<cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor> outputd;
    checkLib(cudnnSetTensor4dDescriptor(outputd._fd, CUDNN_TENSOR_NCHW, 
                CUDNN_DATA_DOUBLE, 1, K, H, W));
    
    SmartCudaFd<cudnnFilterDescriptor_t, cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor> filterd;
    cudnnSetFilter4dDescriptor(filterd._fd, CUDNN_DATA_DOUBLE, 
            CUDNN_TENSOR_NCHW, K, C, FH, FW);

    SmartCudaFd<cudnnConvolutionDescriptor_t, cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor> convd;
    checkLib(cudnnSetConvolution2dDescriptor(convd._fd, 
                1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkLib(cudnnGetConvolutionForwardAlgorithm(cudnn._fd, inputd._fd, 
                filterd._fd, convd._fd, outputd._fd, 
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm));
    
    size_t workspace_size = 0; //1000000;
    checkLib(cudnnGetConvolutionForwardWorkspaceSize(cudnn._fd, inputd._fd, filterd._fd, 
                convd._fd, outputd._fd, convolution_algorithm, &workspace_size));
    
    std::array<size_t, 3> outputA{K, W, H};
    std::array<size_t, 1> workspaceA{workspace_size / sizeof(double)};
 
    NdTensor<3> output(outputA);
    NdTensor<1> workspace(workspaceA);
                                            
    double alpha = 1.0, beta = 0.0;

    auto start = std::chrono::system_clock::now();
    checkLib(cudnnConvolutionForward(cudnn._fd, &alpha, 
                inputd._fd, imageTensor._container, 
                filterd._fd, filter._container, 
                convd._fd, convolution_algorithm, 
                workspace._container, workspace_size, 
                &beta, outputd._fd, output._container));
    auto end = std::chrono::system_clock::now();
    cudaDeviceSynchronize(); 
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << std::fixed << output.checkSum() << "," << elapsed_seconds.count()* 1000 << std::endl;
}

int main(){
    cudaDeviceReset();
    const std::array<size_t, 3> tSize = {3, 4098, 4098};
    const std::array<size_t, 3> ttSize = {3, 4096, 4096};
    const std::array<size_t, 4> fSize = {10, 3, 3, 3};
    const std::array<size_t, 3> oSize = {10, 4096, 4096};

    {
        NdTensor<3> inputImage(ttSize);
        init_tensor(inputImage);

        NdTensor<3> paddedInput(tSize);
        init_tensor_padded(paddedInput);

        NdTensor<4> filter(fSize);
        init_conv_filter_transposed(filter);
    
        NdTensor<3> output(oSize);

        dim3 threadsPerBlock(K_BLOCK, X_BLOCK, Y_BLOCK);
        dim3 numBlocks(oSize[0] / K_BLOCK, oSize[1]/ X_BLOCK, oSize[2]/ Y_BLOCK); //10/2, 4096/16, 4096/16
   
    /******************* C1 *******************/
        auto start = std::chrono::system_clock::now();
        conv_cuda<<<numBlocks, threadsPerBlock>>>(
            output._container,
            filter._container, fSize[0], fSize[1], fSize[2], fSize[3],
            paddedInput._container, oSize[1], oSize[2]
            );
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
    
        double checkSum = output.checkSum();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout.precision(4);
        std::cout << std::fixed << checkSum << "," << elapsed_seconds.count()* 1000 << std::endl;
    }
    /******************* C2 *******************/
    {
        NdTensor<3> paddedInput(tSize);
        init_tensor_padded(paddedInput);
        
        NdTensor<4> filter(fSize);
        init_conv_filter_transposed(filter);
        
        NdTensor<3> output(oSize);
        
        dim3 threadsPerBlock(K_BLOCK, X_BLOCK, Y_BLOCK);
        dim3 numBlocks(oSize[0] / K_BLOCK, oSize[1]/ X_BLOCK, oSize[2]/ Y_BLOCK); //10/2, 4096/16, 4096/16
        auto start = std::chrono::system_clock::now();
        conv_cuda_tiled<<<numBlocks, threadsPerBlock, ((X_BLOCK + fSize[1] - 1) * (Y_BLOCK + fSize[2] - 1) * (fSize[3])) * sizeof(double)>>>(
                output._container,
                filter._container, fSize[0], fSize[1], fSize[2], fSize[3],
                paddedInput._container, oSize[1], oSize[2]
                );
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
    
        double checkSum = output.checkSum();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << std::fixed << checkSum << "," << elapsed_seconds.count() * 1000 << std::endl;
    }
    /******************* C3 *******************/
    {
        NdTensor<3> inputImage(ttSize);
        init_tensor(inputImage);

        NdTensor<4> filter(fSize);
        init_conv_filter_transposed(filter);
    
        cudnn_convolution_simple(inputImage, filter);
    }

    return 0;
}
