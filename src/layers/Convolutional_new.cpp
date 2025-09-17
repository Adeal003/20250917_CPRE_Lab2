#include "Convolutional.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---
//ASDFf
// Compute the convolution for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
    // Get dimensions from layer parameters
    const auto& inputDims = getInputParams().dims;   // [H, W, C_in]
    const auto& outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
    const auto& weightDims = getWeightParams().dims; // [K_H, K_W, C_in, C_out]
    
    size_t inputHeight = inputDims[0];
    size_t inputWidth = inputDims[1];
    size_t inputChannels = inputDims[2];
    
    size_t outputHeight = outputDims[0];
    size_t outputWidth = outputDims[1];
    size_t outputChannels = outputDims[2];
    
    size_t kernelHeight = weightDims[0];
    size_t kernelWidth = weightDims[1];
    
    // Get access to weight and bias data
    const LayerData& weights = getWeightData();
    const LayerData& bias = getBiasData();
    LayerData& output = getOutputData();
    
    // Perform convolution
    for (size_t h_out = 0; h_out < outputHeight; h_out++) {
        for (size_t w_out = 0; w_out < outputWidth; w_out++) {
            for (size_t c_out = 0; c_out < outputChannels; c_out++) {
                
                // Initialize with bias
                fp32 sum = bias.get<fp32>(c_out);
                
                // Convolve with kernel
                for (size_t k_h = 0; k_h < kernelHeight; k_h++) {
                    for (size_t k_w = 0; k_w < kernelWidth; k_w++) {
                        for (size_t c_in = 0; c_in < inputChannels; c_in++) {
                            
                            // Calculate input coordinates
                            size_t h_in = h_out + k_h;
                            size_t w_in = w_out + k_w;
                            
                            // Input index in HWC format: [h][w][c]
                            size_t inputIdx = h_in * (inputWidth * inputChannels) + 
                                            w_in * inputChannels + 
                                            c_in;
                            
                            // Weight index: [k_h][k_w][c_in][c_out]
                            size_t weightIdx = k_h * (kernelWidth * inputChannels * outputChannels) +
                                             k_w * (inputChannels * outputChannels) +
                                             c_in * outputChannels +
                                             c_out;
                            
                            sum += dataIn.get<fp32>(inputIdx) * weights.get<fp32>(weightIdx);
                        }
                    }
                }
                
                // Output index in HWC format: [h][w][c]
                size_t outputIdx = h_out * (outputWidth * outputChannels) + 
                                 w_out * outputChannels + 
                                 c_out;
                
                output.get<fp32>(outputIdx) = sum;
            }
        }
    }
}

// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // For simplicity, use naive implementation with thread hints
    computeNaive(dataIn);
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // For simplicity, use naive implementation 
    computeNaive(dataIn);
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // For simplicity, use naive implementation
    computeNaive(dataIn);
}

}  // namespace ML
