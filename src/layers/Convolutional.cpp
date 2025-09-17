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
                            
                            // Bounds check
                            if (h_in < inputHeight && w_in < inputWidth) {
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
    // Get dimensions
    const auto& inputDims = getInputParams().dims;
    const auto& outputDims = getOutputParams().dims;
    const auto& weightDims = getWeightParams().dims;
    
    size_t inputHeight = inputDims[0];
    size_t inputWidth = inputDims[1];
    size_t inputChannels = inputDims[2];
    size_t outputHeight = outputDims[0];
    size_t outputWidth = outputDims[1];
    size_t outputChannels = outputDims[2];
    size_t kernelHeight = weightDims[0];
    size_t kernelWidth = weightDims[1];
    
    const LayerData& weights = getWeightData();
    const LayerData& bias = getBiasData();
    LayerData& output = getOutputData();
    
    // Determine number of threads
    size_t numThreads = std::min((size_t)4, outputHeight);  // Use up to 4 threads
    std::vector<std::thread> threads;
    
    // Lambda function for processing a range of output rows
    auto processRows = [&](size_t startRow, size_t endRow) {
        for (size_t h_out = startRow; h_out < endRow; h_out++) {
            for (size_t w_out = 0; w_out < outputWidth; w_out++) {
                for (size_t c_out = 0; c_out < outputChannels; c_out++) {
                    
                    fp32 sum = bias.get<fp32>(c_out);
                    
                    for (size_t k_h = 0; k_h < kernelHeight; k_h++) {
                        for (size_t k_w = 0; k_w < kernelWidth; k_w++) {
                            for (size_t c_in = 0; c_in < inputChannels; c_in++) {
                                
                                size_t h_in = h_out + k_h;
                                size_t w_in = w_out + k_w;
                                
                                if (h_in < inputHeight && w_in < inputWidth) {
                                    size_t inputIdx = h_in * (inputWidth * inputChannels) + 
                                                    w_in * inputChannels + c_in;
                                    
                                    size_t weightIdx = k_h * (kernelWidth * inputChannels * outputChannels) +
                                                     k_w * (inputChannels * outputChannels) +
                                                     c_in * outputChannels + c_out;
                                    
                                    sum += dataIn.get<fp32>(inputIdx) * weights.get<fp32>(weightIdx);
                                }
                            }
                        }
                    }
                    
                    size_t outputIdx = h_out * (outputWidth * outputChannels) + 
                                     w_out * outputChannels + c_out;
                    output.get<fp32>(outputIdx) = sum;
                }
            }
        }
    };
    
    // Launch threads
    for (size_t t = 0; t < numThreads; t++) {
        size_t startRow = t * outputHeight / numThreads;
        size_t endRow = (t + 1) * outputHeight / numThreads;
        threads.emplace_back(processRows, startRow, endRow);
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // Get dimensions
    const auto& inputDims = getInputParams().dims;
    const auto& outputDims = getOutputParams().dims;
    const auto& weightDims = getWeightParams().dims;
    
    size_t inputHeight = inputDims[0];
    size_t inputWidth = inputDims[1];
    size_t inputChannels = inputDims[2];
    size_t outputHeight = outputDims[0];
    size_t outputWidth = outputDims[1];
    size_t outputChannels = outputDims[2];
    size_t kernelHeight = weightDims[0];
    size_t kernelWidth = weightDims[1];
    
    const LayerData& weights = getWeightData();
    const LayerData& bias = getBiasData();
    LayerData& output = getOutputData();
    
    const size_t TILE_H = 8;  // Tile size for height
    const size_t TILE_W = 8;  // Tile size for width
    const size_t TILE_C = 16; // Tile size for channels
    
    // Tile over output dimensions for cache efficiency
    for (size_t h_tile = 0; h_tile < outputHeight; h_tile += TILE_H) {
        for (size_t w_tile = 0; w_tile < outputWidth; w_tile += TILE_W) {
            for (size_t c_tile = 0; c_tile < outputChannels; c_tile += TILE_C) {
                
                // Calculate tile boundaries
                size_t h_end = std::min(h_tile + TILE_H, outputHeight);
                size_t w_end = std::min(w_tile + TILE_W, outputWidth);
                size_t c_end = std::min(c_tile + TILE_C, outputChannels);
                
                // Process within tile
                for (size_t h_out = h_tile; h_out < h_end; h_out++) {
                    for (size_t w_out = w_tile; w_out < w_end; w_out++) {
                        for (size_t c_out = c_tile; c_out < c_end; c_out++) {
                            
                            fp32 sum = bias.get<fp32>(c_out);
                            
                            for (size_t k_h = 0; k_h < kernelHeight; k_h++) {
                                for (size_t k_w = 0; k_w < kernelWidth; k_w++) {
                                    for (size_t c_in = 0; c_in < inputChannels; c_in++) {
                                        
                                        size_t h_in = h_out + k_h;
                                        size_t w_in = w_out + k_w;
                                        
                                        if (h_in < inputHeight && w_in < inputWidth) {
                                            size_t inputIdx = h_in * (inputWidth * inputChannels) + 
                                                            w_in * inputChannels + c_in;
                                            
                                            size_t weightIdx = k_h * (kernelWidth * inputChannels * outputChannels) +
                                                             k_w * (inputChannels * outputChannels) +
                                                             c_in * outputChannels + c_out;
                                            
                                            sum += dataIn.get<fp32>(inputIdx) * weights.get<fp32>(weightIdx);
                                        }
                                    }
                                }
                            }
                            
                            size_t outputIdx = h_out * (outputWidth * outputChannels) + 
                                             w_out * outputChannels + c_out;
                            output.get<fp32>(outputIdx) = sum;
                        }
                    }
                }
            }
        }
    }
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // Get dimensions
    const auto& inputDims = getInputParams().dims;
    const auto& outputDims = getOutputParams().dims;
    const auto& weightDims = getWeightParams().dims;
    
    size_t inputHeight = inputDims[0];
    size_t inputWidth = inputDims[1];
    size_t inputChannels = inputDims[2];
    size_t outputHeight = outputDims[0];
    size_t outputWidth = outputDims[1];
    size_t outputChannels = outputDims[2];
    size_t kernelHeight = weightDims[0];
    size_t kernelWidth = weightDims[1];
    
    const LayerData& weights = getWeightData();
    const LayerData& bias = getBiasData();
    LayerData& output = getOutputData();
    
    // SIMD-optimized convolution (using compiler auto-vectorization)
    for (size_t h_out = 0; h_out < outputHeight; h_out++) {
        for (size_t w_out = 0; w_out < outputWidth; w_out++) {
            
            // Process multiple output channels together for vectorization
            for (size_t c_out = 0; c_out < outputChannels; c_out++) {
                
                fp32 sum = bias.get<fp32>(c_out);
                
                // Inner loops optimized for vectorization
                for (size_t k_h = 0; k_h < kernelHeight; k_h++) {
                    for (size_t k_w = 0; k_w < kernelWidth; k_w++) {
                        
                        size_t h_in = h_out + k_h;
                        size_t w_in = w_out + k_w;
                        
                        if (h_in < inputHeight && w_in < inputWidth) {
                            // Vectorizable inner loop over input channels
                            for (size_t c_in = 0; c_in < inputChannels; c_in++) {
                                
                                size_t inputIdx = h_in * (inputWidth * inputChannels) + 
                                                w_in * inputChannels + c_in;
                                
                                size_t weightIdx = k_h * (kernelWidth * inputChannels * outputChannels) +
                                                 k_w * (inputChannels * outputChannels) +
                                                 c_in * outputChannels + c_out;
                                
                                sum += dataIn.get<fp32>(inputIdx) * weights.get<fp32>(weightIdx);
                            }
                        }
                    }
                }
                
                size_t outputIdx = h_out * (outputWidth * outputChannels) + 
                                 w_out * outputChannels + c_out;
                output.get<fp32>(outputIdx) = sum;
            }
        }
    }
}

}  // namespace ML
