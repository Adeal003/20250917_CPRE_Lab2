#include "Softmax.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>
#include <cmath>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void SoftMaxLayer::computeNaive(const LayerData &dataIn) const
    {

        // Get the number of elements to process
        size_t numElements = getInputParams().flat_count();
        
        LayerData& output = getOutputData();

        // Find the maximum value for numerical stability
        fp32 maxVal = -INFINITY;
        for (size_t i = 0; i < numElements; i++)
        {
            fp32 val = dataIn.get<fp32>(i);
            if (val > maxVal)
            {
                maxVal = val;
            }
        }

        // Compute exponentials and sum
        fp32 sumExp = 0.0f;
        for (size_t i = 0; i < numElements; i++)
        {
            fp32 expVal = std::exp(dataIn.get<fp32>(i) - maxVal);
            output.get<fp32>(i) = expVal;
            sumExp += expVal;
        }

        // Normalize by the sum
        for (size_t i = 0; i < numElements; i++)
        {
            output.get<fp32>(i) = output.get<fp32>(i) / sumExp;
        }
    }

    void SoftMaxLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void SoftMaxLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled processing
        computeNaive(dataIn);
    }

    void SoftMaxLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized softmax
        computeNaive(dataIn);
    }

}