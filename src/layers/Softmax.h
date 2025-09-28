#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
    class SoftMaxLayer : public Layer {
    public:
        SoftMaxLayer(const LayerParams inParams, const LayerParams outParams)
            : Layer(inParams, outParams, LayerType::SOFTMAX) {}

        // Virtual functions
        virtual void computeNaive(const LayerData& dataIn) const override;
        virtual void computeThreaded(const LayerData& dataIn) const override;
        virtual void computeTiled(const LayerData& dataIn) const override;
        virtual void computeSIMD(const LayerData& dataIn) const override;

        // Allocate all resources needed for the layer & Load all of the required data for the layer
        virtual void allocLayer() override {
            Layer::allocLayer();
        
        }

        // Fre all resources allocated for the layer
        virtual void freeLayer() override {
            Layer::freeLayer();

        }
    };

}  // namespace ML