#include <iostream>
#include <sstream>
#include <vector>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Build our ML toy model
Model buildToyModel(const Path modelPath) {
    Model model;
    logInfo("--- Building Toy Model ---");

    // --- Conv 1: L1 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32

    // You can pick how you want to implement your layers, both are allowed:

    // LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
    // LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
    // LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
    // LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");
    // auto conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
        LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
        LayerParams{sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "conv1_biases.bin"}            // Bias
    );

    // --- Conv 2: L2 ---
    // Input shape: 60x60x32
    // Output shape: 56x56x32

    // --- MPL 1: L3 ---
    // Input shape: 56x56x32
    // Output shape: 28x28x32

    // --- Conv 3: L4 ---
    // Input shape: 28x28x32
    // Output shape: 26x26x64

    // --- Conv 4: L5 ---
    // Input shape: 26x26x64
    // Output shape: 24x24x64

    // --- MPL 2: L6 ---
    // Input shape: 24x24x64
    // Output shape: 12x12x64

    // --- Conv 5: L7 ---
    // Input shape: 12x12x64
    // Output shape: 10x10x64

    // --- Conv 6: L8 ---
    // Input shape: 10x10x64
    // Output shape: 8x8x128

    // --- MPL 3: L9 ---
    // Input shape: 8x8x128
    // Output shape: 4x4x128

    // --- Flatten 1: L10 ---
    // Input shape: 4x4x128
    // Output shape: 2048

    // --- Dense 1: L11 ---
    // Input shape: 2048
    // Output shape: 256

    // --- Dense 2: L12 ---
    // Input shape: 256
    // Output shape: 200

    // --- Softmax 1: L13 ---
    // Input shape: 200
    // Output shape: 200

    return model;
}

void runBasicTest(const Model& model, const Path& basePath) {
    logInfo("--- Running Basic Test ---");

    // Load an image
    LayerData img = {{sizeof(fp32), {64, 64, 3}, "./data/image_0.bin"}};
    img.loadData();

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
              << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 0.1 and compare again" << std::endl;
    
    LayerData imgCopy = img;
    imgCopy.get<fp32>(0) += 0.1;

    // Compare images
    img.compareWithinPrint<fp32>(imgCopy);

    // Test again with a modified copy
    log("Change a value by 0.1 and compare again...");
    imgCopy.get<fp32>(0) += 0.1;

    // Compare Images
    img.compareWithinPrint<fp32>(imgCopy);
}

void runLayerTest(const std::size_t layerNum, const Model& model, const Path& basePath) {
    // Load an image
    logInfo(std::string("--- Running Layer Test ") + std::to_string(layerNum) + "---");

    // Construct a LayerData object from a LayerParams one
    // LayerData img(model[layerNum].getInputParams(), test_image_files[layerNum].first);
    dimVec inDims = {64, 64, 3};
    LayerData img({sizeof(fp32), inDims, basePath / "image_0.bin"});
    img.loadData();

    Timer timer("Layer Inference");

    // Run inference on the model
    timer.start();
    const LayerData& output = model.inferenceLayer(img, layerNum, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    LayerData expected(output.getParams(), basePath / "image_0_data" / "layer_0_output.bin");
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

void runInferenceTest(const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Inference Test ---");

    // Construct a LayerData object from a LayerParams one
    LayerData img(model[0].getInputParams(), basePath / "image_0.bin");
    img.loadData();

    Timer timer("Full Inference");

    // Run inference on the model
    timer.start();
    const LayerData& output = model.inference(img, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    LayerData expected(model.getOutputLayer().getOutputParams(), basePath / "image_0_data" / "layer_0_output.bin");
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

void runTests() {
    // Base input data path (determined from current directory of where you are running the command)
    Path basePath("data");  // May need to be altered for zedboards loading from SD Cards

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers();

    // Run some framework tests as an example of loading data
    runBasicTest(model, basePath);

    // Run a layer inference test
    runLayerTest(0, model, basePath);

    // Run an end-to-end inference test
    runInferenceTest(model, basePath);

    // Clean up
    model.freeLayers();
    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif