#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

void fillFloatInput(MNN::Interpreter* net, MNN::Session* session, const std::string& name, float scale = 0.01f, float bias = 0.5f) {
    auto* input = net->getSessionInput(session, name.c_str());
    if (input == nullptr) {
        throw std::runtime_error("Missing input: " + name);
    }
    std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(input, false));
    auto* data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = bias + 0.5f * std::sin(static_cast<float>(i) * scale);
    }
    input->copyFromHostTensor(host.get());
}

bool printOutput(MNN::Interpreter* net, MNN::Session* session, const std::string& name) {
    auto* output = net->getSessionOutput(session, name.c_str());
    if (output == nullptr) {
        std::cerr << "Missing output: " << name << "\n";
        return false;
    }
    std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(output, false));
    output->copyToHostTensor(host.get());
    const auto* data = host->host<float>();
    double sum = 0.0;
    for (int i = 0; i < host->elementSize(); ++i) {
        sum += data[i];
    }
    std::cout << name << " shape=[";
    for (int i = 0; i < host->dimensions(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << host->length(i);
    }
    std::cout << "] mean=" << sum / std::max(1, host->elementSize()) << "\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " encoder.mnn [forward_type] [num_frames] [height] [width]\n";
        return 2;
    }
    const std::string modelPath = argv[1];
    auto forwardType = MNN_FORWARD_VULKAN;
    if (argc >= 3) forwardType = static_cast<MNNForwardType>(std::atoi(argv[2]));
    const int numFrames = argc >= 4 ? std::atoi(argv[3]) : 2;
    const int height = argc >= 5 ? std::atoi(argv[4]) : 32;
    const int width = argc >= 6 ? std::atoi(argv[5]) : 32;

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()), MNN::Interpreter::destroy);
    if (!net) {
        std::cerr << "Failed to open model: " << modelPath << "\n";
        return 1;
    }

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    MNN::ScheduleConfig config;
    config.type = forwardType;
    config.numThread = 4;
    config.backendConfig = &backendConfig;

    auto* session = net->createSession(config);
    if (session == nullptr) {
        std::cerr << "Failed to create session for forward_type=" << static_cast<int>(forwardType) << "\n";
        return 1;
    }

    auto* video = net->getSessionInput(session, "video");
    auto* aspect = net->getSessionInput(session, "aspect_ratio");
    if (video == nullptr || aspect == nullptr) {
        std::cerr << "Missing encoder inputs\n";
        return 1;
    }
    net->resizeTensor(video, {1, numFrames, 3, height, width});
    net->resizeTensor(aspect, {1, 1});
    net->resizeSession(session);

    try {
        fillFloatInput(net.get(), session, "video", 0.003f, 0.5f);
        fillFloatInput(net.get(), session, "aspect_ratio", 0.0f, 1.0f);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }

    auto code = net->runSession(session);
    if (code != MNN::NO_ERROR) {
        std::cerr << "runSession failed: " << code << "\n";
        return 1;
    }
    const bool ok = printOutput(net.get(), session, "memory");
    std::cout << "MNN OpenD4RT encoder smoke completed with forward_type=" << static_cast<int>(forwardType) << "\n";
    return ok ? 0 : 1;
}