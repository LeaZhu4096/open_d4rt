#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void fillFloatInput(MNN::Interpreter* net, MNN::Session* session, const std::string& name, float scale = 0.01f) {
    auto* input = net->getSessionInput(session, name.c_str());
    if (input == nullptr) {
        throw std::runtime_error("Missing float input: " + name);
    }
    std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(input, false));
    auto* data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = 0.5f + 0.5f * std::sin(static_cast<float>(i) * scale);
    }
    input->copyFromHostTensor(host.get());
}

void fillAspectRatio(MNN::Interpreter* net, MNN::Session* session) {
    auto* input = net->getSessionInput(session, "aspect_ratio");
    if (input == nullptr) {
        throw std::runtime_error("Missing input: aspect_ratio");
    }
    std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(input, false));
    auto* data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = 1.0f;
    }
    input->copyFromHostTensor(host.get());
}

void fillIntInput(MNN::Interpreter* net, MNN::Session* session, const std::string& name, int modulo) {
    auto* input = net->getSessionInput(session, name.c_str());
    if (input == nullptr) {
        throw std::runtime_error("Missing int input: " + name);
    }
    std::shared_ptr<MNN::Tensor> host(MNN::Tensor::createHostTensorFromDevice(input, false));
    auto* data = host->host<int32_t>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = modulo > 0 ? (i % modulo) : 0;
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
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << host->length(i);
    }
    std::cout << "] mean=" << sum / std::max(1, host->elementSize()) << "\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " opend4rt.mnn [forward_type] [num_frames] [height] [width] [num_queries]\n";
        return 2;
    }

    const std::string modelPath = argv[1];
    auto forwardType = MNN_FORWARD_VULKAN;
    if (argc >= 3) {
        forwardType = static_cast<MNNForwardType>(std::atoi(argv[2]));
    }
    const int numFrames = argc >= 4 ? std::atoi(argv[3]) : 2;
    const int height = argc >= 5 ? std::atoi(argv[4]) : 32;
    const int width = argc >= 6 ? std::atoi(argv[5]) : 32;
    const int numQueries = argc >= 7 ? std::atoi(argv[6]) : 8;

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

    const std::map<std::string, std::vector<int>> shapes = {
        {"video", {1, numFrames, 3, height, width}},
        {"u", {1, numQueries}},
        {"v", {1, numQueries}},
        {"t_src", {1, numQueries}},
        {"t_tgt", {1, numQueries}},
        {"t_cam", {1, numQueries}},
        {"aspect_ratio", {1, 1}},
    };

    for (const auto& item : shapes) {
        auto* input = net->getSessionInput(session, item.first.c_str());
        if (input == nullptr) {
            std::cerr << "Missing input: " << item.first << "\n";
            return 1;
        }
        net->resizeTensor(input, item.second);
    }
    net->resizeSession(session);

    try {
        fillFloatInput(net.get(), session, "video", 0.003f);
        fillFloatInput(net.get(), session, "u", 0.011f);
        fillFloatInput(net.get(), session, "v", 0.017f);
        fillIntInput(net.get(), session, "t_src", numFrames);
        fillIntInput(net.get(), session, "t_tgt", numFrames);
        fillIntInput(net.get(), session, "t_cam", numFrames);
        fillAspectRatio(net.get(), session);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }

    auto code = net->runSession(session);
    if (code != MNN::NO_ERROR) {
        std::cerr << "runSession failed: " << code << "\n";
        return 1;
    }

    bool ok = true;
    for (const std::string& name : {"xyz_3d", "uv_2d", "visibility", "displacement", "normal", "confidence"}) {
        ok = printOutput(net.get(), session, name) && ok;
    }
    std::cout << "MNN OpenD4RT smoke completed with forward_type=" << static_cast<int>(forwardType) << "\n";
    return ok ? 0 : 1;
}