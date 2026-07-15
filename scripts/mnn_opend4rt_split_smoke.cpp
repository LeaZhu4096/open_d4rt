#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

std::shared_ptr<MNN::Tensor> makeHostTensor(MNN::Tensor* deviceTensor) {
    return std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(deviceTensor, false));
}

void fillVideo(MNN::Tensor* host) {
    auto* data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = 0.5f + 0.5f * std::sin(static_cast<float>(i) * 0.003f);
    }
}

void fillFloat(MNN::Tensor* host, float scale, float bias = 0.5f) {
    auto* data = host->host<float>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = bias + 0.5f * std::sin(static_cast<float>(i) * scale);
    }
}

void fillInt(MNN::Tensor* host, int modulo) {
    auto* data = host->host<int32_t>();
    for (int i = 0; i < host->elementSize(); ++i) {
        data[i] = modulo > 0 ? (i % modulo) : 0;
    }
}

bool printOutput(MNN::Interpreter* net, MNN::Session* session, const std::string& name) {
    auto* output = net->getSessionOutput(session, name.c_str());
    if (output == nullptr) {
        std::cerr << "Missing output: " << name << "\n";
        return false;
    }
    auto host = makeHostTensor(output);
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

MNN::Session* createSession(MNN::Interpreter* net, MNNForwardType forwardType) {
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    MNN::ScheduleConfig config;
    config.type = forwardType;
    config.numThread = 4;
    config.backendConfig = &backendConfig;
    return net->createSession(config);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " encoder.mnn decoder.mnn [forward_type] [num_frames] [height] [width] [num_queries] [memory_tokens] [hidden_dim]\n";
        return 2;
    }
    const std::string encoderPath = argv[1];
    const std::string decoderPath = argv[2];
    auto forwardType = MNN_FORWARD_VULKAN;
    if (argc >= 4) forwardType = static_cast<MNNForwardType>(std::atoi(argv[3]));
    const int numFrames = argc >= 5 ? std::atoi(argv[4]) : 2;
    const int height = argc >= 6 ? std::atoi(argv[5]) : 32;
    const int width = argc >= 7 ? std::atoi(argv[6]) : 32;
    const int numQueries = argc >= 8 ? std::atoi(argv[7]) : 8;
    const int memoryTokens = argc >= 9 ? std::atoi(argv[8]) : 5;
    const int hiddenDim = argc >= 10 ? std::atoi(argv[9]) : 1280;

    std::shared_ptr<MNN::Interpreter> encoder(MNN::Interpreter::createFromFile(encoderPath.c_str()), MNN::Interpreter::destroy);
    std::shared_ptr<MNN::Interpreter> decoder(MNN::Interpreter::createFromFile(decoderPath.c_str()), MNN::Interpreter::destroy);
    if (!encoder || !decoder) {
        std::cerr << "Failed to open encoder or decoder model\n";
        return 1;
    }

    auto* encSession = createSession(encoder.get(), forwardType);
    auto* decSession = createSession(decoder.get(), forwardType);
    if (encSession == nullptr || decSession == nullptr) {
        std::cerr << "Failed to create split sessions for forward_type=" << static_cast<int>(forwardType) << "\n";
        return 1;
    }

    auto* encVideo = encoder->getSessionInput(encSession, "video");
    auto* encAspect = encoder->getSessionInput(encSession, "aspect_ratio");
    auto* decMemory = decoder->getSessionInput(decSession, "memory");
    auto* decVideo = decoder->getSessionInput(decSession, "video");
    if (encVideo == nullptr || encAspect == nullptr || decMemory == nullptr || decVideo == nullptr) {
        std::cerr << "Missing required split inputs\n";
        return 1;
    }

    encoder->resizeTensor(encVideo, {1, numFrames, 3, height, width});
    encoder->resizeTensor(encAspect, {1, 1});
    encoder->resizeSession(encSession);

    decoder->resizeTensor(decMemory, {1, memoryTokens, hiddenDim});
    decoder->resizeTensor(decVideo, {1, numFrames, 3, height, width});
    for (const std::string& name : {"u", "v", "t_src", "t_tgt", "t_cam"}) {
        auto* input = decoder->getSessionInput(decSession, name.c_str());
        if (input == nullptr) {
            std::cerr << "Missing decoder input: " << name << "\n";
            return 1;
        }
        decoder->resizeTensor(input, {1, numQueries});
    }
    decoder->resizeSession(decSession);

    auto encVideoHost = makeHostTensor(encVideo);
    fillVideo(encVideoHost.get());
    encVideo->copyFromHostTensor(encVideoHost.get());
    auto encAspectHost = makeHostTensor(encAspect);
    fillFloat(encAspectHost.get(), 0.0f, 1.0f);
    encAspect->copyFromHostTensor(encAspectHost.get());

    auto decVideoHost = makeHostTensor(decVideo);
    fillVideo(decVideoHost.get());
    decVideo->copyFromHostTensor(decVideoHost.get());
    for (const std::string& name : {"u", "v"}) {
        auto* input = decoder->getSessionInput(decSession, name.c_str());
        auto host = makeHostTensor(input);
        fillFloat(host.get(), name == "u" ? 0.011f : 0.017f, 0.5f);
        input->copyFromHostTensor(host.get());
    }
    for (const std::string& name : {"t_src", "t_tgt", "t_cam"}) {
        auto* input = decoder->getSessionInput(decSession, name.c_str());
        auto host = makeHostTensor(input);
        fillInt(host.get(), numFrames);
        input->copyFromHostTensor(host.get());
    }

    auto code = encoder->runSession(encSession);
    if (code != MNN::NO_ERROR) {
        std::cerr << "encoder runSession failed: " << code << "\n";
        return 1;
    }
    auto* encMemory = encoder->getSessionOutput(encSession, "memory");
    if (encMemory == nullptr) {
        std::cerr << "Missing encoder output: memory\n";
        return 1;
    }
    auto memoryHost = makeHostTensor(encMemory);
    encMemory->copyToHostTensor(memoryHost.get());
    decMemory->copyFromHostTensor(memoryHost.get());

    code = decoder->runSession(decSession);
    if (code != MNN::NO_ERROR) {
        std::cerr << "decoder runSession failed: " << code << "\n";
        return 1;
    }

    bool ok = true;
    for (const std::string& name : {"xyz_3d", "uv_2d", "visibility", "displacement", "normal", "confidence"}) {
        ok = printOutput(decoder.get(), decSession, name) && ok;
    }
    std::cout << "MNN OpenD4RT split smoke completed with forward_type=" << static_cast<int>(forwardType) << "\n";
    return ok ? 0 : 1;
}