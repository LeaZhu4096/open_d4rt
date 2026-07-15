#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::shared_ptr<MNN::Tensor> makeHostTensor(MNN::Tensor* deviceTensor) {
    return std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(deviceTensor, false));
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

template <typename T>
std::vector<T> readBinary(const fs::path& path, size_t expectedCount) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    std::vector<T> data(expectedCount);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (static_cast<size_t>(in.gcount()) != data.size() * sizeof(T)) {
        throw std::runtime_error("Unexpected byte count in input file: " + path.string());
    }
    return data;
}

template <typename T>
void writeBinary(const fs::path& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
}

void copyFloatVectorToTensor(MNN::Tensor* tensor, const std::vector<float>& data) {
    auto host = makeHostTensor(tensor);
    if (host->elementSize() != static_cast<int>(data.size())) {
        throw std::runtime_error("Tensor element count mismatch for float input");
    }
    std::copy(data.begin(), data.end(), host->host<float>());
    tensor->copyFromHostTensor(host.get());
}

void copyIntBatchToTensor(MNN::Tensor* tensor, const std::vector<int32_t>& data, int offset, int batch, int actual) {
    auto host = makeHostTensor(tensor);
    if (host->elementSize() != batch) {
        throw std::runtime_error("Tensor element count mismatch for int query input");
    }
    auto* dst = host->host<int32_t>();
    for (int i = 0; i < batch; ++i) {
        const int src = offset + std::min(i, std::max(0, actual - 1));
        dst[i] = data[src];
    }
    tensor->copyFromHostTensor(host.get());
}

void copyFloatBatchToTensor(MNN::Tensor* tensor, const std::vector<float>& data, int offset, int batch, int actual) {
    auto host = makeHostTensor(tensor);
    if (host->elementSize() != batch) {
        throw std::runtime_error("Tensor element count mismatch for float query input");
    }
    auto* dst = host->host<float>();
    for (int i = 0; i < batch; ++i) {
        const int src = offset + std::min(i, std::max(0, actual - 1));
        dst[i] = data[src];
    }
    tensor->copyFromHostTensor(host.get());
}

void appendOutput(MNN::Interpreter* net, MNN::Session* session, const std::string& name, int batch, int actual, std::vector<float>& dst) {
    auto* output = net->getSessionOutput(session, name.c_str());
    if (output == nullptr) {
        throw std::runtime_error("Missing decoder output: " + name);
    }
    auto host = makeHostTensor(output);
    output->copyToHostTensor(host.get());
    const int elements = host->elementSize();
    if (elements % batch != 0) {
        throw std::runtime_error("Output element count is not divisible by query batch for: " + name);
    }
    const int perQuery = elements / batch;
    const float* src = host->host<float>();
    const size_t oldSize = dst.size();
    dst.resize(oldSize + static_cast<size_t>(actual) * perQuery);
    for (int i = 0; i < actual; ++i) {
        std::copy(src + i * perQuery, src + (i + 1) * perQuery, dst.begin() + oldSize + static_cast<size_t>(i) * perQuery);
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " encoder.mnn decoder.mnn input_dir output_dir total_queries [forward_type=7] [num_frames=32] [height=256] [width=256] [query_batch=8] [memory_tokens=4097] [hidden_dim=1280]\n";
        return 2;
    }

    try {
        const std::string encoderPath = argv[1];
        const std::string decoderPath = argv[2];
        const fs::path inputDir = argv[3];
        const fs::path outputDir = argv[4];
        const int totalQueries = std::atoi(argv[5]);
        auto forwardType = argc >= 7 ? static_cast<MNNForwardType>(std::atoi(argv[6])) : MNN_FORWARD_VULKAN;
        const int numFrames = argc >= 8 ? std::atoi(argv[7]) : 32;
        const int height = argc >= 9 ? std::atoi(argv[8]) : 256;
        const int width = argc >= 10 ? std::atoi(argv[9]) : 256;
        const int queryBatch = argc >= 11 ? std::atoi(argv[10]) : 8;
        const int memoryTokens = argc >= 12 ? std::atoi(argv[11]) : 4097;
        const int hiddenDim = argc >= 13 ? std::atoi(argv[12]) : 1280;

        if (totalQueries <= 0 || queryBatch <= 0) {
            throw std::runtime_error("total_queries and query_batch must be positive");
        }
        fs::create_directories(outputDir);

        auto video = readBinary<float>(inputDir / "video_f32.bin", static_cast<size_t>(numFrames) * 3 * height * width);
        auto aspect = readBinary<float>(inputDir / "aspect_ratio_f32.bin", 1);
        auto queryU = readBinary<float>(inputDir / "query_u_f32.bin", totalQueries);
        auto queryV = readBinary<float>(inputDir / "query_v_f32.bin", totalQueries);
        auto queryTSrc = readBinary<int32_t>(inputDir / "query_t_src_i32.bin", totalQueries);
        auto queryTTgt = readBinary<int32_t>(inputDir / "query_t_tgt_i32.bin", totalQueries);
        auto queryTCam = readBinary<int32_t>(inputDir / "query_t_cam_i32.bin", totalQueries);

        std::shared_ptr<MNN::Interpreter> encoder(MNN::Interpreter::createFromFile(encoderPath.c_str()), MNN::Interpreter::destroy);
        std::shared_ptr<MNN::Interpreter> decoder(MNN::Interpreter::createFromFile(decoderPath.c_str()), MNN::Interpreter::destroy);
        if (!encoder || !decoder) {
            throw std::runtime_error("Failed to open encoder or decoder model");
        }

        auto* encSession = createSession(encoder.get(), forwardType);
        auto* decSession = createSession(decoder.get(), forwardType);
        if (encSession == nullptr || decSession == nullptr) {
            throw std::runtime_error("Failed to create MNN sessions");
        }

        auto* encVideo = encoder->getSessionInput(encSession, "video");
        auto* encAspect = encoder->getSessionInput(encSession, "aspect_ratio");
        auto* decMemory = decoder->getSessionInput(decSession, "memory");
        auto* decVideo = decoder->getSessionInput(decSession, "video");
        if (encVideo == nullptr || encAspect == nullptr || decMemory == nullptr || decVideo == nullptr) {
            throw std::runtime_error("Missing required model inputs");
        }

        encoder->resizeTensor(encVideo, {1, numFrames, 3, height, width});
        encoder->resizeTensor(encAspect, {1, 1});
        encoder->resizeSession(encSession);

        decoder->resizeTensor(decMemory, {1, memoryTokens, hiddenDim});
        decoder->resizeTensor(decVideo, {1, numFrames, 3, height, width});
        for (const std::string& name : {"u", "v", "t_src", "t_tgt", "t_cam"}) {
            auto* input = decoder->getSessionInput(decSession, name.c_str());
            if (input == nullptr) throw std::runtime_error("Missing decoder input: " + name);
            decoder->resizeTensor(input, {1, queryBatch});
        }
        decoder->resizeSession(decSession);

        copyFloatVectorToTensor(encVideo, video);
        copyFloatVectorToTensor(encAspect, aspect);
        copyFloatVectorToTensor(decVideo, video);

        auto code = encoder->runSession(encSession);
        if (code != MNN::NO_ERROR) {
            throw std::runtime_error("encoder runSession failed: " + std::to_string(code));
        }
        auto* encMemory = encoder->getSessionOutput(encSession, "memory");
        if (encMemory == nullptr) {
            throw std::runtime_error("Missing encoder output: memory");
        }
        auto memoryHost = makeHostTensor(encMemory);
        encMemory->copyToHostTensor(memoryHost.get());
        decMemory->copyFromHostTensor(memoryHost.get());

        std::vector<float> xyz;
        std::vector<float> uv;
        std::vector<float> visibility;
        std::vector<float> confidence;
        xyz.reserve(static_cast<size_t>(totalQueries) * 3);
        uv.reserve(static_cast<size_t>(totalQueries) * 2);
        visibility.reserve(totalQueries);
        confidence.reserve(totalQueries);

        for (int offset = 0; offset < totalQueries; offset += queryBatch) {
            const int actual = std::min(queryBatch, totalQueries - offset);
            copyFloatBatchToTensor(decoder->getSessionInput(decSession, "u"), queryU, offset, queryBatch, actual);
            copyFloatBatchToTensor(decoder->getSessionInput(decSession, "v"), queryV, offset, queryBatch, actual);
            copyIntBatchToTensor(decoder->getSessionInput(decSession, "t_src"), queryTSrc, offset, queryBatch, actual);
            copyIntBatchToTensor(decoder->getSessionInput(decSession, "t_tgt"), queryTTgt, offset, queryBatch, actual);
            copyIntBatchToTensor(decoder->getSessionInput(decSession, "t_cam"), queryTCam, offset, queryBatch, actual);

            code = decoder->runSession(decSession);
            if (code != MNN::NO_ERROR) {
                throw std::runtime_error("decoder runSession failed at query offset " + std::to_string(offset) + ": " + std::to_string(code));
            }
            appendOutput(decoder.get(), decSession, "xyz_3d", queryBatch, actual, xyz);
            appendOutput(decoder.get(), decSession, "uv_2d", queryBatch, actual, uv);
            appendOutput(decoder.get(), decSession, "visibility", queryBatch, actual, visibility);
            appendOutput(decoder.get(), decSession, "confidence", queryBatch, actual, confidence);
            if (offset == 0 || offset + queryBatch >= totalQueries || ((offset / queryBatch) % 128 == 0)) {
                std::cout << "queries " << std::min(offset + queryBatch, totalQueries) << "/" << totalQueries << "\n";
            }
        }

        writeBinary(outputDir / "xyz_3d_f32.bin", xyz);
        writeBinary(outputDir / "uv_2d_f32.bin", uv);
        writeBinary(outputDir / "visibility_f32.bin", visibility);
        writeBinary(outputDir / "confidence_f32.bin", confidence);
        std::cout << "MNN OpenD4RT video inference completed: queries=" << totalQueries
                  << " forward_type=" << static_cast<int>(forwardType) << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
