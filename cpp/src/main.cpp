
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;
using json = nlohmann::json;


std::string sha256_file(const std::string& filename) {
    std::ifstream stream(fs::path(filename), std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open ONNX model: " + filename);
    }

    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;

    auto cleanup = [&]() {
        if (hash != nullptr) {
            BCryptDestroyHash(hash);
            hash = nullptr;
        }
        if (algorithm != nullptr) {
            BCryptCloseAlgorithmProvider(algorithm, 0);
            algorithm = nullptr;
        }
    };

    auto check = [](NTSTATUS status, const char* operation) {
        if (status < 0) {
            throw std::runtime_error(
                std::string("SHA-256 ") + operation + " failed"
            );
        }
    };

    try {
        check(
            BCryptOpenAlgorithmProvider(
                &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0
            ),
            "provider initialization"
        );

        DWORD object_length = 0;
        DWORD returned = 0;

        check(
            BCryptGetProperty(
                algorithm,
                BCRYPT_OBJECT_LENGTH,
                reinterpret_cast<PUCHAR>(&object_length),
                sizeof(object_length),
                &returned,
                0
            ),
            "object length query"
        );

        if (object_length == 0 || returned != sizeof(object_length)) {
            throw std::runtime_error("Invalid SHA-256 object length");
        }

        std::vector<UCHAR> hash_object(object_length);

        check(
            BCryptCreateHash(
                algorithm, &hash,
                hash_object.data(), object_length,
                nullptr, 0, 0
            ),
            "hash creation"
        );

        std::array<UCHAR, 65536> buffer{};

        while (true) {
            stream.read(
                reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size())
            );

            const auto count = stream.gcount();

            if (count > 0) {
                check(
                    BCryptHashData(
                        hash, buffer.data(),
                        static_cast<ULONG>(count), 0
                    ),
                    "data processing"
                );
            }

            if (stream.eof()) {
                break;
            }
            if (!stream) {
                throw std::runtime_error("ONNX model read failed");
            }
        }

        std::array<UCHAR, 32> digest{};

        check(
            BCryptFinishHash(
                hash, digest.data(),
                static_cast<ULONG>(digest.size()), 0
            ),
            "hash finalization"
        );

        cleanup();

        constexpr char hex_digits[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);

        for (UCHAR byte : digest) {
            result.push_back(hex_digits[byte >> 4]);
            result.push_back(hex_digits[byte & 0x0f]);
        }

        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}


struct Config {
    int width;
    int height;
    int defect_class_id;
    float threshold;
    float pixel_scale;
    std::array<float, 3> mean;
    std::array<float, 3> std;
};

struct Arguments {
    std::string model;
    std::string metadata;
    std::string image;
    std::string input_tensor;
    std::string output_tensor;
    std::string output_json;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for: " + key);
        }

        const std::string value = argv[++i];

        if (key == "--model") {
            args.model = value;
        } else if (key == "--metadata") {
            args.metadata = value;
        } else if (key == "--image") {
            args.image = value;
        } else if (key == "--input-tensor") {
            args.input_tensor = value;
        } else if (key == "--output-tensor") {
            args.output_tensor = value;
        } else if (key == "--output-json") {
            args.output_json = value;
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.model.empty() || args.metadata.empty()) {
        throw std::runtime_error(
            "--model and --metadata are required"
        );
    }

    if (args.image.empty() == args.input_tensor.empty()) {
        throw std::runtime_error(
            "Specify exactly one of --image or --input-tensor"
        );
    }

    return args;
}

Config load_config(
    const std::string& filename,
    const std::string& model_filename
) {
    std::ifstream stream(filename);

    if (!stream) {
        throw std::runtime_error(
            "Cannot open metadata: " + filename
        );
    }

    const json metadata = json::parse(stream);


    const auto onnx_it = metadata.find("onnx");

    if (onnx_it == metadata.end() || !onnx_it->is_object()) {
        throw std::runtime_error("Missing or invalid ONNX SHA-256 in metadata");
    }

    const auto hash_it = onnx_it->find("sha256");

    if (hash_it == onnx_it->end() || !hash_it->is_string()) {
        throw std::runtime_error("Missing or invalid ONNX SHA-256 in metadata");
    }

    std::string expected_hash = hash_it->get<std::string>();

    const bool valid_hash =
        expected_hash.size() == 64 &&
        std::all_of(
            expected_hash.begin(),
            expected_hash.end(),
            [](char c) {
                return (c >= '0' && c <= '9') ||
                       (c >= 'a' && c <= 'f') ||
                       (c >= 'A' && c <= 'F');
            }
        );

    if (!valid_hash) {
        throw std::runtime_error("Missing or invalid ONNX SHA-256 in metadata");
    }

    for (char& c : expected_hash) {
        if (c >= 'A' && c <= 'F') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }

    if (sha256_file(model_filename) != expected_hash) {
        throw std::runtime_error(
            "ONNX SHA-256 mismatch: model and metadata differ"
        );
    }


    const auto& input = metadata.at("input");

    if (metadata.at("output_type").get<std::string>() != "logits") {
        throw std::runtime_error("Expected logits output");
    }

    if (
        input.at("color_space").get<std::string>() != "RGB" ||
        input.at("layout").get<std::string>() != "NCHW" ||
        input.at("dtype").get<std::string>() != "float32" ||
        input.at("resize_method").get<std::string>() != "stretch" ||
        input.at("interpolation").get<std::string>() != "bilinear"
    ) {
        throw std::runtime_error(
            "Unsupported preprocessing configuration"
        );
    }

    if (
        metadata.at("classes").at("0").get<std::string>() != "good" ||
        metadata.at("classes").at("1").get<std::string>() != "defect"
    ) {
        throw std::runtime_error("Unexpected class mapping");
    }

    Config config{};
    config.width = input.at("width").get<int>();
    config.height = input.at("height").get<int>();
    config.defect_class_id =
        metadata.at("defect_class_id").get<int>();
    config.threshold =
        metadata.at("defect_threshold").get<float>();
    config.pixel_scale =
        input.at("pixel_scale").get<float>();

    for (int c = 0; c < 3; ++c) {
        config.mean[c] = input.at("mean").at(c).get<float>();
        config.std[c] = input.at("std").at(c).get<float>();
    }

    if (
        config.width <= 0 ||
        config.height <= 0 ||
        config.defect_class_id != 1 ||
        !std::isfinite(config.threshold) ||
        config.threshold < 0.0f ||
        config.threshold > 1.0f ||
        std::abs(config.pixel_scale - 1.0f / 255.0f) > 1e-7f
    ) {
        throw std::runtime_error("Invalid metadata values");
    }

    for (int c = 0; c < 3; ++c) {
        if (
            !std::isfinite(config.mean[c]) ||
            !std::isfinite(config.std[c]) ||
            config.std[c] <= 0.0f
        ) {
            throw std::runtime_error(
                "Invalid normalization values"
            );
        }
    }

    return config;
}

std::size_t tensor_element_count(const Config& config) {
    return static_cast<std::size_t>(3) *
        static_cast<std::size_t>(config.height) *
        static_cast<std::size_t>(config.width);
}

std::vector<float> preprocess_image(
    const std::string& filename,
    const Config& config
) {
    const cv::Mat bgr = cv::imread(filename, cv::IMREAD_COLOR);

    if (bgr.empty()) {
        throw std::runtime_error("Cannot read image: " + filename);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(
        rgb,
        resized,
        cv::Size(config.width, config.height),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    const std::size_t plane =
        static_cast<std::size_t>(config.width) *
        static_cast<std::size_t>(config.height);

    std::vector<float> tensor(tensor_element_count(config));

    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            const cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);

            const std::size_t offset =
                static_cast<std::size_t>(y) *
                static_cast<std::size_t>(config.width) +
                static_cast<std::size_t>(x);

            for (int c = 0; c < 3; ++c) {
                const float scaled =
                    static_cast<float>(pixel[c]) *
                    config.pixel_scale;

                tensor[
                    static_cast<std::size_t>(c) * plane + offset
                ] = (scaled - config.mean[c]) / config.std[c];
            }
        }
    }

    return tensor;
}

void save_tensor(
    const std::string& filename,
    const std::vector<float>& tensor
) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream) {
        throw std::runtime_error(
            "Cannot write tensor: " + filename
        );
    }

    stream.write(
        reinterpret_cast<const char*>(tensor.data()),
        static_cast<std::streamsize>(tensor.size() * sizeof(float))
    );

    if (!stream) {
        throw std::runtime_error("Tensor write failed");
    }
}

std::vector<float> load_tensor(
    const std::string& filename,
    std::size_t expected_elements
) {
    std::ifstream stream(filename, std::ios::binary | std::ios::ate);

    if (!stream) {
        throw std::runtime_error(
            "Cannot read tensor: " + filename
        );
    }

    const auto file_size = stream.tellg();
    const auto expected_bytes =
        static_cast<std::streamoff>(expected_elements * sizeof(float));

    if (file_size != expected_bytes) {
        throw std::runtime_error("Tensor byte size mismatch");
    }

    stream.seekg(0);

    std::vector<float> tensor(expected_elements);

    stream.read(
        reinterpret_cast<char*>(tensor.data()),
        static_cast<std::streamsize>(expected_bytes)
    );

    if (!stream) {
        throw std::runtime_error("Tensor read failed");
    }

    return tensor;
}

std::array<float, 2> run_inference(
    const std::string& model_path,
    const Config& config,
    std::vector<float>& tensor
) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "defect_inspector");

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    Ort::Session session(
        env,
        fs::path(model_path).wstring().c_str(),
        options
    );

    if (session.GetInputCount() != 1 ||
        session.GetOutputCount() != 1) {
        throw std::runtime_error("Unexpected ONNX model inputs/outputs");
    }

    const std::array<int64_t, 4> shape = {
        1,
        3,
        config.height,
        config.width
    };

    const Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info,
        tensor.data(),
        tensor.size(),
        shape.data(),
        shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input,
        1,
        output_names,
        1
    );

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Invalid ONNX output");
    }

    const auto output_info =
        outputs[0].GetTensorTypeAndShapeInfo();

    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Expected float32 logits");
    }

    if (output_info.GetShape() != std::vector<int64_t>{1, 2}) {
        throw std::runtime_error("Expected logits shape [1, 2]");
    }

    const float* values = outputs[0].GetTensorData<float>();

    if (!std::isfinite(values[0]) || !std::isfinite(values[1])) {
        throw std::runtime_error("Non-finite logits");
    }

    return {values[0], values[1]};
}

double compute_defect_score(
    const std::array<float, 2>& logits,
    int defect_id
) {
    const double largest = std::max(
        static_cast<double>(logits[0]),
        static_cast<double>(logits[1])
    );

    const double e0 = std::exp(
        static_cast<double>(logits[0]) - largest
    );

    const double e1 = std::exp(
        static_cast<double>(logits[1]) - largest
    );

    return (defect_id == 0 ? e0 : e1) / (e0 + e1);
}

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);

        const Config config = load_config(args.metadata, args.model);


        std::vector<float> tensor;

        if (!args.image.empty()) {
            tensor = preprocess_image(args.image, config);
        } else {
            tensor = load_tensor(
                args.input_tensor,
                tensor_element_count(config)
            );
        }

        if (!args.output_tensor.empty()) {
            save_tensor(args.output_tensor, tensor);
        }

        const auto logits =
            run_inference(args.model, config, tensor);

        const double defect_score =
            compute_defect_score(logits, config.defect_class_id);

        const bool is_defect =
            defect_score >= static_cast<double>(config.threshold);

        const json result = {
            {"logits", {logits[0], logits[1]}},
            {"defect_score", defect_score},
            {"threshold", config.threshold},
            {"predicted_class_id", is_defect ? 1 : 0},
            {"predicted_class", is_defect ? "defect" : "good"},
            {"execution_provider", "CPUExecutionProvider"},
            {"input_shape", {
                1, 3, config.height, config.width
            }}
        };

        if (!args.output_json.empty()) {
            std::ofstream stream(args.output_json);

            if (!stream) {
                throw std::runtime_error("Cannot write output JSON");
            }

            stream << result.dump(2) << '\n';
        }

        std::cout << std::setprecision(10)
                  << result.dump(2) << '\n';

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
