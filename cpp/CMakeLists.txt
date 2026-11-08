
cmake_minimum_required(VERSION 3.23)

project(DefectInspectorCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

set(
    ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../external/onnxruntime-win-x64-1.31.1"
    CACHE PATH "ONNX Runtime SDK directory"
)

set(ORT_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ORT_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

foreach(required_file
    "${ORT_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORT_LIBRARY}"
    "${ORT_DLL}"
)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Missing ONNX Runtime file: ${required_file}")
    endif()
endforeach()

add_library(onnxruntime_sdk SHARED IMPORTED)

set_target_properties(onnxruntime_sdk PROPERTIES
    IMPORTED_IMPLIB "${ORT_LIBRARY}"
    IMPORTED_LOCATION "${ORT_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}"
)

add_executable(defect_inspector_cpp
    src/main.cpp
)

target_include_directories(defect_inspector_cpp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(defect_inspector_cpp PRIVATE
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    onnxruntime_sdk
    bcrypt
)

if(MSVC)
    target_compile_options(defect_inspector_cpp PRIVATE
        /W4
        /permissive-
        /EHsc
    )
endif()

add_custom_command(
    TARGET defect_inspector_cpp
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DLL}"
        "$<TARGET_FILE_DIR:defect_inspector_cpp>"
    COMMENT "Copying ONNX Runtime DLL"
)
