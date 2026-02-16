#include <lean/lean.h>
#include <webgpu/webgpu.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <atomic>
#include <memory>
#include <future>

// Note: g_device removed - all functions now take device as parameter
// g_dawn_instance is kept global for deviceWait to use (since deviceWait only gets device, not instance)
static dawn::native::Instance* g_dawn_instance = nullptr;

// Global verbose flag for controlling debug output
static bool g_verbose = true;
#define DBG_PRINT(...) do { if (g_verbose) { __VA_ARGS__; } } while(0)

// ============================================================================
// Promise/Future Pattern for Async WebGPU Operations
// ============================================================================

// CallbackData structure for async operations (following gpu.hpp pattern)
struct CallbackData {
    WGPUBuffer buffer;
    size_t bufferSize;
    void* output;
    std::shared_ptr<std::promise<void>> promise;
    lean_object** result_ptr;  // Pointer to store final ByteArray for caller
};

// FutureData structure for GPU work completion futures
struct FutureData {
    std::shared_ptr<std::promise<void>> promise;  // Keep promise alive
    std::shared_ptr<std::future<void>> future;
    dawn::native::Instance* instance;  // Need instance for wait() template
};

// Template function to wait for future completion (following gpu.hpp pattern)
template<typename T>
T wait(wgpu::Instance& instance, std::future<T>& f) {
    while (f.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        instance.ProcessEvents();
    }
    if constexpr (std::is_void_v<T>) {
        f.get();
        return;
    } else {
        return f.get();
    }
}

// ============================================================================
// Error Scope for Validation and Runtime Errors
// ============================================================================

struct ErrorScopeData {
    bool errorOccurred;
    WGPUErrorType errorType;
    std::string errorMessage;
    std::shared_ptr<std::promise<void>> promise;
};

static void errorScopeCallback(WGPUPopErrorScopeStatus status, WGPUErrorType type,
                               WGPUStringView message, void* userdata1, void* /*userdata2*/) {
    ErrorScopeData* data = static_cast<ErrorScopeData*>(userdata1);

    if (status == WGPUPopErrorScopeStatus_Success && type != WGPUErrorType_NoError) {
        data->errorOccurred = true;
        data->errorType = type;
        data->errorMessage = std::string(message.data, message.length);

        fprintf(stderr, "\n");
        fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ GPU VALIDATION ERROR                                      ║\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "Error Type: ");
        switch (type) {
            case WGPUErrorType_Validation:
                fprintf(stderr, "VALIDATION\n");
                break;
            case WGPUErrorType_OutOfMemory:
                fprintf(stderr, "OUT OF MEMORY\n");
                break;
            case WGPUErrorType_Internal:
                fprintf(stderr, "INTERNAL\n");
                break;
            default:
                fprintf(stderr, "UNKNOWN (%d)\n", (int)type);
                break;
        }
        fprintf(stderr, "Message: %s\n", data->errorMessage.c_str());
        fprintf(stderr, "════════════════════════════════════════════════════════════\n\n");
        fflush(stderr);
    }

    data->promise->set_value();
}

// Helper to run an operation within an error scope
template<typename F>
bool runWithErrorScope(WGPUDevice device, dawn::native::Instance* nativeInstance, const char* opName, F&& operation) {
    // Push error scope
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);

    // Execute operation
    operation();

    // Pop error scope and wait for result
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();

    ErrorScopeData* errorData = new ErrorScopeData{
        .errorOccurred = false,
        .errorType = WGPUErrorType_NoError,
        .errorMessage = "",
        .promise = promise
    };

    WGPUPopErrorScopeCallbackInfo callbackInfo = {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = errorScopeCallback,
        .userdata1 = errorData,
        .userdata2 = nullptr
    };

    wgpuDevicePopErrorScope(device, callbackInfo);

    // Wait for error scope to complete - convert native instance to wgpu::Instance
    wgpu::Instance instance(nativeInstance->Get());
    wait(instance, future);

    bool hadError = errorData->errorOccurred;
    delete errorData;

    if (hadError) {
        fprintf(stderr, "[C++] %s failed due to validation errors (see above)\n", opName);
        fflush(stderr);
    }

    return !hadError;
}

// Helper macros to extract pointers from resource structures
// All resources maintain a reference to their parent to prevent premature GC
#define EXTRACT_DEVICE_PTR(device_struct) \
    static_cast<wgpu::Device*>(lean_get_external_data(lean_ctor_get(device_struct, 0)))

#define EXTRACT_BUFFER_PTR(buffer_struct) \
    static_cast<wgpu::Buffer*>(lean_get_external_data(lean_ctor_get(buffer_struct, 0)))

#define EXTRACT_SHADER_MODULE_PTR(shader_struct) \
    static_cast<wgpu::ShaderModule*>(lean_get_external_data(lean_ctor_get(shader_struct, 0)))

#define EXTRACT_COMPUTE_PIPELINE_PTR(pipeline_struct) \
    static_cast<wgpu::ComputePipeline*>(lean_get_external_data(lean_ctor_get(pipeline_struct, 0)))

#define EXTRACT_BIND_GROUP_PTR(bindgroup_struct) \
    static_cast<wgpu::BindGroup*>(lean_get_external_data(lean_ctor_get(bindgroup_struct, 0)))

#define EXTRACT_BIND_GROUP_LAYOUT_PTR(layout_struct) \
    static_cast<wgpu::BindGroupLayout*>(lean_get_external_data(lean_ctor_get(layout_struct, 0)))

#define EXTRACT_FUTURE_PTR(future_struct) \
    static_cast<FutureData*>(lean_get_external_data(lean_ctor_get(future_struct, 0)))

// Helper function to create resource structures that maintain device reference
// Returns: { ptr : ResourcePtr, device : Device }
inline lean_object* make_resource_with_device(lean_object* resource_external, b_lean_obj_arg device_obj) {
    lean_object* resource_struct = lean_alloc_ctor(0, 2, 0);  // constructor with 2 fields
    lean_inc(device_obj);  // Increment device refcount to keep it alive
    lean_ctor_set(resource_struct, 0, resource_external);  // field 0: ptr
    lean_ctor_set(resource_struct, 1, device_obj);          // field 1: device
    return resource_struct;
}

// External classes for WebGPU resource management
// Instance class is extern so glfw_bridge.cpp can use it
lean_external_class* g_webgpu_instance_class = nullptr;
static lean_external_class* g_webgpu_device_class = nullptr;
static lean_external_class* g_webgpu_buffer_class = nullptr;
static lean_external_class* g_webgpu_shader_module_class = nullptr;
static lean_external_class* g_webgpu_compute_pipeline_class = nullptr;
static lean_external_class* g_webgpu_bind_group_class = nullptr;
static lean_external_class* g_webgpu_bind_group_layout_class = nullptr;
static lean_external_class* g_webgpu_command_encoder_class = nullptr;
static lean_external_class* g_webgpu_future_class = nullptr;

extern "C" {

//=============================================================================
// Structured Error Construction Helpers
// These create Lean WebGPUError ADT constructors matching Hesper/WebGPU/Errors.lean
//=============================================================================

// DeviceError constructors (tag values match Lean inductive definition order)
namespace DeviceError {
    static lean_object* NoAdaptersFound() {
        return lean_alloc_ctor(0, 0, 0);  // NoAdaptersFound : DeviceError
    }

    static lean_object* InvalidAdapterIndex(uint32_t requested, uint32_t available) {
        lean_object* err = lean_alloc_ctor(1, 0, 8);  // Tag 1, 0 boxed, 2 scalars (uint32)
        lean_ctor_set_uint32(err, 0, requested);
        lean_ctor_set_uint32(err, 4, available);
        return err;
    }

    static lean_object* DeviceCreationFailed(uint32_t adapterIdx, const char* reason) {
        lean_object* err = lean_alloc_ctor(2, 1, 4);  // Tag 2, 1 boxed (string), 1 scalar (uint32)
        lean_ctor_set_uint32(err, sizeof(void*), adapterIdx);
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }

    static lean_object* InitializationFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(5, 1, 0);  // Tag 5, 1 boxed (string)
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }
}

// BufferError constructors
namespace BufferError {
    static lean_object* AllocationFailed(size_t size, const char* reason) {
        lean_object* err = lean_alloc_ctor(0, 1, sizeof(size_t));  // Tag 0
        lean_ctor_set_usize(err, sizeof(void*), size);
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }

    static lean_object* MappingFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(1, 1, 0);  // Tag 1
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }
}

// ShaderError constructors
namespace ShaderError {
    static lean_object* CompilationFailed(const char* source, const char* errors) {
        lean_object* err = lean_alloc_ctor(0, 2, 0);  // Tag 0, 2 strings
        lean_ctor_set(err, 0, lean_mk_string(source));
        lean_ctor_set(err, 1, lean_mk_string(errors));
        return err;
    }

    static lean_object* ValidationFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(1, 1, 0);  // Tag 1
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }
}

// PipelineError constructors
namespace PipelineError {
    static lean_object* CreationFailed(const char* pipelineType, const char* reason) {
        lean_object* err = lean_alloc_ctor(0, 2, 0);  // Tag 0, 2 strings
        lean_ctor_set(err, 0, lean_mk_string(pipelineType));
        lean_ctor_set(err, 1, lean_mk_string(reason));
        return err;
    }
}

// SurfaceError constructors
namespace SurfaceError {
    static lean_object* WindowCreationFailed(uint32_t width, uint32_t height, const char* reason) {
        lean_object* err = lean_alloc_ctor(0, 1, 8);  // Tag 0, 1 string, 2 uint32s
        lean_ctor_set_uint32(err, sizeof(void*), width);
        lean_ctor_set_uint32(err, sizeof(void*) + 4, height);
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }

    static lean_object* SurfaceCreationFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(1, 1, 0);  // Tag 1
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }

    static lean_object* ConfigurationFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(2, 1, 0);  // Tag 2
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }

    static lean_object* PresentationFailed(const char* reason) {
        lean_object* err = lean_alloc_ctor(3, 1, 0);  // Tag 3
        lean_ctor_set(err, 0, lean_mk_string(reason));
        return err;
    }
}

// WebGPUError top-level constructors (wraps specific error types)
namespace WebGPUError {
    static lean_object* Device(lean_object* deviceErr) {
        lean_object* err = lean_alloc_ctor(0, 1, 0);  // Tag 0: Device variant
        lean_ctor_set(err, 0, deviceErr);
        return err;
    }

    static lean_object* Buffer(lean_object* bufferErr) {
        lean_object* err = lean_alloc_ctor(1, 1, 0);  // Tag 1: Buffer variant
        lean_ctor_set(err, 0, bufferErr);
        return err;
    }

    static lean_object* Shader(lean_object* shaderErr) {
        lean_object* err = lean_alloc_ctor(2, 1, 0);  // Tag 2: Shader variant
        lean_ctor_set(err, 0, shaderErr);
        return err;
    }

    static lean_object* Pipeline(lean_object* pipelineErr) {
        lean_object* err = lean_alloc_ctor(3, 1, 0);  // Tag 3: Pipeline variant
        lean_ctor_set(err, 0, pipelineErr);
        return err;
    }

    static lean_object* Surface(lean_object* surfaceErr) {
        lean_object* err = lean_alloc_ctor(4, 1, 0);  // Tag 4: Surface variant
        lean_ctor_set(err, 0, surfaceErr);
        return err;
    }

    static lean_object* Unknown(const char* message) {
        lean_object* err = lean_alloc_ctor(6, 1, 0);  // Tag 6: Unknown variant
        lean_ctor_set(err, 0, lean_mk_string(message));
        return err;
    }
}

// Helper to create IO error from WebGPUError
static inline lean_object* make_webgpu_io_error(lean_object* webgpuErr) {
    return lean_io_result_mk_error(lean_mk_io_user_error(webgpuErr));
}

//=============================================================================
// WebGPU Finalizers - Called by Lean GC when resources are collected
//=============================================================================

static void finalize_webgpu_instance(void* ptr) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(ptr);
    if (instance) {
        if (g_verbose) std::cout << "[C++] Finalizing WebGPU Instance" << std::endl;
        delete instance;
    }
}

static void finalize_webgpu_device(void* ptr) {
    wgpu::Device* device = static_cast<wgpu::Device*>(ptr);
    if (device) {
        if (g_verbose) std::cout << "[C++] Finalizing WebGPU Device" << std::endl;
        delete device;
    }
}

static void finalize_webgpu_buffer(void* ptr) {
    wgpu::Buffer* buffer = (wgpu::Buffer*)ptr;
    if (buffer) {
        delete buffer;
    }
}

static void finalize_webgpu_shader_module(void* ptr) {
    wgpu::ShaderModule* module = (wgpu::ShaderModule*)ptr;
    if (module) {
        delete module;
    }
}

static void finalize_webgpu_compute_pipeline(void* ptr) {
    wgpu::ComputePipeline* pipeline = (wgpu::ComputePipeline*)ptr;
    if (pipeline) {
        delete pipeline;
    }
}

static void finalize_webgpu_bind_group(void* ptr) {
    wgpu::BindGroup* group = (wgpu::BindGroup*)ptr;
    if (group) {
        delete group;
    }
}

static void finalize_webgpu_bind_group_layout(void* ptr) {
    wgpu::BindGroupLayout* layout = (wgpu::BindGroupLayout*)ptr;
    if (layout) {
        delete layout;
    }
}

static void finalize_webgpu_command_encoder(void* ptr) {
    wgpu::CommandEncoder* encoder = (wgpu::CommandEncoder*)ptr;
    if (encoder) {
        delete encoder;
    }
}

static void finalize_webgpu_future(void* ptr) {
    FutureData* futureData = static_cast<FutureData*>(ptr);
    if (futureData) {
        if (g_verbose) fprintf(stderr, "[C++] Finalizing GPU Future (ptr=%p)\n", ptr);
        fflush(stderr);
        delete futureData;
    }
}

//=============================================================================
// External Class Registration for WebGPU
//=============================================================================

// No-op foreach: external WebGPU objects don't contain Lean sub-objects,
// but lean_mark_mt requires a non-null foreach function to avoid null call.
static void noop_foreach(void*, b_lean_obj_arg) {}

static void register_webgpu_external_classes() {
    if (g_webgpu_instance_class != nullptr) return;  // Already registered

    g_webgpu_instance_class = lean_register_external_class(finalize_webgpu_instance, noop_foreach);
    g_webgpu_device_class = lean_register_external_class(finalize_webgpu_device, noop_foreach);
    g_webgpu_buffer_class = lean_register_external_class(finalize_webgpu_buffer, noop_foreach);
    g_webgpu_shader_module_class = lean_register_external_class(finalize_webgpu_shader_module, noop_foreach);
    g_webgpu_compute_pipeline_class = lean_register_external_class(finalize_webgpu_compute_pipeline, noop_foreach);
    g_webgpu_bind_group_class = lean_register_external_class(finalize_webgpu_bind_group, noop_foreach);
    g_webgpu_bind_group_layout_class = lean_register_external_class(finalize_webgpu_bind_group_layout, noop_foreach);
    g_webgpu_command_encoder_class = lean_register_external_class(finalize_webgpu_command_encoder, noop_foreach);
    g_webgpu_future_class = lean_register_external_class(finalize_webgpu_future, noop_foreach);

    if (g_verbose) std::cout << "[C++] WebGPU External classes registered" << std::endl;
}

// Lean FFI: Set verbose logging
lean_obj_res lean_hesper_set_verbose(uint8_t verbose, lean_obj_res /* unit */) {
    g_verbose = (verbose != 0);
    return lean_io_result_mk_ok(lean_box(0));
}

// Lean FFI: Initialize Dawn and list GPUs
lean_obj_res lean_hesper_init(lean_obj_res /* unit */) {
    // Register external classes for GC management
    register_webgpu_external_classes();

    // 1. Setup ProcTable (Critical for Native Dawn)
    dawnProcSetProcs(&dawn::native::GetProcs());

    // 2. Create Instance with timed wait support for WaitAny API
    WGPUInstanceFeatureName timedWaitFeature = WGPUInstanceFeatureName_TimedWaitAny;

    WGPUInstanceLimits instanceLimits = {};
    instanceLimits.nextInChain = nullptr;
    instanceLimits.timedWaitAnyMaxCount = 64;  // Support up to 64 concurrent waits

    WGPUInstanceDescriptor instanceDesc = {};
    instanceDesc.nextInChain = nullptr;
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = &timedWaitFeature;
    instanceDesc.requiredLimits = &instanceLimits;

    dawn::native::Instance* instance = new dawn::native::Instance(&instanceDesc);

    // Store global instance pointer for deviceWait to use
    g_dawn_instance = instance;

    // 3. Enumerate Adapters (replaces DiscoverDefaultAdapters + GetAdapters)
    auto adapters = instance->EnumerateAdapters();

    if (g_verbose) std::cout << "[Hesper] Initialized. Found " << adapters.size() << " adapters:" << std::endl;

    for (const auto& adapter : adapters) {
        WGPUAdapter wgpuAdapter = adapter.Get();

        // Get adapter info using the C++ WebGPU API
        wgpu::AdapterInfo info;
        wgpu::Adapter(wgpuAdapter).GetInfo(&info);

        // Convert StringView to std::string_view for printing
        std::string_view descView(info.description.data, info.description.length);

        if (g_verbose) std::cout << "  - " << descView
                  << " (Backend: ";

        switch (info.backendType) {
            case wgpu::BackendType::Vulkan:
                if (g_verbose) std::cout << "Vulkan";
                break;
            case wgpu::BackendType::Metal:
                if (g_verbose) std::cout << "Metal";
                break;
            case wgpu::BackendType::D3D11:
                if (g_verbose) std::cout << "D3D11";
                break;
            case wgpu::BackendType::D3D12:
                if (g_verbose) std::cout << "D3D12";
                break;
            case wgpu::BackendType::OpenGL:
                if (g_verbose) std::cout << "OpenGL";
                break;
            case wgpu::BackendType::OpenGLES:
                if (g_verbose) std::cout << "OpenGLES";
                break;
            default:
                if (g_verbose) std::cout << "Other";
                break;
        }

        if (g_verbose) std::cout << ")" << std::endl;
    }

    // Wrap in External object and return
    lean_object* external = lean_alloc_external(g_webgpu_instance_class, instance);
    return lean_io_result_mk_ok(external);
}

// ============================================================================
// WebGPU Wrapper FFI Functions
// ============================================================================

// Device Limits & Features Configuration
// Reference: https://github.com/AnswerDotAI/gpu.cpp/blob/dev/experimental/kernels/ops.hpp#L14
// Reference: https://github.com/google/dawn/blob/a8fbe981a86cb59536e2de423d2013a82d9b54a0/src/dawn/native/Limits.cpp

static WGPULimits getMaxLimits() {
    WGPULimits limits = WGPU_LIMITS_INIT;
    limits.nextInChain = nullptr;
    limits.maxTextureDimension1D = 8192;
    limits.maxTextureDimension2D = 8192;
    limits.maxTextureDimension3D = 2048;
    limits.maxTextureArrayLayers = 256;
    limits.maxBindGroups = 4;
    limits.maxBindGroupsPlusVertexBuffers = 24;
    limits.maxBindingsPerBindGroup = 1000;
    limits.maxDynamicUniformBuffersPerPipelineLayout = 8;
    limits.maxDynamicStorageBuffersPerPipelineLayout = 4;
    limits.maxSampledTexturesPerShaderStage = 16;
    limits.maxSamplersPerShaderStage = 16;
    limits.maxStorageBuffersPerShaderStage = 8;
    limits.maxStorageTexturesPerShaderStage = 4;
    limits.maxUniformBuffersPerShaderStage = 12;
    limits.maxUniformBufferBindingSize = 65536;
    limits.maxStorageBufferBindingSize = 0x80000000;   // 2 GB (same as maxBufferSize)
    limits.minUniformBufferOffsetAlignment = 256;
    limits.minStorageBufferOffsetAlignment = 256;
    limits.maxVertexBuffers = 8;
    limits.maxBufferSize = 0x80000000;               // 2 GB
    limits.maxVertexAttributes = 16;
    limits.maxVertexBufferArrayStride = 2048;
    limits.maxInterStageShaderVariables = 16;
    limits.maxColorAttachments = 8;
    limits.maxColorAttachmentBytesPerSample = 32;
    limits.maxComputeWorkgroupStorageSize = 16384;
    limits.maxComputeInvocationsPerWorkgroup = 256;
    limits.maxComputeWorkgroupSizeX = 256;
    limits.maxComputeWorkgroupSizeY = 256;
    limits.maxComputeWorkgroupSizeZ = 64;
    limits.maxComputeWorkgroupsPerDimension = 65535;
    return limits;
}

// Shared device creation: max limits + subgroup matrix + ShaderF16
static wgpu::Device createDeviceWithMaxLimits(wgpu::Adapter& adapter) {
    // Toggles: enable experimental APIs for subgroup matrix
    static WGPUDawnTogglesDescriptor toggles = {};
    toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    static const char* enableList[] = {"allow_unsafe_apis"};
    toggles.enabledToggles = enableList;
    toggles.enabledToggleCount = 1;

    // Features: ShaderF16, Subgroups, SubgroupMatrix
    static std::array<WGPUFeatureName, 3> features = {
        WGPUFeatureName_ShaderF16,
        WGPUFeatureName_Subgroups,
        WGPUFeatureName_ChromiumExperimentalSubgroupMatrix
    };

    // Limits: max settings for large model buffers (1 GB storage, 2 GB buffer)
    WGPULimits limits = getMaxLimits();

    // Assemble device descriptor
    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.nextInChain = reinterpret_cast<const wgpu::ChainedStruct*>(&toggles.chain);
    deviceDesc.requiredFeatureCount = features.size();
    deviceDesc.requiredFeatures = reinterpret_cast<const wgpu::FeatureName*>(features.data());
    deviceDesc.requiredLimits = reinterpret_cast<const wgpu::Limits*>(&limits);

    wgpu::Device device = adapter.CreateDevice(&deviceDesc);
    if (device) {
        if (g_verbose) std::cout << "[Hesper] Device created with max limits + subgroup matrix support" << std::endl;
        if (g_verbose) std::cout << "  maxStorageBufferBindingSize: " << limits.maxStorageBufferBindingSize << std::endl;
        if (g_verbose) std::cout << "  maxBufferSize: " << limits.maxBufferSize << std::endl;
    }
    return device;
}

// Device Management
// Create device with advanced features (subgroup matrix, F16, etc.)
lean_obj_res lean_hesper_get_device_with_features(b_lean_obj_arg instance_obj, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();
    if (adapters.empty()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    wgpu::Adapter adapter(adapters[0].Get());
    wgpu::Device device = createDeviceWithMaxLimits(adapter);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(0, "Failed to create device with subgroup features")));
    }

    // Properly allocate device on heap and wrap in External object
    wgpu::Device* devicePtr = new wgpu::Device(device);
    lean_object* device_external = lean_alloc_external(g_webgpu_device_class, devicePtr);

    // Create Device structure: { ptr : DevicePtr, parentInstance : Instance }
    // This keeps the Instance alive as long as the Device is alive (prevents premature GC)
    lean_object* device_struct = lean_alloc_ctor(0, 2, 0);  // constructor with 2 fields, 0 scalars
    lean_inc(instance_obj);  // Increment refcount of instance so it stays alive
    lean_ctor_set(device_struct, 0, device_external);  // field 0: ptr
    lean_ctor_set(device_struct, 1, instance_obj);      // field 1: parentInstance

    return lean_io_result_mk_ok(device_struct);
}

lean_obj_res lean_hesper_get_device(b_lean_obj_arg instance_obj, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();
    if (adapters.empty()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    wgpu::Adapter adapter(adapters[0].Get());
    wgpu::Device device = createDeviceWithMaxLimits(adapter);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(0,
            "Failed to create device")));
    }

    // Allocate device on heap and wrap in External object
    wgpu::Device* devicePtr = new wgpu::Device(device);
    lean_object* device_external = lean_alloc_external(g_webgpu_device_class, devicePtr);

    // Create Device structure: { ptr : DevicePtr, instance : Instance }
    // This keeps the Instance alive as long as the Device is alive (prevents premature GC)
    lean_object* device_struct = lean_alloc_ctor(0, 2, 0);  // constructor with 2 fields, 0 scalars
    lean_inc(instance_obj);  // Increment refcount of instance so it stays alive
    lean_ctor_set(device_struct, 0, device_external);  // field 0: ptr
    lean_ctor_set(device_struct, 1, instance_obj);      // field 1: parentInstance

    return lean_io_result_mk_ok(device_struct);
}

// Multi-GPU support functions

// Get the number of available GPU adapters
lean_obj_res lean_hesper_get_adapter_count(b_lean_obj_arg instance_obj, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();
    return lean_io_result_mk_ok(lean_box(adapters.size()));
}

// Lean FFI: High-precision timing for benchmarks (nanoseconds)
lean_obj_res lean_hesper_get_time_ns(lean_obj_res /* unit */) {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return lean_io_result_mk_ok(lean_box_uint64(static_cast<uint64_t>(ns)));
}

// Get GPU adapter information by index
lean_obj_res lean_hesper_get_adapter_info(b_lean_obj_arg instance_obj, uint32_t gpuIdx, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();

    if (gpuIdx >= adapters.size()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::InvalidAdapterIndex(gpuIdx, adapters.size())));
    }

    WGPUAdapter wgpuAdapter = adapters[gpuIdx].Get();
    wgpu::AdapterInfo info;
    wgpu::Adapter(wgpuAdapter).GetInfo(&info);

    // Create Lean string for description
    std::string_view descView(info.description.data, info.description.length);
    std::string desc(descView);

    // Create Lean structure with GPU info
    // Structure: { name : String, backendType : Nat }
    lean_object* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, lean_mk_string(desc.c_str()));
    lean_ctor_set(result, 1, lean_box(static_cast<uint32_t>(info.backendType)));

    return lean_io_result_mk_ok(result);
}

// Create a device from a specific GPU adapter index
lean_obj_res lean_hesper_get_device_by_index(b_lean_obj_arg instance_obj, uint32_t gpuIdx, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();

    if (adapters.empty()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    if (gpuIdx >= adapters.size()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::InvalidAdapterIndex(gpuIdx, adapters.size())));
    }

    if (g_verbose) std::cout << "[Hesper] Creating device from GPU adapter[" << gpuIdx << "]" << std::endl;

    // Wrap adapter
    wgpu::Adapter adapter(adapters[gpuIdx].Get());

    // Print adapter info
    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    std::string_view descView(info.description.data, info.description.length);
    if (g_verbose) std::cout << "[Hesper] Using GPU: " << descView << std::endl;

    // Create device with max limits + subgroup matrix features
    wgpu::Device device = createDeviceWithMaxLimits(adapter);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(gpuIdx, "Failed to create device")));
    }

    if (g_verbose) std::cout << "[Hesper] Device created successfully from adapter " << gpuIdx << std::endl;

    // Properly allocate device on heap and wrap in External object
    wgpu::Device* devicePtr = new wgpu::Device(device);
    lean_object* external = lean_alloc_external(g_webgpu_device_class, devicePtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_release_device(b_lean_obj_arg /* device */, lean_obj_res /* unit */) {
    // In this simple implementation, we keep the device alive
    // Real impl would use handle map and release specific devices
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_device_tick(b_lean_obj_arg device_obj, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    device->Tick();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_device_wait(b_lean_obj_arg future_struct, lean_obj_res /* unit */) {
    // Extract FutureData from Future structure
    FutureData* futureData = EXTRACT_FUTURE_PTR(future_struct);

    if (g_verbose) fprintf(stderr, "[C++] deviceWait: waiting for GPU work to complete...\n");
    fflush(stderr);

    // Wait for GPU work to complete using the wait() template
    wgpu::Instance instance(futureData->instance->Get());
    wait(instance, *(futureData->future));

    if (g_verbose) fprintf(stderr, "[C++] deviceWait: GPU work completed\n");
    fflush(stderr);

    return lean_io_result_mk_ok(lean_box(0));
}

// Buffer Operations

lean_obj_res lean_hesper_create_buffer(b_lean_obj_arg device_obj, b_lean_obj_arg desc, lean_obj_res /* unit */) {
    // Extract device from External object
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    // C struct matching Lean's BufferDescriptor memory layout
    // structure BufferDescriptor where
    //   size : USize               -- scalar
    //   usage : List BufferUsage   -- boxed
    //   mappedAtCreation : Bool    -- scalar
    // Lean reorders: boxed first, then scalars by size
    typedef struct {
        lean_object* usage;         // Boxed field (pointer)
        size_t size;                // Scalar (USize, 8 bytes)
        uint8_t mappedAtCreation;   // Scalar (Bool, 1 byte)
    } BufferDescriptor_Raw;

    BufferDescriptor_Raw* raw_desc = (BufferDescriptor_Raw*)lean_ctor_obj_cptr(desc);

    if (g_verbose) std::cout << "[C++] createBuffer: size=" << raw_desc->size
              << ", mapped=" << (raw_desc->mappedAtCreation ? "true" : "false") << std::endl;

    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = raw_desc->size;
    // TODO: Parse usage list from raw_desc->usage - for now use default
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    // HARDCODE mappedAtCreation to false to test
    bufferDesc.mappedAtCreation = false;  // ALWAYS FALSE for now

    if (g_verbose) fprintf(stderr, "[C++] Creating buffer with usage=0x%x\n", (unsigned)bufferDesc.usage);
    fflush(stderr);

    if (g_verbose) std::cout << "  bufferDesc.mappedAtCreation = " << bufferDesc.mappedAtCreation << " (HARDCODED TO FALSE)" << std::endl;

    wgpu::Buffer buffer = device->CreateBuffer(&bufferDesc);

    if (g_verbose) std::cout << "  Buffer created, wrapping in Lean External..." << std::endl;

    // Wrap in External with registered class
    wgpu::Buffer* bufferPtr = new wgpu::Buffer(buffer);

    if (g_verbose) fprintf(stderr, "  [BUFFER CREATE] wrapper=%p, WGPUBuffer=%p\n", (void*)bufferPtr, (void*)buffer.Get());
    fflush(stderr);

    lean_object* external = lean_alloc_external(g_webgpu_buffer_class, bufferPtr);

    if (g_verbose) std::cout << "  Created Lean external object at " << (void*)external << std::endl;

    // Wrap in structure with device reference to keep device (and instance) alive
    lean_object* buffer_struct = make_resource_with_device(external, device_obj);

    lean_object* result = lean_io_result_mk_ok(buffer_struct);

    if (g_verbose) std::cout << "  Returning IO result" << std::endl;
    if (g_verbose) std::cout.flush();

    return result;
}

lean_obj_res lean_hesper_write_buffer(b_lean_obj_arg device_obj, b_lean_obj_arg buffer_obj, size_t offset,
                                       b_lean_obj_arg data, lean_obj_res /* unit */) {
    // Extract device and buffer from External objects
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(buffer_obj);

    // Extract ByteArray from Lean
    // ByteArray is a Lean external object wrapping a byte buffer
    size_t data_size = lean_sarray_size(data);
    const uint8_t* data_ptr = lean_sarray_cptr(data);

    if (g_verbose) fprintf(stderr, "[C++] writeBuffer START: offset=%zu, size=%zu\n", offset, data_size);
    if (g_verbose) fprintf(stderr, "[C++]   [BUFFER WRITE] wrapper=%p, WGPUBuffer=%p\n", (void*)buffer, (void*)buffer->Get());
    fflush(stderr);

    // Print data being written (as hex and interpret as floats if size is multiple of 4)
    if (g_verbose && data_size > 0) {
        fprintf(stderr, "[C++]   Data being written (%zu bytes, hex): ", data_size);
        for (size_t i = 0; i < data_size && i < 32; i++) {
            fprintf(stderr, "%02x ", data_ptr[i]);
        }
        fprintf(stderr, "\n");

        // Interpret as floats if size is multiple of 4
        if (data_size % 4 == 0 && data_size >= 4) {
            const float* float_ptr = reinterpret_cast<const float*>(data_ptr);
            size_t num_floats = data_size / 4;
            fprintf(stderr, "[C++]   Data as floats: ");
            for (size_t i = 0; i < num_floats && i < 8; i++) {
                fprintf(stderr, "%.6f ", float_ptr[i]);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }

    if (g_verbose) fprintf(stderr, "[C++]   Calling WriteBuffer...\n");
    fflush(stderr);

    // Write data to GPU buffer
    try {
        device->GetQueue().WriteBuffer(*buffer, offset, data_ptr, data_size);
        if (g_verbose) fprintf(stderr, "[C++]   WriteBuffer returned\n");
        fflush(stderr);
    } catch (const std::exception& e) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR in WriteBuffer: %s\n", e.what());
        fflush(stderr);
        return make_webgpu_io_error(WebGPUError::Buffer(BufferError::AllocationFailed(0, e.what())));
    }

    // Process pending GPU operations
    if (g_verbose) std::cout << "[C++] writeBuffer: ticking device..." << std::endl;
    for (int i = 0; i < 10; i++) {
        device->Tick();
    }
    if (g_verbose) std::cout << "[C++] writeBuffer: complete" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
}

// Static callback functions for async buffer mapping (following gpu.hpp pattern)

static void bufferMapCallback(WGPUMapAsyncStatus status, WGPUStringView message,
                               void* userdata1, void* /*userdata2*/) {
    CallbackData* cbData = static_cast<CallbackData*>(userdata1);

    if (g_verbose) fprintf(stderr, "[C++] MapAsync callback: status=%d\n", static_cast<int>(status));
    fflush(stderr);

    if (status != WGPUMapAsyncStatus_Success) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: MapAsync failed with status=%d\n", static_cast<int>(status));
        if (message.length > 0) {
            if (g_verbose) fprintf(stderr, "[C++] Message: %.*s\n",
                    static_cast<int>(message.length), message.data);
        }
        fflush(stderr);
        cbData->promise->set_value();  // Signal completion even on error
        delete cbData;
        return;
    }

    // Get mapped data
    const void* mappedData = wgpuBufferGetConstMappedRange(cbData->buffer, 0, cbData->bufferSize);

    if (g_verbose) fprintf(stderr, "[C++] mappedData=%p, cbData->output=%p, cbData->result_ptr=%p\n",
            mappedData, cbData->output, cbData->result_ptr);
    fflush(stderr);

    if (mappedData && cbData->output) {
        // Copy data to output ByteArray
        lean_object* byte_array = static_cast<lean_object*>(cbData->output);
        const uint8_t* src = static_cast<const uint8_t*>(mappedData);

        // Print first 16 bytes for debugging
        if (g_verbose && cbData->bufferSize >= 16) {
            fprintf(stderr, "[C++]   Mapped data (hex): ");
            for (size_t i = 0; i < 16; i++) {
                fprintf(stderr, "%02x ", src[i]);
            }
            fprintf(stderr, "\n");

            const float* float_ptr = reinterpret_cast<const float*>(src);
            fprintf(stderr, "[C++]   Mapped data as floats: %.2f, %.2f, %.2f, %.2f\n",
                    float_ptr[0], float_ptr[1], float_ptr[2], float_ptr[3]);
            fflush(stderr);
        }

        // Push bytes to Lean ByteArray
        for (size_t i = 0; i < cbData->bufferSize; i++) {
            byte_array = lean_byte_array_push(byte_array, src[i]);
        }

        // Store the final ByteArray for the caller to retrieve
        if (cbData->result_ptr) {
            if (g_verbose) fprintf(stderr, "[C++] Updating result_ptr: old array size=%zu, new array size=%zu\n",
                    lean_sarray_size(*cbData->result_ptr), lean_sarray_size(byte_array));
            fflush(stderr);
            lean_dec(*cbData->result_ptr);  // Release old empty array
            *cbData->result_ptr = byte_array;  // Store new array with data
            if (g_verbose) fprintf(stderr, "[C++] After update: result_ptr now points to array with size=%zu\n",
                    lean_sarray_size(*cbData->result_ptr));
            fflush(stderr);
        } else {
            if (g_verbose) fprintf(stderr, "[C++] ERROR: result_ptr is NULL!\n");
            fflush(stderr);
        }

        if (g_verbose) fprintf(stderr, "[C++] Copied %zu bytes to ByteArray\n", cbData->bufferSize);
        fflush(stderr);
    } else {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: Skipping copy - mappedData=%p, output=%p\n",
                mappedData, cbData->output);
        fflush(stderr);
    }

    // Unmap buffer
    wgpuBufferUnmap(cbData->buffer);

    // Signal completion
    cbData->promise->set_value();
    delete cbData;
}

static void queueWorkDoneCallback(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                                  void* userdata1, void* /*userdata2*/) {
    CallbackData* cbData = static_cast<CallbackData*>(userdata1);

    if (g_verbose) fprintf(stderr, "[C++] OnSubmittedWorkDone: status=%d\n", static_cast<int>(status));
    if (message.length > 0) {
        if (g_verbose) fprintf(stderr, "[C++] Message: %.*s\n",
                static_cast<int>(message.length), message.data);
    }
    fflush(stderr);

    if (status != WGPUQueueWorkDoneStatus_Success) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: GPU work failed with status=%d\n", static_cast<int>(status));
        if (g_verbose) fprintf(stderr, "[C++] This usually means:\n");
        if (g_verbose) fprintf(stderr, "[C++]   - Shader validation error (check WGSL above)\n");
        if (g_verbose) fprintf(stderr, "[C++]   - Buffer binding mismatch\n");
        if (g_verbose) fprintf(stderr, "[C++]   - Resource usage conflict\n");
        fflush(stderr);
        cbData->promise->set_value();  // Signal completion even on error
        delete cbData;
        return;
    }

    if (g_verbose) fprintf(stderr, "[C++] GPU work completed successfully, now mapping buffer...\n");
    fflush(stderr);

    // Chain to buffer map operation
    WGPUBufferMapCallbackInfo mapCallbackInfo = {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = bufferMapCallback,
        .userdata1 = cbData,  // Pass through the same CallbackData
        .userdata2 = nullptr
    };

    wgpuBufferMapAsync(cbData->buffer, WGPUMapMode_Read, 0, cbData->bufferSize, mapCallbackInfo);
}

lean_obj_res lean_hesper_map_buffer_read(b_lean_obj_arg device_obj, b_lean_obj_arg buffer_obj,
                                          size_t offset, size_t size, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(buffer_obj);

    if (g_verbose) fprintf(stderr, "[C++] mapBufferRead: offset=%zu, size=%zu\n", offset, size);
    if (g_verbose) fprintf(stderr, "[C++]   [BUFFER READ] wrapper=%p, WGPUBuffer=%p\n", (void*)buffer, (void*)buffer->Get());
    fflush(stderr);

    if (!g_dawn_instance) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: g_dawn_instance is null!\n");
        fflush(stderr);
        return lean_io_result_mk_ok(lean_mk_empty_byte_array(lean_box(0)));
    }

    wgpu::Instance instance(g_dawn_instance->Get());

    // Following gpu.hpp toCPUAsync pattern:
    // 1. Create readback buffer (staging buffer for CPU read)
    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = size;
    stagingDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    stagingDesc.mappedAtCreation = false;

    wgpu::Buffer stagingBuffer = device->CreateBuffer(&stagingDesc);

    if (g_verbose) fprintf(stderr, "[C++] Created staging buffer (readback buffer)\n");
    fflush(stderr);

    // 2. Create command encoder and copy from GPU buffer to staging buffer
    wgpu::CommandEncoder encoder = device->CreateCommandEncoder();
    encoder.CopyBufferToBuffer(*buffer, offset, stagingBuffer, 0, size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // 3. Submit the command
    device->GetQueue().Submit(1, &commands);

    if (g_verbose) fprintf(stderr, "[C++] Submitted copy command to GPU queue\n");
    fflush(stderr);

    // 4. Set up promise/future pattern for async buffer mapping
    // Create callback data structure for chained async operations
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();

    // Allocate buffer for mapped data (will be filled by callback)
    lean_object* result_byte_array = lean_mk_empty_byte_array(lean_box(0));
    lean_inc(result_byte_array);  // Keep alive during async operations

    // Set up callback data
    CallbackData* cbData = new CallbackData{
        .buffer = stagingBuffer.Get(),
        .bufferSize = size,
        .output = result_byte_array,
        .promise = promise,
        .result_ptr = &result_byte_array  // Pass pointer so callback can update it
    };

    // 5. Start the async chain with OnSubmittedWorkDone
    if (g_verbose) fprintf(stderr, "[C++] Waiting for previous GPU work to complete...\n");
    fflush(stderr);

    WGPUQueueWorkDoneCallbackInfo workDoneInfo = {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = queueWorkDoneCallback,
        .userdata1 = cbData,
        .userdata2 = nullptr
    };

    wgpuQueueOnSubmittedWorkDone(device->GetQueue().Get(), workDoneInfo);

    // 6. Wait for the entire async chain to complete using promise/future pattern
    if (g_verbose) fprintf(stderr, "[C++] Waiting for async operations to complete...\n");
    fflush(stderr);

    wait(instance, future);

    if (g_verbose) fprintf(stderr, "[C++] All async operations completed\n");
    fflush(stderr);

    // 7. Return the ByteArray (data was filled by the callback via result_ptr)
    size_t final_size = lean_sarray_size(result_byte_array);
    if (g_verbose) fprintf(stderr, "[C++] Returning ByteArray (size=%zu, expected=%zu)\n", final_size, size);
    fflush(stderr);

    // lean_io_result_mk_ok takes ownership
    // Note: We don't need lean_dec here because the callback already managed refcounts:
    // - It dec'd the old empty array
    // - It stored the new array with data (which already has correct refcount from lean_byte_array_push)
    return lean_io_result_mk_ok(result_byte_array);
}

lean_obj_res lean_hesper_unmap_buffer(b_lean_obj_arg buffer, lean_obj_res /* unit */) {
    return lean_io_result_mk_ok(lean_box(0));
}

// Return a stable unique identifier for a GPU buffer (raw WGPUBuffer handle as UInt64)
lean_obj_res lean_hesper_buffer_id(b_lean_obj_arg buffer_obj, lean_obj_res /* unit */) {
    wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(buffer_obj);
    uint64_t id = (uint64_t)(void*)buffer->Get();
    return lean_io_result_mk_ok(lean_box_uint64(id));
}

// Hash an array of buffers into a single UInt64 key (avoids N separate FFI calls)
lean_obj_res lean_hesper_hash_buffer_array(uint64_t seed, b_lean_obj_arg buffers_array, lean_obj_res /* unit */) {
    size_t n = lean_array_size(buffers_array);
    uint64_t h = seed;
    for (size_t i = 0; i < n; i++) {
        lean_object* buf_obj = lean_array_get_core(buffers_array, i);
        wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(buf_obj);
        uint64_t id = (uint64_t)(void*)buffer->Get();
        // FNV-1a style mixing
        h ^= id;
        h *= 0x100000001b3ULL;
    }
    return lean_io_result_mk_ok(lean_box_uint64(h));
}

// Shader Operations

lean_obj_res lean_hesper_create_shader_module(b_lean_obj_arg device_obj, b_lean_obj_arg source, lean_obj_res /* unit */) {
    // Extract device from External object
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    // Extract shader source from Lean string
    const char* shader_code = lean_string_cstr(source);

    if (g_verbose) {
        fprintf(stderr, "[C++] createShaderModule: compiling WGSL...\n");
        fprintf(stderr, "[C++] ======================================\n");
        fprintf(stderr, "[C++] Shader source (length=%zu):\n", strlen(shader_code));
        fprintf(stderr, "--- BEGIN SHADER ---\n%s\n--- END SHADER ---\n", shader_code);
        fprintf(stderr, "[C++] ======================================\n");
        fflush(stderr);
    }

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shader_code;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;

    // Create shader module within error scope to catch validation errors
    wgpu::ShaderModule shaderModule;
    bool success = runWithErrorScope(device->Get(), g_dawn_instance, "CreateShaderModule", [&]() {
        shaderModule = device->CreateShaderModule(&shaderDesc);
    });

    // Check if shader module creation succeeded
    if (!shaderModule || !success) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ CRITICAL ERROR: Shader Module Creation Failed            ║\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
        if (g_verbose) fprintf(stderr, "[C++] CreateShaderModule returned null handle or had validation errors\n");
        fprintf(stderr, "════════════════════════════════════════════════════════════\n\n");
        fflush(stderr);
        lean_object* error_msg = lean_mk_string("Shader module creation failed - check stderr for compilation errors");
        return lean_io_result_mk_error(error_msg);
    }

    if (g_verbose) fprintf(stderr, "[C++] Shader module created successfully\n");
    fflush(stderr);

    // Wrap in External with registered class
    wgpu::ShaderModule* modulePtr = new wgpu::ShaderModule(shaderModule);
    lean_object* external = lean_alloc_external(g_webgpu_shader_module_class, modulePtr);

    // Wrap in structure with device reference to keep device alive
    lean_object* shader_struct = make_resource_with_device(external, device_obj);

    return lean_io_result_mk_ok(shader_struct);
}

// Pipeline Operations

lean_obj_res lean_hesper_create_bind_group_layout(b_lean_obj_arg device_obj, b_lean_obj_arg entries_array, lean_obj_res /* unit */) {
    // Extract device from External object
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    // Extract array of BindGroupLayoutEntry
    size_t num_entries = lean_array_size(entries_array);

    if (g_verbose) fprintf(stderr, "[C++] createBindGroupLayout: %zu entries\n", num_entries);
    fflush(stderr);

    std::vector<wgpu::BindGroupLayoutEntry> layoutEntries(num_entries);

    // CRITICAL: Must properly extract binding info from Lean structures
    //
    // BUG FIX: Previously this was hardcoded and ignored Lean data entirely,
    // causing GPU to receive incorrect binding configurations and return zeros.
    //
    // Lean struct layout: { bindingType : BindingType, binding : UInt32, visibility : ShaderStage }
    // Memory layout: Boxed fields (pointers) first, then scalar fields in declaration order
    //   Field 0 (offset 0): bindingType : BindingType (ADT, boxed)
    //   Field 1 (offset sizeof(void*)): binding : UInt32 (scalar, 4 bytes)
    //   Field 2 (offset sizeof(void*)+4): visibility : ShaderStage (enum/uint8, 1 byte)
    //
    // The bindingType ADT has these constructors:
    //   tag 0: buffer (readOnly : Bool)  - Storage buffer
    //   tag 1: uniformBuffer             - Uniform buffer
    //   tag 2: sampler                   - Texture sampler
    //   tag 3: texture                   - Texture
    for (size_t i = 0; i < num_entries; i++) {
        lean_object* entry = lean_array_get_core(entries_array, i);

        // Field 0: bindingType (BindingType ADT)
        lean_object* binding_type = lean_ctor_get(entry, 0);

        // Field 1: binding (UInt32) - in scalar data area
        uint32_t binding = lean_ctor_get_uint32(entry, sizeof(void*));

        // Field 2: visibility (ShaderStage enum/uint8) - after binding
        uint8_t visibility = lean_ctor_get_uint8(entry, sizeof(void*) + sizeof(uint32_t));

        // Decode BindingType ADT
        // tag 0 = buffer (readOnly : Bool), tag 1 = uniformBuffer, tag 2 = sampler, tag 3 = texture
        uint8_t binding_type_tag = lean_obj_tag(binding_type);

        layoutEntries[i].binding = binding;
        layoutEntries[i].visibility = (visibility == 0) ? wgpu::ShaderStage::Vertex :
                                       (visibility == 1) ? wgpu::ShaderStage::Fragment :
                                                           wgpu::ShaderStage::Compute;

        if (binding_type_tag == 0) {  // buffer
            // Extract readOnly : Bool from the ADT constructor
            bool read_only = lean_ctor_get_uint8(binding_type, 0) != 0;
            layoutEntries[i].buffer.type = read_only ? wgpu::BufferBindingType::ReadOnlyStorage
                                                      : wgpu::BufferBindingType::Storage;
            // CRITICAL: Set minBindingSize = 0 to allow runtime-sized arrays in shaders
            layoutEntries[i].buffer.minBindingSize = 0;
            layoutEntries[i].buffer.hasDynamicOffset = false;

            if (g_verbose) fprintf(stderr, "[C++]   Entry %zu: binding=%u, visibility=Compute, type=Storage (%s)\n",
                    i, binding, read_only ? "read-only" : "read-write");
        } else if (binding_type_tag == 1) {  // uniformBuffer
            layoutEntries[i].buffer.type = wgpu::BufferBindingType::Uniform;
            layoutEntries[i].buffer.minBindingSize = 0;
            layoutEntries[i].buffer.hasDynamicOffset = false;
            if (g_verbose) fprintf(stderr, "[C++]   Entry %zu: binding=%u, visibility=Compute, type=Uniform\n", i, binding);
        }
        // TODO: Handle sampler and texture types if needed

        fflush(stderr);
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = layoutEntries.size();
    bglDesc.entries = layoutEntries.data();

    if (g_verbose) fprintf(stderr, "[C++] Creating BindGroupLayout...\n");
    fflush(stderr);

    wgpu::BindGroupLayout bindGroupLayout = device->CreateBindGroupLayout(&bglDesc);

    if (g_verbose) fprintf(stderr, "[C++] BindGroupLayout created\n");
    fflush(stderr);

    // Wrap in External with registered class
    wgpu::BindGroupLayout* layoutPtr = new wgpu::BindGroupLayout(bindGroupLayout);
    lean_object* external = lean_alloc_external(g_webgpu_bind_group_layout_class, layoutPtr);

    lean_object* layout_struct = make_resource_with_device(external, device_obj);
    return lean_io_result_mk_ok(layout_struct);
}

// C struct matching Lean's BindGroupEntry memory layout
// Lean reorders fields: boxed (pointers) first, then scalars by size (large to small)
typedef struct {
    lean_object* buffer;  // Boxed field (pointer)
    size_t offset;        // Scalar field (USize, 8 bytes)
    size_t size;          // Scalar field (USize, 8 bytes)
    uint32_t binding;     // Scalar field (UInt32, 4 bytes)
} BindGroupEntry_Raw;

lean_obj_res lean_hesper_create_bind_group(b_lean_obj_arg device_obj, b_lean_obj_arg layout_obj,
                                            b_lean_obj_arg entries_array, lean_obj_res /* unit */) {
    // Extract device and layout from External objects
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::BindGroupLayout* layout = EXTRACT_BIND_GROUP_LAYOUT_PTR(layout_obj);

    // Extract array of BindGroupEntry
    size_t num_entries = lean_array_size(entries_array);

    if (g_verbose) fprintf(stderr, "[C++] createBindGroup: %zu entries\n", num_entries);
    fflush(stderr);

    std::vector<wgpu::BindGroupEntry> bgEntries(num_entries);

    for (size_t i = 0; i < num_entries; i++) {
        lean_object* entry = lean_array_get_core(entries_array, i);

        // Cast to our raw struct that matches Lean's memory layout
        BindGroupEntry_Raw* raw = (BindGroupEntry_Raw*)lean_ctor_obj_cptr(entry);

        // Extract buffer from Buffer structure (not direct External anymore)
        wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(raw->buffer);

        // VALIDATION: Check if buffer is valid
        if (!buffer) {
            if (g_verbose) fprintf(stderr, "[C++] ERROR: Entry %zu has NULL buffer pointer!\n", i);
            fflush(stderr);
        }

        // Access fields directly through the struct
        bgEntries[i].binding = raw->binding;
        bgEntries[i].buffer = *buffer;
        bgEntries[i].offset = raw->offset;

        // FIX: size=0 means "use whole buffer" in WebGPU convention
        // Query the buffer's actual size when Lean passes 0
        if (raw->size == 0) {
            bgEntries[i].size = buffer->GetSize();
        } else {
            bgEntries[i].size = raw->size;
        }

        // VALIDATION: Check buffer properties
        uint64_t bufSize = buffer->GetSize();
        uint64_t bufUsage = static_cast<uint64_t>(buffer->GetUsage());

        if (g_verbose) fprintf(stderr, "[C++]   Entry %zu: binding=%u, offset=%llu, size=%llu (raw_size=%zu, actual_buffer_size=%llu, usage=0x%llx)\n",
                i, raw->binding, (unsigned long long)raw->offset, (unsigned long long)bgEntries[i].size, raw->size, bufSize, bufUsage);
        fflush(stderr);

        // Validate buffer size vs requested size
        if (bgEntries[i].offset + bgEntries[i].size > bufSize) {
            if (g_verbose) fprintf(stderr, "[C++] WARNING: Entry %zu: offset(%llu) + size(%llu) = %llu exceeds buffer size %llu\n",
                    i, (unsigned long long)bgEntries[i].offset, (unsigned long long)bgEntries[i].size,
                    (unsigned long long)(bgEntries[i].offset + bgEntries[i].size), bufSize);
            fflush(stderr);
        }
    }

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = *layout;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();

    if (g_verbose) fprintf(stderr, "[C++] Creating BindGroup...\n");
    fflush(stderr);

    wgpu::BindGroup bindGroup = device->CreateBindGroup(&bgDesc);

    if (g_verbose) fprintf(stderr, "[C++] BindGroup created: wrapper=%p, WGPUBindGroup=%p\n", (void*)&bindGroup, (void*)bindGroup.Get());
    fflush(stderr);

    // CRITICAL VALIDATION: Check if BindGroup handle is valid
    if (!bindGroup.Get()) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: CreateBindGroup returned null handle!\n");
        fflush(stderr);
        lean_object* error_msg = lean_mk_string("CreateBindGroup returned null handle");
        return lean_io_result_mk_error(error_msg);
    }

    // Validate all buffer handles are still valid after BindGroup creation
    if (g_verbose) fprintf(stderr, "[C++] Validating buffer handles after BindGroup creation...\n");
    for (size_t i = 0; i < num_entries; i++) {
        lean_object* entry = lean_array_get_core(entries_array, i);
        BindGroupEntry_Raw* raw = (BindGroupEntry_Raw*)lean_ctor_obj_cptr(entry);
        wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(raw->buffer);

        if (!buffer) {
            if (g_verbose) fprintf(stderr, "[C++] ERROR: Buffer %zu wrapper is null!\n", i);
            fflush(stderr);
        } else if (!buffer->Get()) {
            if (g_verbose) fprintf(stderr, "[C++] ERROR: Buffer %zu WGPUBuffer handle is null!\n", i);
            fflush(stderr);
        } else {
            if (g_verbose) fprintf(stderr, "[C++]   Buffer %zu OK: wrapper=%p, WGPUBuffer=%p\n",
                    i, (void*)buffer, (void*)buffer->Get());
            fflush(stderr);
        }
    }

    // Wrap in External with registered class
    wgpu::BindGroup* groupPtr = new wgpu::BindGroup(bindGroup);
    if (g_verbose) fprintf(stderr, "[C++]   Heap wrapper=%p, WGPUBindGroup=%p\n", (void*)groupPtr, (void*)groupPtr->Get());
    fflush(stderr);

    lean_object* external = lean_alloc_external(g_webgpu_bind_group_class, groupPtr);

    lean_object* bindgroup_struct = make_resource_with_device(external, device_obj);
    return lean_io_result_mk_ok(bindgroup_struct);
}

lean_obj_res lean_hesper_create_compute_pipeline(b_lean_obj_arg device_obj, b_lean_obj_arg desc, lean_obj_res /* unit */) {
    // Extract device from External object
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    // Extract fields from ComputePipelineDescriptor
    // struct ComputePipelineDescriptor where
    //   shaderModule : ShaderModule       -- Field 0 (External)
    //   entryPoint : String               -- Field 1 (boxed String)
    //   bindGroupLayout : BindGroupLayout -- Field 2 (External)

    lean_object* shaderModule_obj = lean_ctor_get(desc, 0);
    lean_object* entryPoint_obj = lean_ctor_get(desc, 1);
    lean_object* bindGroupLayout_obj = lean_ctor_get(desc, 2);

    wgpu::ShaderModule* shaderModule = EXTRACT_SHADER_MODULE_PTR(shaderModule_obj);
    const char* entryPoint = lean_string_cstr(entryPoint_obj);
    wgpu::BindGroupLayout* bindGroupLayout = EXTRACT_BIND_GROUP_LAYOUT_PTR(bindGroupLayout_obj);

    if (g_verbose) fprintf(stderr, "[C++] createComputePipeline: entryPoint='%s'\n", entryPoint);
    fflush(stderr);

    // Create pipeline layout
    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device->CreatePipelineLayout(&layoutDesc);

    // Create compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = *shaderModule;
    pipelineDesc.compute.entryPoint = entryPoint;
    pipelineDesc.layout = pipelineLayout;

    if (g_verbose) fprintf(stderr, "[C++] Calling CreateComputePipeline...\n");
    fflush(stderr);

    // Create pipeline within error scope to catch validation errors
    wgpu::ComputePipeline pipeline;
    bool success = runWithErrorScope(device->Get(), g_dawn_instance, "CreateComputePipeline", [&]() {
        pipeline = device->CreateComputePipeline(&pipelineDesc);
    });

    if (!pipeline || !success) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ CRITICAL ERROR: Pipeline Creation Failed                 ║\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
        if (g_verbose) fprintf(stderr, "[C++] CreateComputePipeline returned null or had validation errors\n");
        fprintf(stderr, "════════════════════════════════════════════════════════════\n\n");
        fflush(stderr);
        lean_object* error_msg = lean_mk_string("Compute pipeline creation failed - check stderr for errors");
        return lean_io_result_mk_error(error_msg);
    }

    if (g_verbose) fprintf(stderr, "[C++] ComputePipeline created successfully\n");
    fflush(stderr);

    // Wrap in External with registered class
    wgpu::ComputePipeline* pipelinePtr = new wgpu::ComputePipeline(pipeline);
    lean_object* external = lean_alloc_external(g_webgpu_compute_pipeline_class, pipelinePtr);

    lean_object* pipeline_struct = make_resource_with_device(external, device_obj);
    return lean_io_result_mk_ok(pipeline_struct);
}

// Callback for OnSubmittedWorkDone - used by dispatchCompute
static void dispatchWorkDoneCallback(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                                     void* userdata1, void* /*userdata2*/) {
    FutureData* futureData = static_cast<FutureData*>(userdata1);

    if (status != WGPUQueueWorkDoneStatus_Success) {
        if (g_verbose) fprintf(stderr, "[C++] GPU work done with error: status=%d, message=%.*s\n",
                (int)status, (int)message.length, message.data);
        fflush(stderr);
    }

    if (g_verbose) fprintf(stderr, "[C++] GPU work completed, signaling future\n");
    fflush(stderr);

    futureData->promise->set_value();
}

lean_obj_res lean_hesper_dispatch_compute(b_lean_obj_arg device_obj, b_lean_obj_arg pipeline_obj, b_lean_obj_arg bind_group_obj,
                                           uint32_t workgroupsX, uint32_t workgroupsY, uint32_t workgroupsZ, lean_obj_res /* unit */) {
    // Extract device, pipeline and bind group from External objects
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::ComputePipeline* pipeline = EXTRACT_COMPUTE_PIPELINE_PTR(pipeline_obj);
    wgpu::BindGroup* bindGroup = EXTRACT_BIND_GROUP_PTR(bind_group_obj);

    // Extract instance from device structure (field 1)
    lean_object* instance_obj = lean_ctor_get(device_obj, 1);

    if (g_verbose) fprintf(stderr, "[C++] dispatchCompute: workgroups=(%u,%u,%u)\n", workgroupsX, workgroupsY, workgroupsZ);
    if (g_verbose) fprintf(stderr, "[C++]   device=%p, WGPUDevice=%p\n", (void*)device, (void*)device->Get());
    if (g_verbose) fprintf(stderr, "[C++]   pipeline=%p, WGPUPipeline=%p\n", (void*)pipeline, (void*)pipeline->Get());
    if (g_verbose) fprintf(stderr, "[C++]   bindGroup=%p, WGPUBindGroup=%p\n", (void*)bindGroup, (void*)bindGroup->Get());

    // Validate pointers
    if (!device || !device->Get()) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: Device is null or invalid!\n");
        fflush(stderr);
        return lean_io_result_mk_error(lean_mk_string("Device is invalid"));
    }
    if (!pipeline || !pipeline->Get()) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: Pipeline is null or invalid!\n");
        fflush(stderr);
        return lean_io_result_mk_error(lean_mk_string("Pipeline is invalid"));
    }
    if (!bindGroup || !bindGroup->Get()) {
        if (g_verbose) fprintf(stderr, "[C++] ERROR: BindGroup is null or invalid!\n");
        fflush(stderr);
        return lean_io_result_mk_error(lean_mk_string("BindGroup is invalid"));
    }

    fflush(stderr);

    // Wrap dispatch in error scope to catch any VALIDATION errors (blocking check)
    bool success = runWithErrorScope(device->Get(), g_dawn_instance, "DispatchCompute", [&]() {
        // Use C API like gpu.hpp to avoid C++ wrapper destructor issues
        WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(device->Get(), nullptr);
        WGPUComputePassEncoder computePassEncoder = wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
        wgpuComputePassEncoderSetPipeline(computePassEncoder, pipeline->Get());
        wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup->Get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, workgroupsX, workgroupsY, workgroupsZ);
        wgpuComputePassEncoderEnd(computePassEncoder);
        wgpuComputePassEncoderRelease(computePassEncoder);  // Explicitly release like gpu.hpp
        WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
        wgpuCommandEncoderRelease(commandEncoder);  // Explicitly release like gpu.hpp

        wgpuQueueSubmit(device->GetQueue().Get(), 1, &commandBuffer);
        wgpuCommandBufferRelease(commandBuffer);  // Release after submit like gpu.hpp
    });

    if (!success) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ CRITICAL ERROR: Dispatch Compute Failed                  ║\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
        if (g_verbose) fprintf(stderr, "[C++] DispatchCompute had validation or runtime errors\n");
        fprintf(stderr, "════════════════════════════════════════════════════════════\n\n");
        fflush(stderr);
        lean_object* error_msg = lean_mk_string("Dispatch compute failed - check stderr for errors");
        return lean_io_result_mk_error(error_msg);
    }

    // Create promise/future for ASYNC GPU work completion
    auto promise = std::make_shared<std::promise<void>>();
    auto future = std::make_shared<std::future<void>>(promise->get_future());

    // Create FutureData BEFORE registering callback - this keeps promise alive
    FutureData* futureData = new FutureData{promise, future, g_dawn_instance};

    if (g_verbose) fprintf(stderr, "[C++] Created FutureData at %p\n", futureData);
    fflush(stderr);

    // Register OnSubmittedWorkDone callback for async GPU completion
    // The callback will access promise through FutureData
    WGPUQueueWorkDoneCallbackInfo workDoneInfo = {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = dispatchWorkDoneCallback,
        .userdata1 = futureData,  // Pass FutureData, not just promise
        .userdata2 = nullptr
    };

    wgpuQueueOnSubmittedWorkDone(device->GetQueue().Get(), workDoneInfo);

    if (g_verbose) fprintf(stderr, "[C++] dispatchCompute: submitted async (returns Future for GPU work completion)\n");
    fflush(stderr);

    // Wrap FutureData in External
    lean_object* future_external = lean_alloc_external(g_webgpu_future_class, futureData);

    // Create Future structure: { ptr : FuturePtr, parentInstance : Instance }
    lean_object* future_struct = lean_alloc_ctor(0, 2, 0);  // constructor with 2 fields, 0 scalars
    lean_inc(instance_obj);  // Increment refcount of instance so it stays alive
    lean_ctor_set(future_struct, 0, future_external);  // field 0: ptr
    lean_ctor_set(future_struct, 1, instance_obj);     // field 1: parentInstance

    return lean_io_result_mk_ok(future_struct);
}

// ============================================================================
// Float Conversion Utilities
// ============================================================================

/**
 * @brief Convert f32 bytes (little-endian) to f64
 *
 * BUG FIX: Lean's Float.ofBits expects f64 bits, but GPU returns f32 bits.
 * We need to properly interpret the 32-bit pattern as f32, then convert to f64.
 *
 * @param bytes ByteArray containing f32 data
 * @param offset Byte offset to read from
 * @return f64 value
 */
lean_obj_res lean_hesper_bytes_to_float64(b_lean_obj_arg bytes, uint32_t offset) {
    size_t size = lean_sarray_size(bytes);
    const uint8_t* data = lean_sarray_cptr(bytes);

    if (offset + 4 > size) {
        if (g_verbose) fprintf(stderr, "[C++] bytes_to_float64: offset %u + 4 exceeds size %zu\n", offset, size);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("offset out of bounds")));
    }

    // Read 4 bytes as little-endian f32
    uint32_t bits32 = data[offset] |
                      (data[offset + 1] << 8) |
                      (data[offset + 2] << 16) |
                      (data[offset + 3] << 24);

    // Interpret as f32
    float f32_value;
    memcpy(&f32_value, &bits32, sizeof(float));

    // Convert f32 to f64
    double f64_value = static_cast<double>(f32_value);

    return lean_io_result_mk_ok(lean_box_float(f64_value));
}

// Convert Float64 (Lean Float) to 4 bytes (little-endian f32)
// Properly converts f64→f32 then serializes to bytes
lean_obj_res lean_hesper_float64_to_bytes(double f64_value) {
    if (g_verbose) fprintf(stderr, "[C++] float64_to_bytes called: f64=%.6f\n", f64_value);
    fflush(stderr);

    // Convert f64 to f32 (proper narrowing conversion)
    float f32_value = static_cast<float>(f64_value);

    // Get f32 bit representation
    uint32_t bits32;
    memcpy(&bits32, &f32_value, sizeof(float));

    // Create ByteArray with 4 bytes (little-endian)
    uint8_t bytes[4] = {
        static_cast<uint8_t>(bits32 & 0xFF),
        static_cast<uint8_t>((bits32 >> 8) & 0xFF),
        static_cast<uint8_t>((bits32 >> 16) & 0xFF),
        static_cast<uint8_t>((bits32 >> 24) & 0xFF)
    };

    lean_object* byte_array = lean_alloc_sarray(1, 4, 4);  // tag=1, size=4, capacity=4
    uint8_t* dest = reinterpret_cast<uint8_t*>(lean_sarray_cptr(byte_array));
    memcpy(dest, bytes, 4);

    if (g_verbose) fprintf(stderr, "[C++] float64_to_bytes returning ByteArray size=%zu, bytes=%02x %02x %02x %02x\n",
            lean_sarray_size(byte_array), bytes[0], bytes[1], bytes[2], bytes[3]);
    fflush(stderr);

    return lean_io_result_mk_ok(byte_array);
}

// ============================================================================
// Opaque Array Types: Float32Array, Float16Array, BFloat16Array
// ============================================================================

// These implement zero-copy, in-place mutable arrays for efficient GPU interop
// and tensor operations. Each array type is fully opaque to Lean and uses
// std::vector for native C++ performance.

// ----------------------------------------------------------------------------
// Float32Array Implementation
// ----------------------------------------------------------------------------

struct Float32ArrayWrapper {
    std::vector<float> data;

    explicit Float32ArrayWrapper(size_t size) : data(size, 0.0f) {}
};

static void Float32Array_finalizer(void* ptr) {
    delete static_cast<Float32ArrayWrapper*>(ptr);
}

static lean_external_class* g_Float32Array_class = nullptr;

static lean_external_class* get_Float32Array_class() {
    if (g_Float32Array_class == nullptr) {
        g_Float32Array_class = lean_register_external_class(
            Float32Array_finalizer,
            noop_foreach
        );
    }
    return g_Float32Array_class;
}

// Create new Float32Array
lean_obj_res lean_f32_array_create(size_t size, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = new Float32ArrayWrapper(size);
    lean_object* obj = lean_alloc_external(get_Float32Array_class(), wrapper);
    return lean_io_result_mk_ok(obj);
}

// Set element (in-place mutation)
lean_obj_res lean_f32_array_set(lean_object* arr, size_t index, double value, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));

    if (index >= wrapper->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float32Array index out of bounds"))
        );
    }

    wrapper->data[index] = static_cast<float>(value);
    return lean_io_result_mk_ok(lean_box(0));
}

// Get element
lean_obj_res lean_f32_array_get(lean_object* arr, size_t index, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));

    if (index >= wrapper->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float32Array index out of bounds"))
        );
    }

    double value = static_cast<double>(wrapper->data[index]);
    return lean_io_result_mk_ok(lean_box_float(value));
}

// Get size
size_t lean_f32_array_size(lean_object* arr) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));
    return wrapper->data.size();
}

// Get byte size
lean_obj_res lean_f32_array_byte_size(lean_object* arr, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));
    size_t byte_size = wrapper->data.size() * sizeof(float);
    return lean_io_result_mk_ok(lean_box_usize(byte_size));
}

// Get raw pointer (UNSAFE - for GPU interop)
lean_obj_res lean_f32_array_ptr(lean_object* arr, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));
    size_t ptr = reinterpret_cast<size_t>(wrapper->data.data());
    return lean_io_result_mk_ok(lean_box_usize(ptr));
}

// Convert from FloatArray (Array Float)
lean_obj_res lean_f32_array_from_float_array(lean_object* float_arr, lean_object* /* io_world */) {
    size_t size = lean_array_size(float_arr);
    Float32ArrayWrapper* wrapper = new Float32ArrayWrapper(size);

    for (size_t i = 0; i < size; i++) {
        lean_object* elem = lean_array_get_core(float_arr, i);
        double val = lean_unbox_float(elem);
        wrapper->data[i] = static_cast<float>(val);
    }

    lean_object* obj = lean_alloc_external(get_Float32Array_class(), wrapper);
    return lean_io_result_mk_ok(obj);
}

// Convert to FloatArray (Array Float)
lean_obj_res lean_f32_array_to_float_array(lean_object* arr, lean_object* /* io_world */) {
    Float32ArrayWrapper* wrapper = static_cast<Float32ArrayWrapper*>(lean_get_external_data(arr));
    size_t size = wrapper->data.size();

    lean_object* result = lean_alloc_array(size, size);
    for (size_t i = 0; i < size; i++) {
        double val = static_cast<double>(wrapper->data[i]);
        lean_array_set_core(result, i, lean_box_float(val));
    }

    return lean_io_result_mk_ok(result);
}

// SIMD Add (calls existing simd_add_f32 if available, or scalar fallback)
lean_obj_res lean_f32_array_simd_add(lean_object* a, lean_object* b, lean_object* io_world) {
    Float32ArrayWrapper* wa = static_cast<Float32ArrayWrapper*>(lean_get_external_data(a));
    Float32ArrayWrapper* wb = static_cast<Float32ArrayWrapper*>(lean_get_external_data(b));

    if (wa->data.size() != wb->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float32Array size mismatch"))
        );
    }

    size_t size = wa->data.size();
    Float32ArrayWrapper* result = new Float32ArrayWrapper(size);

    // Scalar fallback for now (TODO: call SIMD implementation)
    for (size_t i = 0; i < size; i++) {
        result->data[i] = wa->data[i] + wb->data[i];
    }

    lean_object* obj = lean_alloc_external(get_Float32Array_class(), result);
    return lean_io_result_mk_ok(obj);
}

// SIMD Mul
lean_obj_res lean_f32_array_simd_mul(lean_object* a, lean_object* b, lean_object* io_world) {
    Float32ArrayWrapper* wa = static_cast<Float32ArrayWrapper*>(lean_get_external_data(a));
    Float32ArrayWrapper* wb = static_cast<Float32ArrayWrapper*>(lean_get_external_data(b));

    if (wa->data.size() != wb->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float32Array size mismatch"))
        );
    }

    size_t size = wa->data.size();
    Float32ArrayWrapper* result = new Float32ArrayWrapper(size);

    for (size_t i = 0; i < size; i++) {
        result->data[i] = wa->data[i] * wb->data[i];
    }

    lean_object* obj = lean_alloc_external(get_Float32Array_class(), result);
    return lean_io_result_mk_ok(obj);
}

// ----------------------------------------------------------------------------
// Float16Array Implementation (Minimal Storage Box)
// ----------------------------------------------------------------------------

// FP32 -> FP16 conversion (software fallback if F16C not available)
// Based on IEEE 754 half-precision format
static uint16_t float_to_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    // Handle special cases
    if (exponent <= 0) {
        // Underflow to zero
        return static_cast<uint16_t>(sign);
    } else if (exponent >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    // Normal case: round mantissa to 10 bits
    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

// FP16 -> FP32 conversion
static float half_to_float(uint16_t half) {
    uint32_t sign = (half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 0x1F;
    uint32_t mantissa = half & 0x3FF;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            bits = sign;
        } else {
            // Denormalized number
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal number
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

struct Float16ArrayWrapper {
    std::vector<uint16_t> data;  // FP16 stored as raw 16-bit values

    explicit Float16ArrayWrapper(size_t size) : data(size, 0) {}
};

static void Float16Array_finalizer(void* ptr) {
    delete static_cast<Float16ArrayWrapper*>(ptr);
}

static lean_external_class* g_Float16Array_class = nullptr;

static lean_external_class* get_Float16Array_class() {
    if (g_Float16Array_class == nullptr) {
        g_Float16Array_class = lean_register_external_class(
            Float16Array_finalizer,
            noop_foreach
        );
    }
    return g_Float16Array_class;
}

// Hardware support check (returns false for now, can be implemented later with CPUID)
lean_obj_res lean_f16_hardware_check(lean_object* /* io_world */) {
    // TODO: Check for F16C (x86) or FP16 (ARM) support via CPUID/HWCAP
    // For now, always return false (use software conversion)
    return lean_io_result_mk_ok(lean_box(0));
}

// Create new Float16Array
lean_obj_res lean_f16_array_create(size_t size, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = new Float16ArrayWrapper(size);
    lean_object* obj = lean_alloc_external(get_Float16Array_class(), wrapper);
    return lean_io_result_mk_ok(obj);
}

// Set element (converts f64 -> f32 -> f16)
lean_obj_res lean_f16_array_set(lean_object* arr, size_t index, double value, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));

    if (index >= wrapper->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float16Array index out of bounds"))
        );
    }

    wrapper->data[index] = float_to_half(static_cast<float>(value));
    return lean_io_result_mk_ok(lean_box(0));
}

// Get element (converts f16 -> f32 -> f64)
lean_obj_res lean_f16_array_get(lean_object* arr, size_t index, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));

    if (index >= wrapper->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float16Array index out of bounds"))
        );
    }

    float f32 = half_to_float(wrapper->data[index]);
    double f64 = static_cast<double>(f32);
    return lean_io_result_mk_ok(lean_box_float(f64));
}

// Get size
size_t lean_f16_array_size(lean_object* arr) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));
    return wrapper->data.size();
}

// Get byte size
lean_obj_res lean_f16_array_byte_size(lean_object* arr, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));
    size_t byte_size = wrapper->data.size() * sizeof(uint16_t);
    return lean_io_result_mk_ok(lean_box_usize(byte_size));
}

// Get raw pointer (for GPU uploads)
lean_obj_res lean_f16_array_ptr(lean_object* arr, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));
    size_t ptr = reinterpret_cast<size_t>(wrapper->data.data());
    return lean_io_result_mk_ok(lean_box_usize(ptr));
}

// Convert from Array Float
lean_obj_res lean_f16_array_from_float_array(lean_object* float_arr, lean_object* /* io_world */) {
    size_t size = lean_array_size(float_arr);
    Float16ArrayWrapper* wrapper = new Float16ArrayWrapper(size);

    for (size_t i = 0; i < size; i++) {
        lean_object* elem = lean_array_get_core(float_arr, i);
        double val = lean_unbox_float(elem);
        wrapper->data[i] = float_to_half(static_cast<float>(val));
    }

    lean_object* obj = lean_alloc_external(get_Float16Array_class(), wrapper);
    return lean_io_result_mk_ok(obj);
}

// Convert to Array Float
lean_obj_res lean_f16_array_to_float_array(lean_object* arr, lean_object* /* io_world */) {
    Float16ArrayWrapper* wrapper = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));
    size_t size = wrapper->data.size();

    lean_object* result = lean_alloc_array(size, size);
    for (size_t i = 0; i < size; i++) {
        float f32 = half_to_float(wrapper->data[i]);
        double f64 = static_cast<double>(f32);
        lean_array_set_core(result, i, lean_box_float(f64));
    }

    return lean_io_result_mk_ok(result);
}

// Convert from Float32Array
lean_obj_res lean_f16_array_from_f32_array(lean_object* f32_arr, lean_object* /* io_world */) {
    Float32ArrayWrapper* src = static_cast<Float32ArrayWrapper*>(lean_get_external_data(f32_arr));
    Float16ArrayWrapper* dst = new Float16ArrayWrapper(src->data.size());

    for (size_t i = 0; i < src->data.size(); i++) {
        dst->data[i] = float_to_half(src->data[i]);
    }

    lean_object* obj = lean_alloc_external(get_Float16Array_class(), dst);
    return lean_io_result_mk_ok(obj);
}

// Convert to Float32Array
lean_obj_res lean_f16_array_to_f32_array(lean_object* arr, lean_object* /* io_world */) {
    Float16ArrayWrapper* src = static_cast<Float16ArrayWrapper*>(lean_get_external_data(arr));
    Float32ArrayWrapper* dst = new Float32ArrayWrapper(src->data.size());

    for (size_t i = 0; i < src->data.size(); i++) {
        dst->data[i] = half_to_float(src->data[i]);
    }

    lean_object* obj = lean_alloc_external(get_Float32Array_class(), dst);
    return lean_io_result_mk_ok(obj);
}

// SIMD operations (scalar fallback for now)
lean_obj_res lean_f16_array_simd_add(lean_object* a, lean_object* b, lean_object* /* io_world */) {
    Float16ArrayWrapper* wa = static_cast<Float16ArrayWrapper*>(lean_get_external_data(a));
    Float16ArrayWrapper* wb = static_cast<Float16ArrayWrapper*>(lean_get_external_data(b));

    if (wa->data.size() != wb->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float16Array size mismatch"))
        );
    }

    size_t size = wa->data.size();
    Float16ArrayWrapper* result = new Float16ArrayWrapper(size);

    // Scalar fallback (TODO: SIMD with F16C/NEON FP16)
    for (size_t i = 0; i < size; i++) {
        float a_f32 = half_to_float(wa->data[i]);
        float b_f32 = half_to_float(wb->data[i]);
        result->data[i] = float_to_half(a_f32 + b_f32);
    }

    lean_object* obj = lean_alloc_external(get_Float16Array_class(), result);
    return lean_io_result_mk_ok(obj);
}

lean_obj_res lean_f16_array_simd_mul(lean_object* a, lean_object* b, lean_object* /* io_world */) {
    Float16ArrayWrapper* wa = static_cast<Float16ArrayWrapper*>(lean_get_external_data(a));
    Float16ArrayWrapper* wb = static_cast<Float16ArrayWrapper*>(lean_get_external_data(b));

    if (wa->data.size() != wb->data.size()) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("Float16Array size mismatch"))
        );
    }

    size_t size = wa->data.size();
    Float16ArrayWrapper* result = new Float16ArrayWrapper(size);

    for (size_t i = 0; i < size; i++) {
        float a_f32 = half_to_float(wa->data[i]);
        float b_f32 = half_to_float(wb->data[i]);
        result->data[i] = float_to_half(a_f32 * b_f32);
    }

    lean_object* obj = lean_alloc_external(get_Float16Array_class(), result);
    return lean_io_result_mk_ok(obj);
}

// ============================================================================
// Command Buffer Batching - Record multiple dispatches, submit once
// ============================================================================

// Helper macro for CommandEncoder extraction
#define EXTRACT_COMMAND_ENCODER_PTR(encoder_struct) \
    static_cast<wgpu::CommandEncoder*>(lean_get_external_data(lean_ctor_get(encoder_struct, 0)))

lean_obj_res lean_hesper_create_command_encoder(b_lean_obj_arg device_obj, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    if (!device || !device->Get()) {
        return lean_io_result_mk_error(lean_mk_string("Device is invalid"));
    }

    WGPUCommandEncoder raw = wgpuDeviceCreateCommandEncoder(device->Get(), nullptr);
    if (!raw) {
        return lean_io_result_mk_error(lean_mk_string("Failed to create command encoder"));
    }

    wgpu::CommandEncoder* encoderPtr = new wgpu::CommandEncoder(raw);
    lean_object* external = lean_alloc_external(g_webgpu_command_encoder_class, encoderPtr);
    lean_object* encoder_struct = make_resource_with_device(external, device_obj);
    return lean_io_result_mk_ok(encoder_struct);
}

lean_obj_res lean_hesper_record_dispatch(b_lean_obj_arg encoder_obj, b_lean_obj_arg pipeline_obj,
                                          b_lean_obj_arg bind_group_obj,
                                          uint32_t workgroupsX, uint32_t workgroupsY, uint32_t workgroupsZ,
                                          lean_obj_res /* unit */) {
    wgpu::CommandEncoder* encoder = EXTRACT_COMMAND_ENCODER_PTR(encoder_obj);
    wgpu::ComputePipeline* pipeline = EXTRACT_COMPUTE_PIPELINE_PTR(pipeline_obj);
    wgpu::BindGroup* bindGroup = EXTRACT_BIND_GROUP_PTR(bind_group_obj);

    if (!encoder || !encoder->Get()) {
        return lean_io_result_mk_error(lean_mk_string("CommandEncoder is invalid"));
    }
    if (!pipeline || !pipeline->Get()) {
        return lean_io_result_mk_error(lean_mk_string("Pipeline is invalid"));
    }
    if (!bindGroup || !bindGroup->Get()) {
        return lean_io_result_mk_error(lean_mk_string("BindGroup is invalid"));
    }

    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder->Get(), nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline->Get());
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup->Get(), 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_submit_and_wait(b_lean_obj_arg device_obj, b_lean_obj_arg encoder_obj, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::CommandEncoder* encoder = EXTRACT_COMMAND_ENCODER_PTR(encoder_obj);

    if (!device || !device->Get()) {
        return lean_io_result_mk_error(lean_mk_string("Device is invalid"));
    }
    if (!encoder || !encoder->Get()) {
        return lean_io_result_mk_error(lean_mk_string("CommandEncoder is invalid"));
    }

    // Finish encoder and submit
    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder->Get(), nullptr);
    wgpuQueueSubmit(device->GetQueue().Get(), 1, &commandBuffer);
    wgpuCommandBufferRelease(commandBuffer);

    // Create promise/future for async completion
    auto promise = std::make_shared<std::promise<void>>();
    auto future = std::make_shared<std::future<void>>(promise->get_future());

    FutureData* futureData = new FutureData{promise, future, g_dawn_instance};

    WGPUQueueWorkDoneCallbackInfo workDoneInfo = {
        .nextInChain = nullptr,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = dispatchWorkDoneCallback,
        .userdata1 = futureData,
        .userdata2 = nullptr
    };
    wgpuQueueOnSubmittedWorkDone(device->GetQueue().Get(), workDoneInfo);

    // Wait synchronously
    wgpu::Instance instance(g_dawn_instance->Get());
    wait(instance, *future);

    // Clean up FutureData (not returned to Lean, so we own it)
    delete futureData;

    return lean_io_result_mk_ok(lean_box(0));
}

}
