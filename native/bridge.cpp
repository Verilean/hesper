#include <lean/lean.h>
#include <webgpu/webgpu.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <atomic>
#include <memory>

// Note: g_device removed - all functions now take device as parameter
// g_dawn_instance is kept global for deviceWait to use (since deviceWait only gets device, not instance)
static dawn::native::Instance* g_dawn_instance = nullptr;

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
        std::cout << "[C++] Finalizing WebGPU Instance" << std::endl;
        delete instance;
    }
}

static void finalize_webgpu_device(void* ptr) {
    wgpu::Device* device = static_cast<wgpu::Device*>(ptr);
    if (device) {
        std::cout << "[C++] Finalizing WebGPU Device" << std::endl;
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

//=============================================================================
// External Class Registration for WebGPU
//=============================================================================

static void register_webgpu_external_classes() {
    if (g_webgpu_instance_class != nullptr) return;  // Already registered

    g_webgpu_instance_class = lean_register_external_class(finalize_webgpu_instance, nullptr);
    g_webgpu_device_class = lean_register_external_class(finalize_webgpu_device, nullptr);
    g_webgpu_buffer_class = lean_register_external_class(finalize_webgpu_buffer, nullptr);
    g_webgpu_shader_module_class = lean_register_external_class(finalize_webgpu_shader_module, nullptr);
    g_webgpu_compute_pipeline_class = lean_register_external_class(finalize_webgpu_compute_pipeline, nullptr);
    g_webgpu_bind_group_class = lean_register_external_class(finalize_webgpu_bind_group, nullptr);
    g_webgpu_bind_group_layout_class = lean_register_external_class(finalize_webgpu_bind_group_layout, nullptr);
    g_webgpu_command_encoder_class = lean_register_external_class(finalize_webgpu_command_encoder, nullptr);

    std::cout << "[C++] WebGPU External classes registered" << std::endl;
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

    std::cout << "[Hesper] Initialized. Found " << adapters.size() << " adapters:" << std::endl;

    for (const auto& adapter : adapters) {
        WGPUAdapter wgpuAdapter = adapter.Get();

        // Get adapter info using the C++ WebGPU API
        wgpu::AdapterInfo info;
        wgpu::Adapter(wgpuAdapter).GetInfo(&info);

        // Convert StringView to std::string_view for printing
        std::string_view descView(info.description.data, info.description.length);

        std::cout << "  - " << descView
                  << " (Backend: ";

        switch (info.backendType) {
            case wgpu::BackendType::Vulkan:
                std::cout << "Vulkan";
                break;
            case wgpu::BackendType::Metal:
                std::cout << "Metal";
                break;
            case wgpu::BackendType::D3D11:
                std::cout << "D3D11";
                break;
            case wgpu::BackendType::D3D12:
                std::cout << "D3D12";
                break;
            case wgpu::BackendType::OpenGL:
                std::cout << "OpenGL";
                break;
            case wgpu::BackendType::OpenGLES:
                std::cout << "OpenGLES";
                break;
            default:
                std::cout << "Other";
                break;
        }

        std::cout << ")" << std::endl;
    }

    // Wrap in External object and return
    lean_object* external = lean_alloc_external(g_webgpu_instance_class, instance);
    return lean_io_result_mk_ok(external);
}

// Lean FFI: Simple GPU Vector Addition (Hello World Compute)
lean_obj_res lean_hesper_vector_add(b_lean_obj_arg instance_obj, uint32_t size, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    std::cout << "[Hesper] Running GPU Vector Addition (size: " << size << ")" << std::endl;

    // Get first adapter and create device
    auto adapters = instance->EnumerateAdapters();
    if (adapters.empty()) {
        std::cerr << "No GPU adapters found!" << std::endl;
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    wgpu::Adapter adapter(adapters[0].Get());

    // Create device
    wgpu::DeviceDescriptor deviceDesc{};
    wgpu::Device device = adapter.CreateDevice(&deviceDesc);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(0, "Failed to create device")));
    }

    std::cout << "  Device created successfully" << std::endl;

    // Create buffers
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = size * sizeof(float);
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;

    wgpu::Buffer bufferA = device.CreateBuffer(&bufferDesc);
    wgpu::Buffer bufferB = device.CreateBuffer(&bufferDesc);
    wgpu::Buffer bufferC = device.CreateBuffer(&bufferDesc);

    std::cout << "  Buffers created" << std::endl;

    // Create test data
    std::vector<float> dataA(size), dataB(size);
    for (uint32_t i = 0; i < size; i++) {
        dataA[i] = static_cast<float>(i);
        dataB[i] = static_cast<float>(size - i);
    }

    // Upload data
    device.GetQueue().WriteBuffer(bufferA, 0, dataA.data(), size * sizeof(float));
    device.GetQueue().WriteBuffer(bufferB, 0, dataB.data(), size * sizeof(float));

    std::cout << "  Data uploaded to GPU" << std::endl;

    // WGSL shader
    const char* shaderCode = R"(
        @group(0) @binding(0) var<storage, read_write> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if (i < arrayLength(&a)) {
                c[i] = a[i] + b[i];
            }
        }
    )";

    // Create shader module
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);

    std::cout << "  Shader compiled" << std::endl;

    // Create bind group layout
    std::vector<wgpu::BindGroupLayoutEntry> entries(3);
    for (int i = 0; i < 3; i++) {
        entries[i].binding = i;
        entries[i].visibility = wgpu::ShaderStage::Compute;
        entries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&bglDesc);

    // Create compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipelineDesc);

    std::cout << "  Compute pipeline created" << std::endl;

    // Create bind group
    std::vector<wgpu::BindGroupEntry> bgEntries(3);
    bgEntries[0].binding = 0;
    bgEntries[0].buffer = bufferA;
    bgEntries[0].size = size * sizeof(float);
    bgEntries[1].binding = 1;
    bgEntries[1].buffer = bufferB;
    bgEntries[1].size = size * sizeof(float);
    bgEntries[2].binding = 2;
    bgEntries[2].buffer = bufferC;
    bgEntries[2].size = size * sizeof(float);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bgDesc);

    // Dispatch compute
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    uint32_t workgroups = (size + 255) / 256;
    pass.DispatchWorkgroups(workgroups, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    std::cout << "  Compute dispatched (" << workgroups << " workgroups)" << std::endl;

    // Read results
    wgpu::BufferDescriptor readDesc{};
    readDesc.size = size * sizeof(float);
    readDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readBuffer = device.CreateBuffer(&readDesc);

    wgpu::CommandEncoder copyEncoder = device.CreateCommandEncoder();
    copyEncoder.CopyBufferToBuffer(bufferC, 0, readBuffer, 0, size * sizeof(float));
    wgpu::CommandBuffer copyCommands = copyEncoder.Finish();
    device.GetQueue().Submit(1, &copyCommands);

    // Wait and map buffer using C API (like webgpu-dawn)
    struct MapCallbackData {
        WGPUMapAsyncStatus status;
        bool done;
    };
    MapCallbackData callbackData = {WGPUMapAsyncStatus_Error, false};

    WGPUBufferMapCallbackInfo callbackInfo = {};
    callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    callbackInfo.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto* data = static_cast<MapCallbackData*>(userdata1);
        data->status = status;
        data->done = true;
    };
    callbackInfo.userdata1 = &callbackData;

    wgpuBufferMapAsync(readBuffer.Get(), WGPUMapMode_Read, 0, size * sizeof(float), callbackInfo);

    while (!callbackData.done) {
        wgpuDeviceTick(device.Get());
    }

    if (callbackData.status != WGPUMapAsyncStatus_Success) {
        std::cerr << "Failed to map buffer!" << std::endl;
        return make_webgpu_io_error(WebGPUError::Buffer(BufferError::MappingFailed("Failed to map buffer")));
    }

    const float* result = static_cast<const float*>(wgpuBufferGetConstMappedRange(readBuffer.Get(), 0, size * sizeof(float)));

    // Verify and print results
    std::cout << "  Results:" << std::endl;
    std::cout << "    First 5: [";
    for (uint32_t i = 0; i < std::min(5u, size); i++) {
        std::cout << result[i];
        if (i < 4 && i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "    Last 5: [";
    for (uint32_t i = std::max(0u, size - 5); i < size; i++) {
        std::cout << result[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify correctness
    bool correct = true;
    for (uint32_t i = 0; i < size; i++) {
        float expected = dataA[i] + dataB[i];
        if (std::abs(result[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "  ✓ Results correct!" << std::endl;
    } else {
        std::cout << "  ✗ Results incorrect!" << std::endl;
    }

    wgpuBufferUnmap(readBuffer.Get());

    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// WebGPU Wrapper FFI Functions
// ============================================================================

// Device Management
// Create device with advanced features (subgroup matrix, F16, etc.)
lean_obj_res lean_hesper_get_device_with_features(b_lean_obj_arg instance_obj, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();
    if (adapters.empty()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    wgpu::Adapter adapter(adapters[0].Get());

    // Enable experimental features for subgroup matrix operations
    static WGPUDawnTogglesDescriptor toggles = {};
    toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    const char* enableList[] = {"allow_unsafe_apis"};
    toggles.enabledToggles = enableList;
    toggles.enabledToggleCount = 1;

    // Request advanced features
    static std::array<WGPUFeatureName, 3> features = {
        WGPUFeatureName_ShaderF16,
        WGPUFeatureName_Subgroups,
        WGPUFeatureName_ChromiumExperimentalSubgroupMatrix
    };

    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.nextInChain = reinterpret_cast<const wgpu::ChainedStruct*>(&toggles.chain);
    deviceDesc.requiredFeatureCount = features.size();
    deviceDesc.requiredFeatures = reinterpret_cast<const wgpu::FeatureName*>(features.data());

    wgpu::Device device = adapter.CreateDevice(&deviceDesc);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(0, "Failed to create device with subgroup features")));
    }

    std::cout << "[Hesper] Device created with subgroup matrix support" << std::endl;

    // Properly allocate device on heap and wrap in External object
    wgpu::Device* devicePtr = new wgpu::Device(device);
    lean_object* external = lean_alloc_external(g_webgpu_device_class, devicePtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_get_device(b_lean_obj_arg instance_obj, lean_obj_res /* unit */) {
    dawn::native::Instance* instance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    auto adapters = instance->EnumerateAdapters();
    if (adapters.empty()) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::NoAdaptersFound()));
    }

    // Wrap adapter (same as vectorAdd does)
    wgpu::Adapter adapter(adapters[0].Get());

    // Create device
    wgpu::DeviceDescriptor deviceDesc{};
    wgpu::Device device = adapter.CreateDevice(&deviceDesc);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(0, "Failed to create device")));
    }

    std::cout << "[Hesper] Device created successfully" << std::endl;

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

    std::cout << "[Hesper] Creating device from GPU adapter[" << gpuIdx << "]" << std::endl;

    // Wrap adapter
    wgpu::Adapter adapter(adapters[gpuIdx].Get());

    // Print adapter info
    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    std::string_view descView(info.description.data, info.description.length);
    std::cout << "[Hesper] Using GPU: " << descView << std::endl;

    // Create device
    wgpu::DeviceDescriptor deviceDesc{};
    wgpu::Device device = adapter.CreateDevice(&deviceDesc);

    if (!device) {
        return make_webgpu_io_error(WebGPUError::Device(DeviceError::DeviceCreationFailed(gpuIdx, "Failed to create device")));
    }

    std::cout << "[Hesper] Device created successfully from adapter " << gpuIdx << std::endl;

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

lean_obj_res lean_hesper_device_wait(b_lean_obj_arg device_obj, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    fprintf(stderr, "[C++] deviceWait: no-op (waiting happens in mapBufferRead via WaitAny)\n");
    fflush(stderr);

    // deviceWait is now a no-op. The actual waiting for GPU operations
    // happens in mapBufferRead using OnSubmittedWorkDone + WaitAny pattern.
    // Calling Tick() or ProcessEvents here causes crashes because there may be
    // stale callbacks from other operations that have gone out of scope.

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

    std::cout << "[C++] createBuffer: size=" << raw_desc->size
              << ", mapped=" << (raw_desc->mappedAtCreation ? "true" : "false") << std::endl;

    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = raw_desc->size;
    // TODO: Parse usage list from raw_desc->usage - for now use default
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    // HARDCODE mappedAtCreation to false to test
    bufferDesc.mappedAtCreation = false;  // ALWAYS FALSE for now

    fprintf(stderr, "[C++] Creating buffer with usage=0x%x\n", (unsigned)bufferDesc.usage);
    fflush(stderr);

    std::cout << "  bufferDesc.mappedAtCreation = " << bufferDesc.mappedAtCreation << " (HARDCODED TO FALSE)" << std::endl;

    wgpu::Buffer buffer = device->CreateBuffer(&bufferDesc);

    std::cout << "  Buffer created, wrapping in Lean External..." << std::endl;

    // Wrap in External with registered class
    wgpu::Buffer* bufferPtr = new wgpu::Buffer(buffer);

    fprintf(stderr, "  [BUFFER CREATE] wrapper=%p, WGPUBuffer=%p\n", (void*)bufferPtr, (void*)buffer.Get());
    fflush(stderr);

    lean_object* external = lean_alloc_external(g_webgpu_buffer_class, bufferPtr);

    std::cout << "  Created Lean external object at " << (void*)external << std::endl;

    // Wrap in structure with device reference to keep device (and instance) alive
    lean_object* buffer_struct = make_resource_with_device(external, device_obj);

    lean_object* result = lean_io_result_mk_ok(buffer_struct);

    std::cout << "  Returning IO result" << std::endl;
    std::cout.flush();

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

    fprintf(stderr, "[C++] writeBuffer START: offset=%zu, size=%zu\n", offset, data_size);
    fprintf(stderr, "[C++]   [BUFFER WRITE] wrapper=%p, WGPUBuffer=%p\n", (void*)buffer, (void*)buffer->Get());
    fflush(stderr);

    // Print first 16 bytes of data being written (as hex and float)
    if (data_size >= 16) {
        fprintf(stderr, "[C++]   Data being written (hex): ");
        for (size_t i = 0; i < 16; i++) {
            fprintf(stderr, "%02x ", data_ptr[i]);
        }
        fprintf(stderr, "\n");

        // Interpret as floats
        const float* float_ptr = reinterpret_cast<const float*>(data_ptr);
        fprintf(stderr, "[C++]   Data as floats: %.2f, %.2f, %.2f, %.2f\n",
                float_ptr[0], float_ptr[1], float_ptr[2], float_ptr[3]);
        fflush(stderr);
    }

    fprintf(stderr, "[C++]   Calling WriteBuffer...\n");
    fflush(stderr);

    // Write data to GPU buffer
    try {
        device->GetQueue().WriteBuffer(*buffer, offset, data_ptr, data_size);
        fprintf(stderr, "[C++]   WriteBuffer returned\n");
        fflush(stderr);
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++] ERROR in WriteBuffer: %s\n", e.what());
        fflush(stderr);
        return make_webgpu_io_error(WebGPUError::Buffer(BufferError::AllocationFailed(0, e.what())));
    }

    // Process pending GPU operations
    std::cout << "[C++] writeBuffer: ticking device..." << std::endl;
    for (int i = 0; i < 10; i++) {
        device->Tick();
    }
    std::cout << "[C++] writeBuffer: complete" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_map_buffer_read(b_lean_obj_arg device_obj, b_lean_obj_arg buffer_obj,
                                          size_t offset, size_t size, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(buffer_obj);

    fprintf(stderr, "[C++] mapBufferRead: offset=%zu, size=%zu\n", offset, size);
    fprintf(stderr, "[C++]   [BUFFER READ] wrapper=%p, WGPUBuffer=%p\n", (void*)buffer, (void*)buffer->Get());
    fflush(stderr);

    // CRITICAL: Wait for all previous GPU work (compute, writes) to complete BEFORE copying
    fprintf(stderr, "[C++] Waiting for previous GPU work to complete...\n");
    fflush(stderr);

    // Use C API for OnSubmittedWorkDone with new struct-based API
    struct WorkDoneData {
        bool done;
    };

    static WorkDoneData workData = {false};  // Static to ensure lifetime during callback

    if (!g_dawn_instance) {
        fprintf(stderr, "[C++] ERROR: g_dawn_instance is null!\n");
        fflush(stderr);
        return lean_io_result_mk_ok(lean_mk_empty_byte_array(lean_box(0)));
    }

    wgpu::Instance instance(g_dawn_instance->Get());

    WGPUQueueWorkDoneCallbackInfo callbackInfo = {};
    callbackInfo.nextInChain = nullptr;
    callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    callbackInfo.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message,
                                void* userdata1, void* userdata2) {
        WorkDoneData* data = static_cast<WorkDoneData*>(userdata1);
        data->done = true;
        fprintf(stderr, "[C++] OnSubmittedWorkDone: status=%d\n", static_cast<int>(status));
        if (message.length > 0) {
            fprintf(stderr, "[C++] Message: %.*s\n",
                    static_cast<int>(message.length), message.data);
        }
        fflush(stderr);
    };
    callbackInfo.userdata1 = &workData;
    callbackInfo.userdata2 = nullptr;

    wgpuQueueOnSubmittedWorkDone(device->GetQueue().Get(), callbackInfo);

    // Process events until work is done
    for (int i = 0; i < 5000 && !workData.done; i++) {
        instance.ProcessEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!workData.done) {
        fprintf(stderr, "[C++] WARNING: GPU work did not complete in time!\n");
        fflush(stderr);
    } else {
        fprintf(stderr, "[C++] GPU work completed, now copying to staging buffer\n");
        fflush(stderr);
    }

    // Following gpu.hpp toCPUAsync pattern:
    // 1. Create readback buffer (staging buffer for CPU read)
    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.size = size;
    stagingDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    stagingDesc.mappedAtCreation = false;  // Don't map at creation

    wgpu::Buffer stagingBuffer = device->CreateBuffer(&stagingDesc);

    fprintf(stderr, "[C++] Created staging buffer (readback buffer)\n");
    fflush(stderr);

    // 2. Create command encoder and copy from GPU buffer to staging buffer
    wgpu::CommandEncoder encoder = device->CreateCommandEncoder();
    encoder.CopyBufferToBuffer(*buffer, offset, stagingBuffer, 0, size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // 3. Submit the command and release command buffer immediately (like gpu.hpp)
    device->GetQueue().Submit(1, &commands);

    fprintf(stderr, "[C++] Submitted copy command to GPU queue\n");
    fflush(stderr);

    // 4. Process GPU queue using ProcessEvents (like gpu.hpp pattern)
    fprintf(stderr, "[C++] Processing GPU queue with ProcessEvents...\n");
    fflush(stderr);

    // ProcessEvents in a loop like gpu.hpp (wait up to 5 seconds)
    for (int i = 0; i < 5000; i++) {
        instance.ProcessEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    fprintf(stderr, "[C++] GPU queue processed, now mapping buffer...\n");
    fflush(stderr);

    // 5. Map the buffer using MapAsync + Tick()
    bool mapped = false;
    wgpu::MapAsyncStatus mapStatus = wgpu::MapAsyncStatus::Success;  // Initialize to success, will be updated by callback

    stagingBuffer.MapAsync(
        wgpu::MapMode::Read,
        0,
        size,
        wgpu::CallbackMode::AllowProcessEvents,
        [&mapped, &mapStatus](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            mapStatus = status;
            if (status == wgpu::MapAsyncStatus::Success) {
                mapped = true;
                fprintf(stderr, "[C++] MapAsync SUCCESS\n");
            } else {
                fprintf(stderr, "[C++] MapAsync FAILED: status=%d\n", static_cast<int>(status));
                if (message.length > 0) {
                    fprintf(stderr, "[C++] Message: %.*s\n",
                            static_cast<int>(message.length), message.data);
                }
            }
            fflush(stderr);
        }
    );

    fprintf(stderr, "[C++] Waiting for MapAsync with ProcessEvents()...\n");
    fflush(stderr);

    // ProcessEvents until mapping completes (up to 5 seconds)
    for (int i = 0; i < 5000 && !mapped; i++) {
        instance.ProcessEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!mapped) {
        fprintf(stderr, "[C++] ERROR: MapAsync timed out, status=%d\n", static_cast<int>(mapStatus));
        fflush(stderr);
        return lean_io_result_mk_ok(lean_mk_empty_byte_array(lean_box(0)));
    }

    fprintf(stderr, "[C++] Buffer mapped successfully\n");
    fflush(stderr);

    // 6. Get the mapped range (like gpu.hpp's wgpuBufferGetConstMappedRange)
    const uint8_t* mappedData = static_cast<const uint8_t*>(stagingBuffer.GetConstMappedRange(0, size));

    if (!mappedData) {
        fprintf(stderr, "[C++] ERROR: GetConstMappedRange returned nullptr\n");
        fflush(stderr);
        stagingBuffer.Unmap();
        return lean_io_result_mk_ok(lean_mk_empty_byte_array(lean_box(0)));
    }

    fprintf(stderr, "[C++] Got mapped range, creating ByteArray...\n");
    fflush(stderr);

    // Print first 16 bytes of mapped data (as hex and float)
    if (size >= 16) {
        fprintf(stderr, "[C++]   Mapped data (hex): ");
        for (size_t i = 0; i < 16; i++) {
            fprintf(stderr, "%02x ", mappedData[i]);
        }
        fprintf(stderr, "\n");

        // Interpret as floats
        const float* float_ptr = reinterpret_cast<const float*>(mappedData);
        fprintf(stderr, "[C++]   Mapped data as floats: %.2f, %.2f, %.2f, %.2f\n",
                float_ptr[0], float_ptr[1], float_ptr[2], float_ptr[3]);
        fflush(stderr);
    }

    // 7. Copy data to Lean ByteArray using push API
    lean_object* byte_array = lean_mk_empty_byte_array(lean_box(0));

    fprintf(stderr, "[C++] Created empty ByteArray, now pushing %zu bytes...\n", size);
    fflush(stderr);

    // Push bytes one by one using Lean API
    for (size_t i = 0; i < size; i++) {
        byte_array = lean_byte_array_push(byte_array, mappedData[i]);
    }

    size_t final_size = lean_sarray_size(byte_array);
    fprintf(stderr, "[C++] Data copied to ByteArray (final_size=%zu, expected=%zu)\n", final_size, size);
    fflush(stderr);

    // 8. Unmap the buffer BEFORE returning (very important - mapped range becomes invalid after unmap)
    stagingBuffer.Unmap();

    fprintf(stderr, "[C++] Buffer unmapped, returning ByteArray\n");
    fflush(stderr);

    // 9. Return the ByteArray (lean_io_result_mk_ok takes ownership, no need to incref)
    return lean_io_result_mk_ok(byte_array);
}

lean_obj_res lean_hesper_unmap_buffer(b_lean_obj_arg buffer, lean_obj_res /* unit */) {
    return lean_io_result_mk_ok(lean_box(0));
}

// Shader Operations

lean_obj_res lean_hesper_create_shader_module(b_lean_obj_arg device_obj, b_lean_obj_arg source, lean_obj_res /* unit */) {
    // Extract device from External object
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    // Extract shader source from Lean string
    const char* shader_code = lean_string_cstr(source);

    fprintf(stderr, "[C++] createShaderModule: compiling WGSL...\n");
    fprintf(stderr, "[C++] ======================================\n");
    fprintf(stderr, "[C++] Shader source (length=%zu):\n", strlen(shader_code));
    fprintf(stderr, "--- BEGIN SHADER ---\n%s\n--- END SHADER ---\n", shader_code);
    fprintf(stderr, "[C++] ======================================\n");
    fflush(stderr);

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shader_code;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;

    // Create shader module - Dawn will automatically output compilation errors to stderr
    wgpu::ShaderModule shaderModule = device->CreateShaderModule(&shaderDesc);

    // Check if shader module creation succeeded
    if (!shaderModule) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ CRITICAL ERROR: Shader Module Creation Failed            ║\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "[C++] CreateShaderModule returned null handle\n");
        fprintf(stderr, "════════════════════════════════════════════════════════════\n\n");
        fflush(stderr);
        lean_object* error_msg = lean_mk_string("Shader module creation failed - check stderr for compilation errors");
        return lean_io_result_mk_error(error_msg);
    }

    fprintf(stderr, "[C++] Shader module created successfully\n");
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

    fprintf(stderr, "[C++] createBindGroupLayout: %zu entries\n", num_entries);
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

            fprintf(stderr, "[C++]   Entry %zu: binding=%u, visibility=Compute, type=Storage (%s)\n",
                    i, binding, read_only ? "read-only" : "read-write");
        } else if (binding_type_tag == 1) {  // uniformBuffer
            layoutEntries[i].buffer.type = wgpu::BufferBindingType::Uniform;
            fprintf(stderr, "[C++]   Entry %zu: binding=%u, visibility=Compute, type=Uniform\n", i, binding);
        }
        // TODO: Handle sampler and texture types if needed

        fflush(stderr);
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = layoutEntries.size();
    bglDesc.entries = layoutEntries.data();

    fprintf(stderr, "[C++] Creating BindGroupLayout...\n");
    fflush(stderr);

    wgpu::BindGroupLayout bindGroupLayout = device->CreateBindGroupLayout(&bglDesc);

    fprintf(stderr, "[C++] BindGroupLayout created\n");
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

    fprintf(stderr, "[C++] createBindGroup: %zu entries\n", num_entries);
    fflush(stderr);

    std::vector<wgpu::BindGroupEntry> bgEntries(num_entries);

    for (size_t i = 0; i < num_entries; i++) {
        lean_object* entry = lean_array_get_core(entries_array, i);

        // Cast to our raw struct that matches Lean's memory layout
        BindGroupEntry_Raw* raw = (BindGroupEntry_Raw*)lean_ctor_obj_cptr(entry);

        // Extract buffer from Buffer structure (not direct External anymore)
        wgpu::Buffer* buffer = EXTRACT_BUFFER_PTR(raw->buffer);

        // Access fields directly through the struct
        bgEntries[i].binding = raw->binding;
        bgEntries[i].buffer = *buffer;
        bgEntries[i].offset = raw->offset;
        bgEntries[i].size = raw->size;

        fprintf(stderr, "[C++]   Entry %zu: binding=%u, offset=%zu, size=%zu\n",
                i, raw->binding, raw->offset, raw->size);
        fflush(stderr);
    }

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = *layout;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();

    fprintf(stderr, "[C++] Creating BindGroup...\n");
    fflush(stderr);

    wgpu::BindGroup bindGroup = device->CreateBindGroup(&bgDesc);

    fprintf(stderr, "[C++] BindGroup created\n");
    fflush(stderr);

    // Wrap in External with registered class
    wgpu::BindGroup* groupPtr = new wgpu::BindGroup(bindGroup);
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

    fprintf(stderr, "[C++] createComputePipeline: entryPoint='%s'\n", entryPoint);
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

    fprintf(stderr, "[C++] Calling CreateComputePipeline...\n");
    fflush(stderr);

    wgpu::ComputePipeline pipeline = device->CreateComputePipeline(&pipelineDesc);

    fprintf(stderr, "[C++] ComputePipeline created successfully\n");
    fflush(stderr);

    // Wrap in External with registered class
    wgpu::ComputePipeline* pipelinePtr = new wgpu::ComputePipeline(pipeline);
    lean_object* external = lean_alloc_external(g_webgpu_compute_pipeline_class, pipelinePtr);

    lean_object* pipeline_struct = make_resource_with_device(external, device_obj);
    return lean_io_result_mk_ok(pipeline_struct);
}

lean_obj_res lean_hesper_dispatch_compute(b_lean_obj_arg device_obj, b_lean_obj_arg pipeline_obj, b_lean_obj_arg bind_group_obj,
                                           uint32_t workgroupsX, uint32_t workgroupsY, uint32_t workgroupsZ, lean_obj_res /* unit */) {
    // Extract device, pipeline and bind group from External objects
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);
    wgpu::ComputePipeline* pipeline = EXTRACT_COMPUTE_PIPELINE_PTR(pipeline_obj);
    wgpu::BindGroup* bindGroup = EXTRACT_BIND_GROUP_PTR(bind_group_obj);

    fprintf(stderr, "[C++] dispatchCompute: workgroups=(%u,%u,%u)\n", workgroupsX, workgroupsY, workgroupsZ);
    fflush(stderr);

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

    fprintf(stderr, "[C++] dispatchCompute: submitted (async, will complete when mapBufferRead is called)\n");
    fflush(stderr);

    // Following gpu.hpp pattern: dispatchCompute just submits work and returns immediately.
    // The actual waiting for GPU completion happens in mapBufferRead via OnSubmittedWorkDone + WaitAny.
    // This is the correct async pattern - don't wait here!

    return lean_io_result_mk_ok(lean_box(0));
}

// Matrix multiplication with subgroup operations
lean_obj_res lean_hesper_matmul_subgroup(b_lean_obj_arg device_obj, b_lean_obj_arg shader_lean, uint32_t m, uint32_t k, uint32_t n, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    std::cout << "[Hesper] Running Subgroup Matrix Multiplication" << std::endl;
    std::cout << "  Dimensions: " << m << "x" << k << " * " << n << "x" << k << std::endl;

    // Extract shader code from Lean string
    const char* shaderCode = lean_string_cstr(shader_lean);

    // Create buffers
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = m * k * sizeof(float);
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer bufferA = device->CreateBuffer(&bufferDesc);

    bufferDesc.size = n * k * sizeof(float);
    wgpu::Buffer bufferB = device->CreateBuffer(&bufferDesc);

    bufferDesc.size = m * n * sizeof(float);
    wgpu::Buffer bufferC = device->CreateBuffer(&bufferDesc);

    std::cout << "  Buffers created" << std::endl;

    // Initialize test data
    std::vector<float> dataA(m * k);
    std::vector<float> dataB(n * k);
    for (uint32_t i = 0; i < m * k; i++) {
        dataA[i] = static_cast<float>(i % 10) / 10.0f;
    }
    for (uint32_t i = 0; i < n * k; i++) {
        dataB[i] = static_cast<float>((i + 5) % 10) / 10.0f;
    }

    // Upload data
    device->GetQueue().WriteBuffer(bufferA, 0, dataA.data(), m * k * sizeof(float));
    device->GetQueue().WriteBuffer(bufferB, 0, dataB.data(), n * k * sizeof(float));
    std::cout << "  Data uploaded" << std::endl;

    // Create shader module
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;
    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = device->CreateShaderModule(&shaderDesc);
    std::cout << "  Shader compiled" << std::endl;

    // Create bind group layout
    std::vector<wgpu::BindGroupLayoutEntry> entries(3);
    for (int i = 0; i < 3; i++) {
        entries[i].binding = i;
        entries[i].visibility = wgpu::ShaderStage::Compute;
        entries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    wgpu::BindGroupLayout bindGroupLayout = device->CreateBindGroupLayout(&bglDesc);

    // Create pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = device->CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = device->CreateComputePipeline(&pipelineDesc);
    std::cout << "  Pipeline created" << std::endl;

    // Create bind group
    std::vector<wgpu::BindGroupEntry> bindEntries(3);
    bindEntries[0].binding = 0;
    bindEntries[0].buffer = bufferA;
    bindEntries[0].size = m * k * sizeof(float);
    bindEntries[1].binding = 1;
    bindEntries[1].buffer = bufferB;
    bindEntries[1].size = n * k * sizeof(float);
    bindEntries[2].binding = 2;
    bindEntries[2].buffer = bufferC;
    bindEntries[2].size = m * n * sizeof(float);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = bindEntries.size();
    bgDesc.entries = bindEntries.data();
    wgpu::BindGroup bindGroup = device->CreateBindGroup(&bgDesc);

    // Dispatch compute
    uint32_t numWorkgroupsX = (m + 32 - 1) / 32;
    uint32_t numWorkgroupsY = (n + 16 - 1) / 16;

    std::cout << "  Dispatching (" << numWorkgroupsX << ", " << numWorkgroupsY << ", 1)" << std::endl;

    wgpu::CommandEncoder encoder = device->CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    device->GetQueue().Submit(1, &commands);

    std::cout << "  ✓ Subgroup matmul completed!" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
}

// 4K Matrix multiplication benchmark with FLOPS calculation
lean_obj_res lean_hesper_matmul_subgroup_4k(b_lean_obj_arg device_obj, b_lean_obj_arg shader_lean, lean_obj_res /* unit */) {
    wgpu::Device* device = EXTRACT_DEVICE_PTR(device_obj);

    constexpr uint32_t m = 4096;
    constexpr uint32_t k = 4096;
    constexpr uint32_t n = 4096;
    constexpr uint32_t tm = 4;
    constexpr uint32_t tn = 8;
    constexpr uint32_t lid0 = 32;
    constexpr uint32_t lid1 = 2;
    constexpr int nIter = 10;  // Run multiple times for accurate timing

    std::cout << "[Hesper] 4K Subgroup Matrix Multiplication Benchmark" << std::endl;
    std::cout << "  Configuration: TM=" << tm << ", TN=" << tn << ", LID0=" << lid0 << ", LID1=" << lid1 << std::endl;

    // Extract shader code from Lean string
    const char* shaderCode = lean_string_cstr(shader_lean);

    // Create buffers
    std::cout << "  Creating GPU buffers..." << std::endl;
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = static_cast<size_t>(m) * k * sizeof(float);
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer bufferA = device->CreateBuffer(&bufferDesc);

    bufferDesc.size = static_cast<size_t>(n) * k * sizeof(float);
    wgpu::Buffer bufferB = device->CreateBuffer(&bufferDesc);

    bufferDesc.size = static_cast<size_t>(m) * n * sizeof(float);
    wgpu::Buffer bufferC = device->CreateBuffer(&bufferDesc);

    // Initialize with random-ish data
    std::cout << "  Initializing data..." << std::endl;
    std::vector<float> dataA(static_cast<size_t>(m) * k);
    std::vector<float> dataB(static_cast<size_t>(n) * k);
    for (size_t i = 0; i < dataA.size(); i++) {
        dataA[i] = static_cast<float>((i * 7 + 13) % 100) / 100.0f;
    }
    for (size_t i = 0; i < dataB.size(); i++) {
        dataB[i] = static_cast<float>((i * 11 + 17) % 100) / 100.0f;
    }

    device->GetQueue().WriteBuffer(bufferA, 0, dataA.data(), dataA.size() * sizeof(float));
    device->GetQueue().WriteBuffer(bufferB, 0, dataB.data(), dataB.size() * sizeof(float));

    // Create shader and pipeline
    std::cout << "  Compiling shader..." << std::endl;
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;
    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = device->CreateShaderModule(&shaderDesc);

    std::vector<wgpu::BindGroupLayoutEntry> entries(3);
    for (int i = 0; i < 3; i++) {
        entries[i].binding = i;
        entries[i].visibility = wgpu::ShaderStage::Compute;
        entries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    wgpu::BindGroupLayout bindGroupLayout = device->CreateBindGroupLayout(&bglDesc);

    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = device->CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = device->CreateComputePipeline(&pipelineDesc);

    // Create bind group
    std::vector<wgpu::BindGroupEntry> bindEntries(3);
    bindEntries[0].binding = 0;
    bindEntries[0].buffer = bufferA;
    bindEntries[0].size = static_cast<size_t>(m) * k * sizeof(float);
    bindEntries[1].binding = 1;
    bindEntries[1].buffer = bufferB;
    bindEntries[1].size = static_cast<size_t>(n) * k * sizeof(float);
    bindEntries[2].binding = 2;
    bindEntries[2].buffer = bufferC;
    bindEntries[2].size = static_cast<size_t>(m) * n * sizeof(float);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = bindEntries.size();
    bgDesc.entries = bindEntries.data();
    wgpu::BindGroup bindGroup = device->CreateBindGroup(&bgDesc);

    // Calculate workgroup count
    uint32_t numWorkgroupsX = (m + 32 - 1) / 32;
    uint32_t numWorkgroupsY = (n + 64 - 1) / 64;

    std::cout << "  Workgroups: (" << numWorkgroupsX << ", " << numWorkgroupsY << ", 1)" << std::endl;
    std::cout << "  Total workgroups: " << (numWorkgroupsX * numWorkgroupsY) << std::endl;
    std::cout << std::endl;

    // Warmup run
    std::cout << "  Warmup run..." << std::endl;
    wgpu::CommandEncoder encoder = device->CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    device->GetQueue().Submit(1, &commands);

    // Benchmark runs
    std::cout << "  Running " << nIter << " benchmark iterations..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < nIter; iter++) {
        wgpu::CommandEncoder enc = device->CreateCommandEncoder();
        wgpu::ComputePassEncoder p = enc.BeginComputePass();
        p.SetPipeline(pipeline);
        p.SetBindGroup(0, bindGroup);
        p.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
        p.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        device->GetQueue().Submit(1, &cmd);
    }

    // Wait for completion (simple approach - just tick a bunch)
    for (int i = 0; i < 1000; i++) {
        device->Tick();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double totalTime = elapsed.count();
    double avgTime = totalTime / nIter;

    // Calculate GFLOPS
    double totalOps = 2.0 * m * n * k;  // 2 ops per MAC
    double gflops = (totalOps * nIter / totalTime) / 1.0e9;
    double avgGflops = totalOps / avgTime / 1.0e9;

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  4K Matrix Multiplication Performance Results" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Matrix size: " << m << "×" << k << " × " << n << "×" << k << std::endl;
    std::cout << "  Total operations: " << std::fixed << std::setprecision(2)
              << (totalOps / 1.0e9) << " GFLOP" << std::endl;
    std::cout << "  Iterations: " << nIter << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3)
              << totalTime << " seconds" << std::endl;
    std::cout << "  Average time per iteration: " << std::fixed << std::setprecision(3)
              << (avgTime * 1000) << " ms" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2)
              << avgGflops << " GFLOPS" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
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
        fprintf(stderr, "[C++] bytes_to_float64: offset %u + 4 exceeds size %zu\n", offset, size);
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

}
