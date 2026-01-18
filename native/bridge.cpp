#include <lean/lean.h>
#include <webgpu/webgpu.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Global instance to keep Dawn alive (shared with glfw_bridge.cpp)
std::unique_ptr<dawn::native::Instance> g_instance;
wgpu::Device g_device;

// External classes for WebGPU resource management
static lean_external_class* g_webgpu_device_class = nullptr;
static lean_external_class* g_webgpu_buffer_class = nullptr;
static lean_external_class* g_webgpu_shader_module_class = nullptr;
static lean_external_class* g_webgpu_compute_pipeline_class = nullptr;
static lean_external_class* g_webgpu_bind_group_class = nullptr;
static lean_external_class* g_webgpu_bind_group_layout_class = nullptr;
static lean_external_class* g_webgpu_command_encoder_class = nullptr;

extern "C" {

//=============================================================================
// WebGPU Finalizers - Called by Lean GC when resources are collected
//=============================================================================

static void finalize_webgpu_device(void* ptr) {
    // Device is a global singleton, so we don't actually release it
    // This finalizer is here for type consistency
    std::cout << "[C++] Finalizing WebGPU Device (no-op for global device)" << std::endl;
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
    if (g_webgpu_device_class != nullptr) return;  // Already registered

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

    // 2. Create Instance
    g_instance = std::make_unique<dawn::native::Instance>();

    // 3. Enumerate Adapters (replaces DiscoverDefaultAdapters + GetAdapters)
    auto adapters = g_instance->EnumerateAdapters();

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

    return lean_io_result_mk_ok(lean_box(0));
}

// Lean FFI: Simple GPU Vector Addition (Hello World Compute)
lean_obj_res lean_hesper_vector_add(uint32_t size, lean_obj_res /* unit */) {
    std::cout << "[Hesper] Running GPU Vector Addition (size: " << size << ")" << std::endl;

    // Get first adapter and create device
    auto adapters = g_instance->EnumerateAdapters();
    if (adapters.empty()) {
        std::cerr << "No GPU adapters found!" << std::endl;
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("No GPU adapters")));
    }

    wgpu::Adapter adapter(adapters[0].Get());

    // Create device
    wgpu::DeviceDescriptor deviceDesc{};
    g_device = adapter.CreateDevice(&deviceDesc);

    if (!g_device) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create device")));
    }

    std::cout << "  Device created successfully" << std::endl;

    // Create buffers
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = size * sizeof(float);
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;

    wgpu::Buffer bufferA = g_device.CreateBuffer(&bufferDesc);
    wgpu::Buffer bufferB = g_device.CreateBuffer(&bufferDesc);
    wgpu::Buffer bufferC = g_device.CreateBuffer(&bufferDesc);

    std::cout << "  Buffers created" << std::endl;

    // Create test data
    std::vector<float> dataA(size), dataB(size);
    for (uint32_t i = 0; i < size; i++) {
        dataA[i] = static_cast<float>(i);
        dataB[i] = static_cast<float>(size - i);
    }

    // Upload data
    g_device.GetQueue().WriteBuffer(bufferA, 0, dataA.data(), size * sizeof(float));
    g_device.GetQueue().WriteBuffer(bufferB, 0, dataB.data(), size * sizeof(float));

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
    wgpu::ShaderModule shaderModule = g_device.CreateShaderModule(&shaderDesc);

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
    wgpu::BindGroupLayout bindGroupLayout = g_device.CreateBindGroupLayout(&bglDesc);

    // Create compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = g_device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = g_device.CreateComputePipeline(&pipelineDesc);

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
    wgpu::BindGroup bindGroup = g_device.CreateBindGroup(&bgDesc);

    // Dispatch compute
    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    uint32_t workgroups = (size + 255) / 256;
    pass.DispatchWorkgroups(workgroups, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    g_device.GetQueue().Submit(1, &commands);

    std::cout << "  Compute dispatched (" << workgroups << " workgroups)" << std::endl;

    // Read results
    wgpu::BufferDescriptor readDesc{};
    readDesc.size = size * sizeof(float);
    readDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readBuffer = g_device.CreateBuffer(&readDesc);

    wgpu::CommandEncoder copyEncoder = g_device.CreateCommandEncoder();
    copyEncoder.CopyBufferToBuffer(bufferC, 0, readBuffer, 0, size * sizeof(float));
    wgpu::CommandBuffer copyCommands = copyEncoder.Finish();
    g_device.GetQueue().Submit(1, &copyCommands);

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
        wgpuDeviceTick(g_device.Get());
    }

    if (callbackData.status != WGPUMapAsyncStatus_Success) {
        std::cerr << "Failed to map buffer!" << std::endl;
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to map buffer")));
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
lean_obj_res lean_hesper_get_device_with_features(lean_obj_res /* unit */) {
    auto adapters = g_instance->EnumerateAdapters();
    if (adapters.empty()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("No GPU adapters found")));
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

    g_device = adapter.CreateDevice(&deviceDesc);

    if (!g_device) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create device with subgroup features")));
    }

    std::cout << "[Hesper] Device created with subgroup matrix support" << std::endl;

    // Wrap in External (using dummy pointer since device is global singleton)
    lean_object* external = lean_alloc_external(g_webgpu_device_class, (void*)1);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_get_device(lean_obj_res /* unit */) {
    auto adapters = g_instance->EnumerateAdapters();
    if (adapters.empty()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("No GPU adapters found")));
    }

    // Wrap adapter (same as vectorAdd does)
    wgpu::Adapter adapter(adapters[0].Get());

    // Create device
    wgpu::DeviceDescriptor deviceDesc{};
    g_device = adapter.CreateDevice(&deviceDesc);

    if (!g_device) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create device")));
    }

    std::cout << "[Hesper] Device created successfully" << std::endl;

    // Wrap in External (using dummy pointer since device is global singleton)
    lean_object* external = lean_alloc_external(g_webgpu_device_class, (void*)1);

    return lean_io_result_mk_ok(external);
}

// Multi-GPU support functions

// Get the number of available GPU adapters
lean_obj_res lean_hesper_get_adapter_count(lean_obj_res /* unit */) {
    auto adapters = g_instance->EnumerateAdapters();
    return lean_io_result_mk_ok(lean_box(adapters.size()));
}

// Get GPU adapter information by index
lean_obj_res lean_hesper_get_adapter_info(uint32_t gpuIdx, lean_obj_res /* unit */) {
    auto adapters = g_instance->EnumerateAdapters();

    if (gpuIdx >= adapters.size()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Invalid GPU index")));
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
lean_obj_res lean_hesper_get_device_by_index(uint32_t gpuIdx, lean_obj_res /* unit */) {
    auto adapters = g_instance->EnumerateAdapters();

    if (adapters.empty()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("No GPU adapters found")));
    }

    if (gpuIdx >= adapters.size()) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Invalid GPU index")));
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
    g_device = adapter.CreateDevice(&deviceDesc);

    if (!g_device) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create device")));
    }

    std::cout << "[Hesper] Device created successfully from adapter " << gpuIdx << std::endl;

    // Wrap in External
    lean_object* external = lean_alloc_external(g_webgpu_device_class, (void*)1);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_release_device(b_lean_obj_arg /* device */, lean_obj_res /* unit */) {
    // In this simple implementation, we keep the device alive
    // Real impl would use handle map and release specific devices
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_device_tick(b_lean_obj_arg /* device */, lean_obj_res /* unit */) {
    g_device.Tick();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_device_wait(b_lean_obj_arg /* device */, lean_obj_res /* unit */) {
    // Simple wait - just tick until idle
    // Real impl would use proper synchronization
    for (int i = 0; i < 100; i++) {
        g_device.Tick();
    }
    return lean_io_result_mk_ok(lean_box(0));
}

// Buffer Operations

lean_obj_res lean_hesper_create_buffer(b_lean_obj_arg /* device */, b_lean_obj_arg desc, lean_obj_res /* unit */) {
    // Extract buffer descriptor fields from Lean object
    // desc is a Lean structure with: size, usage, mappedAtCreation
    // For simplicity, we'll create a basic storage buffer

    size_t size = 1024 * sizeof(float);  // Simplified: use fixed size for now

    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = size;
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    bufferDesc.mappedAtCreation = false;

    wgpu::Buffer buffer = g_device.CreateBuffer(&bufferDesc);

    // Wrap in External with registered class
    wgpu::Buffer* bufferPtr = new wgpu::Buffer(buffer);
    lean_object* external = lean_alloc_external(g_webgpu_buffer_class, bufferPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_write_buffer(b_lean_obj_arg /* device */, b_lean_obj_arg buffer_obj, size_t offset,
                                       b_lean_obj_arg data, lean_obj_res /* unit */) {
    // Extract buffer from External
    wgpu::Buffer* buffer = (wgpu::Buffer*)lean_get_external_data(buffer_obj);

    // data is a ByteArray - extract bytes
    // For now, write dummy data
    std::vector<float> dummy_data(256, 1.0f);
    g_device.GetQueue().WriteBuffer(*buffer, offset, dummy_data.data(), dummy_data.size() * sizeof(float));

    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_hesper_map_buffer_read(b_lean_obj_arg /* device */, b_lean_obj_arg buffer,
                                          size_t offset, size_t size, lean_obj_res /* unit */) {
    // Simplified: return empty byte array
    lean_obj_res byte_array = lean_mk_empty_byte_array(lean_box(0));
    return lean_io_result_mk_ok(byte_array);
}

lean_obj_res lean_hesper_unmap_buffer(b_lean_obj_arg buffer, lean_obj_res /* unit */) {
    return lean_io_result_mk_ok(lean_box(0));
}

// Shader Operations

lean_obj_res lean_hesper_create_shader_module(b_lean_obj_arg /* device */, b_lean_obj_arg source, lean_obj_res /* unit */) {
    // Extract shader source from Lean string
    const char* shader_code = lean_string_cstr(source);

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shader_code;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = g_device.CreateShaderModule(&shaderDesc);

    // Wrap in External with registered class
    wgpu::ShaderModule* modulePtr = new wgpu::ShaderModule(shaderModule);
    lean_object* external = lean_alloc_external(g_webgpu_shader_module_class, modulePtr);

    return lean_io_result_mk_ok(external);
}

// Pipeline Operations

lean_obj_res lean_hesper_create_bind_group_layout(b_lean_obj_arg /* device */, b_lean_obj_arg entries, lean_obj_res /* unit */) {
    // Simplified: create a basic 1-buffer layout
    std::vector<wgpu::BindGroupLayoutEntry> layoutEntries(1);
    layoutEntries[0].binding = 0;
    layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    layoutEntries[0].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = layoutEntries.size();
    bglDesc.entries = layoutEntries.data();
    wgpu::BindGroupLayout bindGroupLayout = g_device.CreateBindGroupLayout(&bglDesc);

    // Wrap in External with registered class
    wgpu::BindGroupLayout* layoutPtr = new wgpu::BindGroupLayout(bindGroupLayout);
    lean_object* external = lean_alloc_external(g_webgpu_bind_group_layout_class, layoutPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_create_bind_group(b_lean_obj_arg /* device */, b_lean_obj_arg layout_obj,
                                            b_lean_obj_arg entries, lean_obj_res /* unit */) {
    // Extract layout from External
    wgpu::BindGroupLayout* layout = (wgpu::BindGroupLayout*)lean_get_external_data(layout_obj);

    // Simplified: create bind group with layout
    // entries parameter would contain buffer bindings in a real implementation
    std::vector<wgpu::BindGroupEntry> bgEntries(1);
    bgEntries[0].binding = 0;
    // Note: In a real implementation, we'd extract buffer from entries parameter
    bgEntries[0].buffer = wgpu::Buffer();  // Placeholder
    bgEntries[0].size = 1024 * sizeof(float);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = *layout;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();
    wgpu::BindGroup bindGroup = g_device.CreateBindGroup(&bgDesc);

    // Wrap in External with registered class
    wgpu::BindGroup* groupPtr = new wgpu::BindGroup(bindGroup);
    lean_object* external = lean_alloc_external(g_webgpu_bind_group_class, groupPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_create_compute_pipeline(b_lean_obj_arg /* device */, b_lean_obj_arg desc, lean_obj_res /* unit */) {
    // desc should contain shader module and bind group layout
    // For simplified implementation, using placeholder

    wgpu::ComputePipelineDescriptor pipelineDesc{};
    // Note: In a real implementation, extract shader from desc parameter
    pipelineDesc.compute.module = wgpu::ShaderModule();  // Placeholder
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 0;
    layoutDesc.bindGroupLayouts = nullptr;
    pipelineDesc.layout = g_device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = g_device.CreateComputePipeline(&pipelineDesc);

    // Wrap in External with registered class
    wgpu::ComputePipeline* pipelinePtr = new wgpu::ComputePipeline(pipeline);
    lean_object* external = lean_alloc_external(g_webgpu_compute_pipeline_class, pipelinePtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_hesper_dispatch_compute(b_lean_obj_arg /* device */, b_lean_obj_arg pipeline_obj, b_lean_obj_arg bind_group_obj,
                                           uint32_t workgroupsX, uint32_t workgroupsY, uint32_t workgroupsZ, lean_obj_res /* unit */) {
    // Extract pipeline and bind group from External
    wgpu::ComputePipeline* pipeline = (wgpu::ComputePipeline*)lean_get_external_data(pipeline_obj);
    wgpu::BindGroup* bindGroup = (wgpu::BindGroup*)lean_get_external_data(bind_group_obj);

    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(*pipeline);
    pass.SetBindGroup(0, *bindGroup);
    pass.DispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    g_device.GetQueue().Submit(1, &commands);

    return lean_io_result_mk_ok(lean_box(0));
}

// Matrix multiplication with subgroup operations
lean_obj_res lean_hesper_matmul_subgroup(b_lean_obj_arg shader_lean, uint32_t m, uint32_t k, uint32_t n, lean_obj_res /* unit */) {
    std::cout << "[Hesper] Running Subgroup Matrix Multiplication" << std::endl;
    std::cout << "  Dimensions: " << m << "x" << k << " * " << n << "x" << k << std::endl;

    // Extract shader code from Lean string
    const char* shaderCode = lean_string_cstr(shader_lean);

    // Create buffers
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = m * k * sizeof(float);
    bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer bufferA = g_device.CreateBuffer(&bufferDesc);

    bufferDesc.size = n * k * sizeof(float);
    wgpu::Buffer bufferB = g_device.CreateBuffer(&bufferDesc);

    bufferDesc.size = m * n * sizeof(float);
    wgpu::Buffer bufferC = g_device.CreateBuffer(&bufferDesc);

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
    g_device.GetQueue().WriteBuffer(bufferA, 0, dataA.data(), m * k * sizeof(float));
    g_device.GetQueue().WriteBuffer(bufferB, 0, dataB.data(), n * k * sizeof(float));
    std::cout << "  Data uploaded" << std::endl;

    // Create shader module
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;
    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = g_device.CreateShaderModule(&shaderDesc);
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
    wgpu::BindGroupLayout bindGroupLayout = g_device.CreateBindGroupLayout(&bglDesc);

    // Create pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = g_device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = g_device.CreateComputePipeline(&pipelineDesc);
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
    wgpu::BindGroup bindGroup = g_device.CreateBindGroup(&bgDesc);

    // Dispatch compute
    uint32_t numWorkgroupsX = (m + 32 - 1) / 32;
    uint32_t numWorkgroupsY = (n + 16 - 1) / 16;

    std::cout << "  Dispatching (" << numWorkgroupsX << ", " << numWorkgroupsY << ", 1)" << std::endl;

    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    g_device.GetQueue().Submit(1, &commands);

    std::cout << "  ✓ Subgroup matmul completed!" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
}

// 4K Matrix multiplication benchmark with FLOPS calculation
lean_obj_res lean_hesper_matmul_subgroup_4k(b_lean_obj_arg shader_lean, lean_obj_res /* unit */) {
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
    wgpu::Buffer bufferA = g_device.CreateBuffer(&bufferDesc);

    bufferDesc.size = static_cast<size_t>(n) * k * sizeof(float);
    wgpu::Buffer bufferB = g_device.CreateBuffer(&bufferDesc);

    bufferDesc.size = static_cast<size_t>(m) * n * sizeof(float);
    wgpu::Buffer bufferC = g_device.CreateBuffer(&bufferDesc);

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

    g_device.GetQueue().WriteBuffer(bufferA, 0, dataA.data(), dataA.size() * sizeof(float));
    g_device.GetQueue().WriteBuffer(bufferB, 0, dataB.data(), dataB.size() * sizeof(float));

    // Create shader and pipeline
    std::cout << "  Compiling shader..." << std::endl;
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;
    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule shaderModule = g_device.CreateShaderModule(&shaderDesc);

    std::vector<wgpu::BindGroupLayoutEntry> entries(3);
    for (int i = 0; i < 3; i++) {
        entries[i].binding = i;
        entries[i].visibility = wgpu::ShaderStage::Compute;
        entries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = entries.size();
    bglDesc.entries = entries.data();
    wgpu::BindGroupLayout bindGroupLayout = g_device.CreateBindGroupLayout(&bglDesc);

    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";

    wgpu::PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.layout = g_device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipeline pipeline = g_device.CreateComputePipeline(&pipelineDesc);

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
    wgpu::BindGroup bindGroup = g_device.CreateBindGroup(&bgDesc);

    // Calculate workgroup count
    uint32_t numWorkgroupsX = (m + 32 - 1) / 32;
    uint32_t numWorkgroupsY = (n + 64 - 1) / 64;

    std::cout << "  Workgroups: (" << numWorkgroupsX << ", " << numWorkgroupsY << ", 1)" << std::endl;
    std::cout << "  Total workgroups: " << (numWorkgroupsX * numWorkgroupsY) << std::endl;
    std::cout << std::endl;

    // Warmup run
    std::cout << "  Warmup run..." << std::endl;
    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    g_device.GetQueue().Submit(1, &commands);

    // Benchmark runs
    std::cout << "  Running " << nIter << " benchmark iterations..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < nIter; iter++) {
        wgpu::CommandEncoder enc = g_device.CreateCommandEncoder();
        wgpu::ComputePassEncoder p = enc.BeginComputePass();
        p.SetPipeline(pipeline);
        p.SetBindGroup(0, bindGroup);
        p.DispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
        p.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        g_device.GetQueue().Submit(1, &cmd);
    }

    // Wait for completion (simple approach - just tick a bunch)
    for (int i = 0; i < 1000; i++) {
        g_device.Tick();
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

}
