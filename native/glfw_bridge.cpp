#include <lean/lean.h>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <webgpu/webgpu_glfw.h>

#include <iostream>
#include <memory>

// No global state - all resources managed via Lean GC

// Forward declaration of Instance External class (defined in bridge.cpp)
extern lean_external_class* g_webgpu_instance_class;

// External classes for proper GC management
static lean_external_class* g_window_class = nullptr;
static lean_external_class* g_surface_class = nullptr;
static lean_external_class* g_texture_class = nullptr;
static lean_external_class* g_texture_view_class = nullptr;
static lean_external_class* g_command_encoder_class = nullptr;
static lean_external_class* g_render_pass_encoder_class = nullptr;
static lean_external_class* g_command_buffer_class = nullptr;
static lean_external_class* g_shader_module_class = nullptr;
static lean_external_class* g_render_pipeline_class = nullptr;

//=============================================================================
// Error Handling Infrastructure (shared with bridge.cpp)
//=============================================================================

enum class ErrorCategory : uint32_t {
    Device = 0,
    Buffer = 1,
    Shader = 2,
    Pipeline = 3,
    Surface = 4,
    Validation = 5,
    Unknown = 6
};

enum class SurfaceErrorTag : uint32_t {
    WindowCreationFailed = 0,
    SurfaceCreationFailed = 1,
    ConfigurationFailed = 2,
    PresentationFailed = 3,
    NoSupportedFormat = 4,
    InvalidDimensions = 5
};

// Helper functions to create surface errors
static lean_object* make_surface_error_window_creation_failed(uint32_t width, uint32_t height, const char* reason) {
    lean_object* err = lean_alloc_ctor(static_cast<uint32_t>(SurfaceErrorTag::WindowCreationFailed), 3, 0);
    lean_ctor_set(err, 0, lean_box(width));
    lean_ctor_set(err, 1, lean_box(height));
    lean_ctor_set(err, 2, lean_mk_string(reason));
    return err;
}

static lean_object* make_surface_error_creation_failed(const char* reason) {
    lean_object* err = lean_alloc_ctor(static_cast<uint32_t>(SurfaceErrorTag::SurfaceCreationFailed), 1, 0);
    lean_ctor_set(err, 0, lean_mk_string(reason));
    return err;
}

static lean_object* make_surface_error_configuration_failed(const char* reason) {
    lean_object* err = lean_alloc_ctor(static_cast<uint32_t>(SurfaceErrorTag::ConfigurationFailed), 1, 0);
    lean_ctor_set(err, 0, lean_mk_string(reason));
    return err;
}

static lean_object* make_surface_error_presentation_failed(const char* reason) {
    lean_object* err = lean_alloc_ctor(static_cast<uint32_t>(SurfaceErrorTag::PresentationFailed), 1, 0);
    lean_ctor_set(err, 0, lean_mk_string(reason));
    return err;
}

// Shader error constructors
static lean_object* make_shader_error_compilation_failed(const char* source, const char* errors) {
    lean_object* err = lean_alloc_ctor(0, 2, 0);  // CompilationFailed tag 0
    lean_ctor_set(err, 0, lean_mk_string(source));
    lean_ctor_set(err, 1, lean_mk_string(errors));
    return err;
}

// Pipeline error constructors
static lean_object* make_pipeline_error_creation_failed(const char* pipelineType, const char* reason) {
    lean_object* err = lean_alloc_ctor(0, 2, 0);  // CreationFailed tag 0
    lean_ctor_set(err, 0, lean_mk_string(pipelineType));
    lean_ctor_set(err, 1, lean_mk_string(reason));
    return err;
}

// Wrap error in WebGPUError.Surface
static lean_object* make_webgpu_error(ErrorCategory category, lean_object* inner_error) {
    lean_object* err = lean_alloc_ctor(static_cast<uint32_t>(category), 1, 0);
    lean_ctor_set(err, 0, inner_error);
    return err;
}

// Create IO error from WebGPUError
static lean_object* make_io_error(lean_object* webgpu_error) {
    return lean_io_result_mk_error(lean_mk_io_user_error(webgpu_error));
}

// Convenience: Create IO error from surface error
static lean_object* surface_error(lean_object* surf_err) {
    return make_io_error(make_webgpu_error(ErrorCategory::Surface, surf_err));
}

// Convenience: Create IO error from shader error
static lean_object* shader_error(lean_object* shader_err) {
    return make_io_error(make_webgpu_error(ErrorCategory::Shader, shader_err));
}

// Convenience: Create IO error from pipeline error
static lean_object* pipeline_error(lean_object* pipeline_err) {
    return make_io_error(make_webgpu_error(ErrorCategory::Pipeline, pipeline_err));
}

extern "C" {

//=============================================================================
// Finalizers - Called by Lean GC when resources are collected
//=============================================================================

static void finalize_window(void* ptr) {
    GLFWwindow* window = (GLFWwindow*)ptr;
    if (window) {
        std::cout << "[C++] Finalizing window at " << ptr << std::endl;
        glfwDestroyWindow(window);
    }
}

static void finalize_surface(void* ptr) {
    wgpu::Surface* surface = (wgpu::Surface*)ptr;
    if (surface) {
        std::cout << "[C++] Finalizing surface" << std::endl;
        delete surface;
    }
}

static void finalize_texture(void* ptr) {
    wgpu::Texture* texture = (wgpu::Texture*)ptr;
    if (texture) {
        delete texture;
    }
}

static void finalize_texture_view(void* ptr) {
    wgpu::TextureView* view = (wgpu::TextureView*)ptr;
    if (view) {
        delete view;
    }
}

static void finalize_command_encoder(void* ptr) {
    wgpu::CommandEncoder* encoder = (wgpu::CommandEncoder*)ptr;
    if (encoder) {
        delete encoder;
    }
}

static void finalize_render_pass_encoder(void* ptr) {
    wgpu::RenderPassEncoder* pass = (wgpu::RenderPassEncoder*)ptr;
    if (pass) {
        delete pass;
    }
}

static void finalize_command_buffer(void* ptr) {
    wgpu::CommandBuffer* cmd = (wgpu::CommandBuffer*)ptr;
    if (cmd) {
        delete cmd;
    }
}

static void finalize_shader_module(void* ptr) {
    wgpu::ShaderModule* shader = (wgpu::ShaderModule*)ptr;
    if (shader) {
        delete shader;
    }
}

static void finalize_render_pipeline(void* ptr) {
    wgpu::RenderPipeline* pipeline = (wgpu::RenderPipeline*)ptr;
    if (pipeline) {
        delete pipeline;
    }
}

//=============================================================================
// External Class Registration
//=============================================================================

static void register_external_classes() {
    if (g_window_class != nullptr) return;  // Already registered

    g_window_class = lean_register_external_class(finalize_window, nullptr);
    g_surface_class = lean_register_external_class(finalize_surface, nullptr);
    g_texture_class = lean_register_external_class(finalize_texture, nullptr);
    g_texture_view_class = lean_register_external_class(finalize_texture_view, nullptr);
    g_command_encoder_class = lean_register_external_class(finalize_command_encoder, nullptr);
    g_render_pass_encoder_class = lean_register_external_class(finalize_render_pass_encoder, nullptr);
    g_command_buffer_class = lean_register_external_class(finalize_command_buffer, nullptr);
    g_shader_module_class = lean_register_external_class(finalize_shader_module, nullptr);
    g_render_pipeline_class = lean_register_external_class(finalize_render_pipeline, nullptr);

    std::cout << "[C++] External classes registered" << std::endl;
}

//=============================================================================
// GLFW Initialization
//=============================================================================

lean_obj_res lean_glfw_init(lean_obj_res /* unit */) {
    std::cout << "[C++] lean_glfw_init called" << std::endl;
    std::cout.flush();

    // Register external classes for GC management
    register_external_classes();

    if (!glfwInit()) {
        std::cerr << "[C++] glfwInit failed!" << std::endl;
        return surface_error(make_surface_error_window_creation_failed(0, 0, "Failed to initialize GLFW - check GLFW installation"));
    }

    std::cout << "[C++] glfwInit succeeded" << std::endl;
    std::cout.flush();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_terminate(lean_obj_res /* unit */) {
    glfwTerminate();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_window_should_close(b_lean_obj_arg window_obj, lean_obj_res /* unit */) {
    GLFWwindow* window = (GLFWwindow*)lean_get_external_data(window_obj);
    uint8_t result = glfwWindowShouldClose(window) ? 1 : 0;
    return lean_io_result_mk_ok(lean_box(result));
}

//=============================================================================
// Window Management
//=============================================================================

lean_obj_res lean_glfw_create_window(uint32_t width, uint32_t height,
                                      b_lean_obj_arg title_obj, lean_obj_res /* unit */) {
    const char* title = lean_string_cstr(title_obj);

    // Validate dimensions
    if (width == 0 || height == 0) {
        return surface_error(make_surface_error_window_creation_failed(width, height, "Window dimensions must be greater than 0"));
    }

    // Tell GLFW not to create an OpenGL context (we're using WebGPU)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "[C++] Failed to create GLFW window" << std::endl;
        return surface_error(make_surface_error_window_creation_failed(width, height, "GLFW window creation failed - check GLFW initialization and monitor availability"));
    }

    std::cout << "[C++] Created window at address: " << (void*)window << std::endl;

    // Wrap in External with registered class (finalizer already set in class)
    lean_object* external = lean_alloc_external(g_window_class, window);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_poll_events(lean_obj_res /* unit */) {
    glfwPollEvents();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_window_get_key(b_lean_obj_arg window_obj, uint32_t key,
                                       lean_obj_res /* unit */) {
    GLFWwindow* window = (GLFWwindow*)lean_get_external_data(window_obj);
    int state = glfwGetKey(window, key);
    return lean_io_result_mk_ok(lean_box(state));
}

//=============================================================================
// Surface Management
//=============================================================================

lean_obj_res lean_glfw_create_surface(b_lean_obj_arg instance_obj, b_lean_obj_arg window_obj,
                                       lean_obj_res /* unit */) {
    dawn::native::Instance* nativeInstance = static_cast<dawn::native::Instance*>(lean_get_external_data(instance_obj));

    std::cout << "[C++] Creating surface for window..." << std::endl;
    std::cout << "[C++] window_obj type: " << lean_obj_tag(window_obj) << std::endl;
    std::cout << "[C++] is_external: " << lean_is_external(window_obj) << std::endl;
    std::cout.flush();

    GLFWwindow* window = (GLFWwindow*)lean_get_external_data(window_obj);
    std::cout << "[C++] Window pointer extracted: " << (void*)window << std::endl;
    std::cout.flush();

    wgpu::Instance instance = nativeInstance->Get();

    // Use Dawn's GLFW utility to create surface (handles all platform-specific details)
    std::cout << "[C++] Calling wgpu::glfw::CreateSurfaceForWindow..." << std::endl;
    std::cout.flush();
    wgpu::Surface surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);

    if (!surface) {
        std::cerr << "[C++] Failed to create surface!" << std::endl;
        return surface_error(make_surface_error_creation_failed("Failed to create WebGPU surface for window - ensure window and device are valid"));
    }

    std::cout << "[C++] Surface created successfully" << std::endl;

    // Store the surface and wrap in External with registered class
    wgpu::Surface* surfacePtr = new wgpu::Surface(surface);
    lean_object* external = lean_alloc_external(g_surface_class, surfacePtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_configure_surface(b_lean_obj_arg device_obj, b_lean_obj_arg surface_obj,
                                          uint32_t width, uint32_t height, uint32_t format,
                                          lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    std::cout << "[C++] Configuring surface..." << std::endl;

    wgpu::SurfaceConfiguration config{};
    config.device = *device;
    config.width = width;
    config.height = height;
    config.format = (wgpu::TextureFormat)format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.presentMode = wgpu::PresentMode::Fifo;

    std::cout << "[C++] Calling surface->Configure()..." << std::endl;
    surface->Configure(&config);

    std::cout << "[C++] Surface configured successfully" << std::endl;

    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_surface_get_preferred_format(b_lean_obj_arg device_obj, b_lean_obj_arg surface_obj,
                                                      lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    wgpu::SurfaceCapabilities capabilities;
    surface->GetCapabilities(device->GetAdapter(), &capabilities);

    wgpu::TextureFormat format = capabilities.formats[0];
    return lean_io_result_mk_ok(lean_box((uint32_t)format));
}

lean_obj_res lean_glfw_surface_get_current_texture(b_lean_obj_arg surface_obj,
                                                     lean_obj_res /* unit */) {
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    wgpu::SurfaceTexture surfaceTexture;
    surface->GetCurrentTexture(&surfaceTexture);

    if (!surfaceTexture.texture) {
        return surface_error(make_surface_error_presentation_failed("Failed to get current texture - surface may be unconfigured or window minimized"));
    }

    wgpu::Texture* texture = new wgpu::Texture(surfaceTexture.texture);
    lean_object* external = lean_alloc_external(g_texture_class, texture);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_surface_present(b_lean_obj_arg surface_obj, lean_obj_res /* unit */) {
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);
    surface->Present();
    return lean_io_result_mk_ok(lean_box(0));
}

//=============================================================================
// Texture Management
//=============================================================================

lean_obj_res lean_glfw_texture_create_view(b_lean_obj_arg texture_obj, lean_obj_res /* unit */) {
    wgpu::Texture* texture = (wgpu::Texture*)lean_get_external_data(texture_obj);

    wgpu::TextureViewDescriptor viewDesc{};
    wgpu::TextureView view = texture->CreateView(&viewDesc);

    if (!view) {
        return surface_error(make_surface_error_presentation_failed("Failed to create texture view"));
    }

    wgpu::TextureView* viewPtr = new wgpu::TextureView(view);
    lean_object* external = lean_alloc_external(g_texture_view_class, viewPtr);

    return lean_io_result_mk_ok(external);
}

//=============================================================================
// Command Encoding
//=============================================================================

lean_obj_res lean_glfw_create_command_encoder(b_lean_obj_arg device_obj,
                                               lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));

    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = device->CreateCommandEncoder(&encoderDesc);

    if (!encoder) {
        lean_object* unknown_err = lean_alloc_ctor(static_cast<uint32_t>(ErrorCategory::Unknown), 1, 0);
        lean_ctor_set(unknown_err, 0, lean_mk_string("Failed to create command encoder"));
        return make_io_error(unknown_err);
    }

    wgpu::CommandEncoder* encoderPtr = new wgpu::CommandEncoder(encoder);
    lean_object* external = lean_alloc_external(g_command_encoder_class, encoderPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_begin_render_pass(b_lean_obj_arg encoder_obj, b_lean_obj_arg view_obj,
                                          lean_obj_res /* unit */) {
    wgpu::CommandEncoder* encoder = (wgpu::CommandEncoder*)lean_get_external_data(encoder_obj);
    wgpu::TextureView* view = (wgpu::TextureView*)lean_get_external_data(view_obj);

    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = *view;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = {0.0, 0.0, 0.0, 1.0}; // Clear to black

    wgpu::RenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    wgpu::RenderPassEncoder pass = encoder->BeginRenderPass(&renderPassDesc);

    if (!pass) {
        lean_object* unknown_err = lean_alloc_ctor(static_cast<uint32_t>(ErrorCategory::Unknown), 1, 0);
        lean_ctor_set(unknown_err, 0, lean_mk_string("Failed to begin render pass"));
        return make_io_error(unknown_err);
    }

    wgpu::RenderPassEncoder* passPtr = new wgpu::RenderPassEncoder(pass);
    lean_object* external = lean_alloc_external(g_render_pass_encoder_class, passPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_render_pass_set_pipeline(b_lean_obj_arg pass_obj, b_lean_obj_arg pipeline_obj,
                                                  lean_obj_res /* unit */) {
    wgpu::RenderPassEncoder* pass = (wgpu::RenderPassEncoder*)lean_get_external_data(pass_obj);
    wgpu::RenderPipeline* pipeline = (wgpu::RenderPipeline*)lean_get_external_data(pipeline_obj);

    pass->SetPipeline(*pipeline);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_render_pass_draw(b_lean_obj_arg pass_obj, uint32_t vertex_count,
                                         lean_obj_res /* unit */) {
    wgpu::RenderPassEncoder* pass = (wgpu::RenderPassEncoder*)lean_get_external_data(pass_obj);
    pass->Draw(vertex_count, 1, 0, 0);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_render_pass_end(b_lean_obj_arg pass_obj, lean_obj_res /* unit */) {
    wgpu::RenderPassEncoder* pass = (wgpu::RenderPassEncoder*)lean_get_external_data(pass_obj);
    pass->End();
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res lean_glfw_encoder_finish(b_lean_obj_arg encoder_obj, lean_obj_res /* unit */) {
    wgpu::CommandEncoder* encoder = (wgpu::CommandEncoder*)lean_get_external_data(encoder_obj);

    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer commandBuffer = encoder->Finish(&cmdBufferDesc);

    if (!commandBuffer) {
        lean_object* unknown_err = lean_alloc_ctor(static_cast<uint32_t>(ErrorCategory::Unknown), 1, 0);
        lean_ctor_set(unknown_err, 0, lean_mk_string("Failed to finish encoder"));
        return make_io_error(unknown_err);
    }

    wgpu::CommandBuffer* cmdPtr = new wgpu::CommandBuffer(commandBuffer);
    lean_object* external = lean_alloc_external(g_command_buffer_class, cmdPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_queue_submit(b_lean_obj_arg device_obj, b_lean_obj_arg cmd_obj,
                                     lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));
    wgpu::CommandBuffer* cmd = (wgpu::CommandBuffer*)lean_get_external_data(cmd_obj);

    wgpu::Queue queue = device->GetQueue();
    queue.Submit(1, cmd);

    return lean_io_result_mk_ok(lean_box(0));
}

//=============================================================================
// Pipeline Management
//=============================================================================

lean_obj_res lean_glfw_create_shader_module(b_lean_obj_arg device_obj, b_lean_obj_arg code_obj,
                                             lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));
    const char* code = lean_string_cstr(code_obj);

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;

    wgpu::ShaderModule shader = device->CreateShaderModule(&shaderDesc);

    if (!shader) {
        return shader_error(make_shader_error_compilation_failed(code, "Shader module creation failed"));
    }

    wgpu::ShaderModule* shaderPtr = new wgpu::ShaderModule(shader);
    lean_object* external = lean_alloc_external(g_shader_module_class, shaderPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_create_render_pipeline(b_lean_obj_arg device_obj, b_lean_obj_arg shader_obj,
                                               uint32_t format, lean_obj_res /* unit */) {
    wgpu::Device* device = static_cast<wgpu::Device*>(lean_get_external_data(device_obj));
    wgpu::ShaderModule* shader = (wgpu::ShaderModule*)lean_get_external_data(shader_obj);

    // Vertex state
    wgpu::VertexState vertexState{};
    vertexState.module = *shader;
    vertexState.entryPoint = "vertexMain";

    // Fragment state
    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = (wgpu::TextureFormat)format;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fragmentState{};
    fragmentState.module = *shader;
    fragmentState.entryPoint = "fragmentMain";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    // Pipeline descriptor
    wgpu::RenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.vertex = vertexState;
    pipelineDesc.fragment = &fragmentState;
    pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;

    wgpu::RenderPipeline pipeline = device->CreateRenderPipeline(&pipelineDesc);

    if (!pipeline) {
        return pipeline_error(make_pipeline_error_creation_failed("render", "Render pipeline creation failed - check shader compatibility and descriptor"));
    }

    wgpu::RenderPipeline* pipelinePtr = new wgpu::RenderPipeline(pipeline);
    lean_object* external = lean_alloc_external(g_render_pipeline_class, pipelinePtr);

    return lean_io_result_mk_ok(external);
}

} // extern "C"
