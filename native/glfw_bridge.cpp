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

// Global state
extern std::unique_ptr<dawn::native::Instance> g_instance;
extern wgpu::Device g_device;

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
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to initialize GLFW")));
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

    // Tell GLFW not to create an OpenGL context (we're using WebGPU)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "[C++] Failed to create GLFW window" << std::endl;
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create window")));
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

lean_obj_res lean_glfw_create_surface(size_t device_handle, b_lean_obj_arg window_obj,
                                       lean_obj_res /* unit */) {
    std::cout << "[C++] Creating surface for window..." << std::endl;
    std::cout << "[C++] window_obj type: " << lean_obj_tag(window_obj) << std::endl;
    std::cout << "[C++] is_external: " << lean_is_external(window_obj) << std::endl;
    std::cout.flush();

    GLFWwindow* window = (GLFWwindow*)lean_get_external_data(window_obj);
    std::cout << "[C++] Window pointer extracted: " << (void*)window << std::endl;
    std::cout.flush();

    wgpu::Instance instance = g_instance->Get();

    // Use Dawn's GLFW utility to create surface (handles all platform-specific details)
    std::cout << "[C++] Calling wgpu::glfw::CreateSurfaceForWindow..." << std::endl;
    std::cout.flush();
    wgpu::Surface surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);

    if (!surface) {
        std::cerr << "[C++] Failed to create surface!" << std::endl;
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create surface")));
    }

    std::cout << "[C++] Surface created successfully" << std::endl;

    // Store the surface and wrap in External with registered class
    wgpu::Surface* surfacePtr = new wgpu::Surface(surface);
    lean_object* external = lean_alloc_external(g_surface_class, surfacePtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_configure_surface(b_lean_obj_arg surface_obj, uint32_t width,
                                          uint32_t height, uint32_t format,
                                          lean_obj_res /* unit */) {
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    std::cout << "[C++] Configuring surface..." << std::endl;

    wgpu::SurfaceConfiguration config{};
    config.device = g_device;
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

lean_obj_res lean_glfw_surface_get_preferred_format(b_lean_obj_arg surface_obj,
                                                      lean_obj_res /* unit */) {
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    wgpu::SurfaceCapabilities capabilities;
    surface->GetCapabilities(g_device.GetAdapter(), &capabilities);

    wgpu::TextureFormat format = capabilities.formats[0];
    return lean_io_result_mk_ok(lean_box((uint32_t)format));
}

lean_obj_res lean_glfw_surface_get_current_texture(b_lean_obj_arg surface_obj,
                                                     lean_obj_res /* unit */) {
    wgpu::Surface* surface = (wgpu::Surface*)lean_get_external_data(surface_obj);

    wgpu::SurfaceTexture surfaceTexture;
    surface->GetCurrentTexture(&surfaceTexture);

    if (!surfaceTexture.texture) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to get current texture")));
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
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create texture view")));
    }

    wgpu::TextureView* viewPtr = new wgpu::TextureView(view);
    lean_object* external = lean_alloc_external(g_texture_view_class, viewPtr);

    return lean_io_result_mk_ok(external);
}

//=============================================================================
// Command Encoding
//=============================================================================

lean_obj_res lean_glfw_create_command_encoder(size_t device_handle,
                                               lean_obj_res /* unit */) {
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder(&encoderDesc);

    if (!encoder) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create command encoder")));
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
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to begin render pass")));
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
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to finish encoder")));
    }

    wgpu::CommandBuffer* cmdPtr = new wgpu::CommandBuffer(commandBuffer);
    lean_object* external = lean_alloc_external(g_command_buffer_class, cmdPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_queue_submit(size_t device_handle, b_lean_obj_arg cmd_obj,
                                     lean_obj_res /* unit */) {
    wgpu::CommandBuffer* cmd = (wgpu::CommandBuffer*)lean_get_external_data(cmd_obj);

    wgpu::Queue queue = g_device.GetQueue();
    queue.Submit(1, cmd);

    return lean_io_result_mk_ok(lean_box(0));
}

//=============================================================================
// Pipeline Management
//=============================================================================

lean_obj_res lean_glfw_create_shader_module(size_t device_handle, b_lean_obj_arg code_obj,
                                             lean_obj_res /* unit */) {
    const char* code = lean_string_cstr(code_obj);

    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;

    wgpu::ShaderModule shader = g_device.CreateShaderModule(&shaderDesc);

    if (!shader) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create shader module")));
    }

    wgpu::ShaderModule* shaderPtr = new wgpu::ShaderModule(shader);
    lean_object* external = lean_alloc_external(g_shader_module_class, shaderPtr);

    return lean_io_result_mk_ok(external);
}

lean_obj_res lean_glfw_create_render_pipeline(size_t device_handle, b_lean_obj_arg shader_obj,
                                               uint32_t format, lean_obj_res /* unit */) {
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

    wgpu::RenderPipeline pipeline = g_device.CreateRenderPipeline(&pipelineDesc);

    if (!pipeline) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("Failed to create render pipeline")));
    }

    wgpu::RenderPipeline* pipelinePtr = new wgpu::RenderPipeline(pipeline);
    lean_object* external = lean_alloc_external(g_render_pipeline_class, pipelinePtr);

    return lean_io_result_mk_ok(external);
}

} // extern "C"
