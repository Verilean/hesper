/-!
# WebGPU Error Types

This module defines comprehensive error types for WebGPU operations.
All FFI operations should return `Except WebGPUError α` or `IO (Except WebGPUError α)`
to ensure no silent failures.

## Error Categories

- `DeviceError`: GPU device initialization and management errors
- `BufferError`: GPU buffer allocation and access errors
- `ShaderError`: Shader compilation and validation errors
- `PipelineError`: Pipeline creation and configuration errors
- `SurfaceError`: Window surface and presentation errors
- `ValidationError`: Input validation and precondition violations

## Error Handling Strategy

1. All FFI functions return Result types (IO (Except Error A))
2. Errors include context (operation, resource, expected vs actual)
3. Errors provide recovery suggestions when possible
4. No exceptions/panics in production code
-/

namespace Hesper.WebGPU

/-- GPU backend types -/
inductive BackendType where
  | Null
  | WebGPU
  | D3D11
  | D3D12
  | Metal
  | Vulkan
  | OpenGL
  | OpenGLES
  | Unknown
  deriving Repr, BEq, Inhabited

/-- Convert backend type to readable string -/
def BackendType.toString : BackendType → String
  | .Null => "Null"
  | .WebGPU => "WebGPU"
  | .D3D11 => "D3D11"
  | .D3D12 => "D3D12"
  | .Metal => "Metal"
  | .Vulkan => "Vulkan"
  | .OpenGL => "OpenGL"
  | .OpenGLES => "OpenGLES"
  | .Unknown => "Unknown"

/-- Device initialization and management errors -/
inductive DeviceError where
  | NoAdaptersFound : DeviceError
  | InvalidAdapterIndex (requested : Nat) (available : Nat) : DeviceError
  | DeviceCreationFailed (adapterIdx : Nat) (reason : String) : DeviceError
  | DeviceLost (reason : String) : DeviceError
  | BackendNotSupported (backend : BackendType) : DeviceError
  | InitializationFailed (reason : String) : DeviceError
  deriving Repr, Inhabited

def DeviceError.toString : DeviceError → String
  | .NoAdaptersFound =>
      "No GPU adapters found.\n" ++
      "Recovery: Ensure GPU drivers are installed and up to date."
  | .InvalidAdapterIndex requested available =>
      s!"Invalid GPU adapter index: {requested} (only {available} adapters available).\n" ++
      s!"Recovery: Use index 0 to {available - 1}. Call getAdapterCount() to check available adapters."
  | .DeviceCreationFailed idx reason =>
      s!"Failed to create GPU device from adapter {idx}: {reason}\n" ++
      "Recovery: Try a different adapter or check GPU availability."
  | .DeviceLost reason =>
      s!"GPU device lost: {reason}\n" ++
      "Recovery: Recreate the device. This may indicate driver issues or GPU reset."
  | .BackendNotSupported backend =>
      s!"GPU backend not supported: {backend.toString}\n" ++
      "Recovery: Try a different backend (e.g., Vulkan on Linux, Metal on macOS, D3D12 on Windows)."
  | .InitializationFailed reason =>
      s!"WebGPU initialization failed: {reason}\n" ++
      "Recovery: Check that Dawn/WebGPU is properly installed and configured."

/-- Buffer allocation and access errors -/
inductive BufferError where
  | AllocationFailed (size : Nat) (reason : String) : BufferError
  | MappingFailed (reason : String) : BufferError
  | UnmappingFailed (reason : String) : BufferError
  | OutOfBounds (requested : Nat) (size : Nat) : BufferError
  | InvalidSize (requested : Nat) (reason : String) : BufferError
  | InvalidUsage (usage : String) (reason : String) : BufferError
  | AlreadyMapped : BufferError
  | NotMapped : BufferError
  deriving Repr, Inhabited

def BufferError.toString : BufferError → String
  | .AllocationFailed size reason =>
      s!"Failed to allocate GPU buffer of size {size} bytes: {reason}\n" ++
      "Recovery: Reduce buffer size or free existing buffers."
  | .MappingFailed reason =>
      s!"Failed to map GPU buffer: {reason}\n" ++
      "Recovery: Ensure buffer was created with MAP_READ or MAP_WRITE usage."
  | .UnmappingFailed reason =>
      s!"Failed to unmap GPU buffer: {reason}\n" ++
      "Recovery: This may indicate corruption. Consider recreating the buffer."
  | .OutOfBounds requested size =>
      s!"Buffer access out of bounds: requested offset/size {requested} but buffer size is {size}\n" ++
      "Recovery: Check array indices and buffer dimensions."
  | .InvalidSize requested reason =>
      s!"Invalid buffer size {requested}: {reason}\n" ++
      "Recovery: Buffer size must be positive and aligned to 4 bytes."
  | .InvalidUsage usage reason =>
      s!"Invalid buffer usage '{usage}': {reason}\n" ++
      "Recovery: Check WebGPU buffer usage flags documentation."
  | .AlreadyMapped =>
      "Buffer is already mapped.\n" ++
      "Recovery: Unmap the buffer before mapping again."
  | .NotMapped =>
      "Buffer is not currently mapped.\n" ++
      "Recovery: Map the buffer before accessing its contents."

/-- Shader compilation and validation errors -/
inductive ShaderError where
  | CompilationFailed (source : String) (errors : String) : ShaderError
  | ValidationFailed (reason : String) : ShaderError
  | InvalidEntryPoint (entryPoint : String) (availableEntryPoints : List String) : ShaderError
  | TypeMismatch (expected : String) (actual : String) (location : String) : ShaderError
  | MissingBindGroup (group : Nat) : ShaderError
  | InvalidWorkgroupSize (x : Nat) (y : Nat) (z : Nat) (reason : String) : ShaderError
  deriving Repr, Inhabited

def ShaderError.toString : ShaderError → String
  | .CompilationFailed source errors =>
      s!"Shader compilation failed:\n{errors}\n" ++
      s!"Source (first 200 chars): {source.take 200}\n" ++
      "Recovery: Check shader syntax and WGSL language version."
  | .ValidationFailed reason =>
      s!"Shader validation failed: {reason}\n" ++
      "Recovery: Ensure shader conforms to WebGPU requirements."
  | .InvalidEntryPoint requested available =>
      s!"Entry point '{requested}' not found. Available: {available}\n" ++
      "Recovery: Use one of the available entry points or add @vertex/@fragment/@compute attribute."
  | .TypeMismatch expected actual location =>
      s!"Type mismatch at {location}: expected '{expected}', got '{actual}'\n" ++
      "Recovery: Ensure shader types match buffer/binding layouts."
  | .MissingBindGroup group =>
      s!"Missing bind group {group}\n" ++
      "Recovery: Create and set bind group before drawing/dispatching."
  | .InvalidWorkgroupSize x y z reason =>
      s!"Invalid workgroup size ({x}, {y}, {z}): {reason}\n" ++
      "Recovery: Workgroup size must not exceed GPU limits (typically 256 total threads)."

/-- Pipeline creation and configuration errors -/
inductive PipelineError where
  | CreationFailed (pipelineType : String) (reason : String) : PipelineError
  | InvalidLayout (reason : String) : PipelineError
  | ShaderMismatch (reason : String) : PipelineError
  | InvalidVertexFormat (format : String) (reason : String) : PipelineError
  | InvalidColorFormat (format : String) (reason : String) : PipelineError
  deriving Repr, Inhabited

def PipelineError.toString : PipelineError → String
  | .CreationFailed pipelineType reason =>
      s!"Failed to create {pipelineType} pipeline: {reason}\n" ++
      "Recovery: Check pipeline descriptor and shader compatibility."
  | .InvalidLayout reason =>
      s!"Invalid pipeline layout: {reason}\n" ++
      "Recovery: Ensure bind group layouts match shader expectations."
  | .ShaderMismatch reason =>
      s!"Shader stage mismatch: {reason}\n" ++
      "Recovery: Ensure vertex/fragment shaders have compatible inputs/outputs."
  | .InvalidVertexFormat format reason =>
      s!"Invalid vertex format '{format}': {reason}\n" ++
      "Recovery: Use WebGPU-supported vertex formats (float32, uint32, etc.)."
  | .InvalidColorFormat format reason =>
      s!"Invalid color attachment format '{format}': {reason}\n" ++
      "Recovery: Use render-target-compatible formats (rgba8unorm, bgra8unorm, etc.)."

/-- Window surface and presentation errors -/
inductive SurfaceError where
  | WindowCreationFailed (width : Nat) (height : Nat) (reason : String) : SurfaceError
  | SurfaceCreationFailed (reason : String) : SurfaceError
  | ConfigurationFailed (reason : String) : SurfaceError
  | PresentationFailed (reason : String) : SurfaceError
  | NoSupportedFormat : SurfaceError
  | InvalidDimensions (width : Nat) (height : Nat) (reason : String) : SurfaceError
  deriving Repr, Inhabited

def SurfaceError.toString : SurfaceError → String
  | .WindowCreationFailed w h reason =>
      s!"Failed to create window ({w}x{h}): {reason}\n" ++
      "Recovery: Check GLFW initialization and monitor availability."
  | .SurfaceCreationFailed reason =>
      s!"Failed to create rendering surface: {reason}\n" ++
      "Recovery: Ensure window and device are valid."
  | .ConfigurationFailed reason =>
      s!"Failed to configure surface: {reason}\n" ++
      "Recovery: Check surface format and dimensions."
  | .PresentationFailed reason =>
      s!"Failed to present frame: {reason}\n" ++
      "Recovery: This may indicate window was destroyed or GPU error occurred."
  | .NoSupportedFormat =>
      "No supported surface format found.\n" ++
      "Recovery: This platform may not support WebGPU rendering."
  | .InvalidDimensions w h reason =>
      s!"Invalid surface dimensions ({w}x{h}): {reason}\n" ++
      "Recovery: Dimensions must be positive and within GPU limits."

/-- Input validation and precondition errors -/
inductive ValidationError where
  | NullPointer (paramName : String) : ValidationError
  | InvalidParameter (paramName : String) (value : String) (reason : String) : ValidationError
  | PreconditionFailed (condition : String) (reason : String) : ValidationError
  | InvalidState (expectedState : String) (actualState : String) : ValidationError
  deriving Repr, Inhabited

def ValidationError.toString : ValidationError → String
  | .NullPointer param =>
      s!"Null pointer for parameter '{param}'.\n" ++
      "Recovery: This is a programming error. Ensure all required parameters are provided."
  | .InvalidParameter param value reason =>
      s!"Invalid parameter '{param}' = '{value}': {reason}\n" ++
      "Recovery: Check API documentation for valid parameter ranges."
  | .PreconditionFailed condition reason =>
      s!"Precondition failed: {condition}\n{reason}\n" ++
      "Recovery: Ensure required operations are completed before calling this function."
  | .InvalidState expected actual =>
      s!"Invalid state: expected '{expected}', but state is '{actual}'.\n" ++
      "Recovery: Check operation order and state transitions."

/-- Unified WebGPU error type -/
inductive WebGPUError where
  | Device (err : DeviceError) : WebGPUError
  | Buffer (err : BufferError) : WebGPUError
  | Shader (err : ShaderError) : WebGPUError
  | Pipeline (err : PipelineError) : WebGPUError
  | Surface (err : SurfaceError) : WebGPUError
  | Validation (err : ValidationError) : WebGPUError
  | Unknown (message : String) : WebGPUError
  deriving Repr, Inhabited

def WebGPUError.toString : WebGPUError → String
  | .Device err => s!"[Device Error] {err.toString}"
  | .Buffer err => s!"[Buffer Error] {err.toString}"
  | .Shader err => s!"[Shader Error] {err.toString}"
  | .Pipeline err => s!"[Pipeline Error] {err.toString}"
  | .Surface err => s!"[Surface Error] {err.toString}"
  | .Validation err => s!"[Validation Error] {err.toString}"
  | .Unknown msg => s!"[Unknown Error] {msg}"

instance : ToString WebGPUError where
  toString := WebGPUError.toString

/-- Convert WebGPUError to IO.Error for throwing -/
def WebGPUError.toIOError (err : WebGPUError) : IO.Error :=
  IO.userError err.toString

/-- Throw a WebGPU error in IO context -/
def throwError (err : WebGPUError) : IO α :=
  throw err.toIOError

end Hesper.WebGPU

-- Helper namespace to create device errors
namespace Hesper.WebGPU.DeviceError
  def noAdapters : WebGPUError := .Device .NoAdaptersFound
  def invalidIndex (req : Nat) (avail : Nat) : WebGPUError :=
    .Device (.InvalidAdapterIndex req avail)
  def creationFailed (idx : Nat) (reason : String) : WebGPUError :=
    .Device (.DeviceCreationFailed idx reason)
end Hesper.WebGPU.DeviceError

-- Helper namespace to create buffer errors
namespace Hesper.WebGPU.BufferError
  def allocationFailed (size : Nat) (reason : String) : WebGPUError :=
    .Buffer (.AllocationFailed size reason)
  def outOfBounds (req : Nat) (size : Nat) : WebGPUError :=
    .Buffer (.OutOfBounds req size)
end Hesper.WebGPU.BufferError

-- Helper namespace to create shader errors
namespace Hesper.WebGPU.ShaderError
  def compilationFailed (src : String) (errs : String) : WebGPUError :=
    .Shader (.CompilationFailed src errs)
  def validationFailed (reason : String) : WebGPUError :=
    .Shader (.ValidationFailed reason)
end Hesper.WebGPU.ShaderError

-- Helper namespace to create pipeline errors
namespace Hesper.WebGPU.PipelineError
  def creationFailed (pipelineType : String) (reason : String) : WebGPUError :=
    .Pipeline (.CreationFailed pipelineType reason)
  def invalidLayout (reason : String) : WebGPUError :=
    .Pipeline (.InvalidLayout reason)
end Hesper.WebGPU.PipelineError

-- Helper namespace to create surface errors
namespace Hesper.WebGPU.SurfaceError
  def windowCreationFailed (width : Nat) (height : Nat) (reason : String) : WebGPUError :=
    .Surface (.WindowCreationFailed width height reason)
  def surfaceCreationFailed (reason : String) : WebGPUError :=
    .Surface (.SurfaceCreationFailed reason)
  def configurationFailed (reason : String) : WebGPUError :=
    .Surface (.ConfigurationFailed reason)
end Hesper.WebGPU.SurfaceError

-- Helper namespace to create validation errors
namespace Hesper.WebGPU.ValidationError
  def nullPointer (paramName : String) : WebGPUError :=
    .Validation (.NullPointer paramName)
  def invalidParameter (paramName : String) (value : String) (reason : String) : WebGPUError :=
    .Validation (.InvalidParameter paramName value reason)
  def preconditionFailed (condition : String) (reason : String) : WebGPUError :=
    .Validation (.PreconditionFailed condition reason)
end Hesper.WebGPU.ValidationError
