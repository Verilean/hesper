-- This module serves as the root of the `Hesper` library.
-- Import modules here that should be built as part of the library.
import Hesper.Basic

-- WGSL DSL modules
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Shader
import Hesper.WGSL.Kernel
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WGSL.Execute

-- Profiling modules
import Hesper.Profile
import Hesper.Profile.Trace

-- WebGPU API modules
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline

-- High-level Compute API
import Hesper.Compute

-- Tensor operations
import Hesper.Tensor.Types
import Hesper.Tensor.MatMul

-- Neural network operations
import Hesper.NN.Activation
import Hesper.NN.Conv

-- Automatic differentiation
import Hesper.AD.Reverse

-- Optimizers
import Hesper.Optimizer.SGD
import Hesper.Optimizer.Adam

-- Async operations
import Hesper.Async

-- GLFW window management and rendering
import Hesper.GLFW.Types
import Hesper.GLFW.Internal
import Hesper.GLFW

namespace Hesper

/-- Initialize the Hesper WebGPU engine.
    Discovers available GPU adapters and sets up the Dawn instance. -/
@[extern "lean_hesper_init"]
opaque init : IO Unit

/-- Run GPU vector addition (Hello World compute example).
    Adds two vectors of the given size element-wise on the GPU. -/
@[extern "lean_hesper_vector_add"]
opaque vectorAdd (size : UInt32) : IO Unit

end Hesper
