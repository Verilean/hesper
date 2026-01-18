/-!
# GLFW Type Definitions

Opaque types for GLFW resources. Resources are managed via Lean External objects
with automatic cleanup by finalizers on the C++ side.
-/

namespace Hesper.GLFW

/-- Opaque window handle with automatic cleanup via External finalizer -/
opaque Window : Type

/-- Opaque surface handle with automatic cleanup via External finalizer -/
opaque Surface : Type

/-- Opaque texture handle with automatic cleanup via External finalizer -/
opaque Texture : Type

/-- Opaque texture view handle with automatic cleanup via External finalizer -/
opaque TextureView : Type

/-- Opaque command encoder handle with automatic cleanup via External finalizer -/
opaque CommandEncoder : Type

/-- Opaque render pass encoder handle with automatic cleanup via External finalizer -/
opaque RenderPassEncoder : Type

/-- Opaque command buffer handle with automatic cleanup via External finalizer -/
opaque CommandBuffer : Type

/-- Opaque shader module handle with automatic cleanup via External finalizer -/
opaque ShaderModule : Type

/-- Opaque render pipeline handle with automatic cleanup via External finalizer -/
opaque RenderPipeline : Type

/-- GLFW key codes -/
inductive Key
  | escape
  | w | a | s | d
  | space
  | enter
  | other (code : Nat)
  deriving Inhabited, Repr

namespace Key

def toNat : Key → Nat
  | escape => 256
  | w => 87
  | a => 65
  | s => 83
  | d => 68
  | space => 32
  | enter => 257
  | other code => code

end Key

/-- Key action states -/
inductive KeyAction
  | release
  | press
  | repeated  -- Note: 'repeat' is a reserved keyword in Lean
  deriving Inhabited, Repr, BEq

namespace KeyAction

def fromNat : Nat → KeyAction
  | 0 => release
  | 1 => press
  | 2 => repeated
  | _ => release

end KeyAction

end Hesper.GLFW
