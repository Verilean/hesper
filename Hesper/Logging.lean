/-!
# Global Logging Control

Provides a global verbosity flag to control debug output during inference.
Set to `false` for clean interactive output.
-/

namespace Hesper.Logging

/-- Global verbosity flag (default: true for backward compatibility) -/
initialize verboseRef : IO.Ref Bool ← IO.mkRef true

/-- FFI: Set C++ bridge verbose flag -/
@[extern "lean_hesper_set_verbose"]
opaque setNativeVerbose : Bool → IO Unit

/-- Set global verbosity (both Lean and C++ bridge) -/
def setVerbose (v : Bool) : IO Unit := do
  verboseRef.set v
  setNativeVerbose v

/-- Check if verbose logging is enabled -/
def isVerbose : IO Bool := verboseRef.get

/-- Print only when verbose mode is on -/
def logVerbose (msg : String) : IO Unit := do
  if ← isVerbose then IO.println msg

end Hesper.Logging
