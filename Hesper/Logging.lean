/-!
# Global Logging Control

Provides a global verbosity flag to control debug output during inference.

## How to silence Hesper logs

Three options, in order of convenience:

1. **Environment variable** (recommended for notebooks and CLI):
   ```
   HESPER_LOG_LEVEL=quiet   # or: silent, error, off, 0, false, none
   ```
   This is read once at process start by both the Lean and C++ bridge
   sides.  Anything else (or unset) keeps logs ON for backward compat.

2. **Programmatic override** (inside a notebook cell, after a stray
   verbose call has already happened):
   ```lean
   import Hesper.Logging
   #eval Hesper.Logging.setVerbose false
   ```
   This flips both Lean and C++ flags at runtime.  Subsequent
   `getDevice` / `createBuffer` / etc. calls run silently.

3. **Per-cell suppression**: wrap the noisy code in `IO.FS.withIsolatedStreams`.
-/

namespace Hesper.Logging

/-- Returns `true` unless `HESPER_LOG_LEVEL` is set to a "quiet" value.
    Recognised quiet values (case-insensitive): silent, quiet, error,
    warn, warning, off, 0, false, none.  Matches the C++ bridge's
    initialisation logic in `native/bridge.cpp::g_verbose`. -/
private def envWantsVerbose : IO Bool := do
  match ← IO.getEnv "HESPER_LOG_LEVEL" with
  | some v =>
    let lc := v.toLower
    pure <| !(lc == "silent" || lc == "quiet" || lc == "error" ||
              lc == "warn"   || lc == "warning" ||
              lc == "off"    || lc == "0" || lc == "false" || lc == "none")
  | none => pure true

/-- Global verbosity flag.  Default reflects `HESPER_LOG_LEVEL`; falls
    back to `true` for backward compatibility when the env var is unset. -/
initialize verboseRef : IO.Ref Bool ← IO.mkRef (← envWantsVerbose)

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
