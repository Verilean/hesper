import Hesper
import Hesper.WebGPU.Device
import Hesper.Models.DiffusionGemma.Loader

/-!
# DiffusionGemma native load smoke test (WebGPU/Metal)

Loads the real DiffusionGemma GGUF onto the Metal backend via Hesper's
native loader.  ~16 GB read + GPU upload (~32 GB peak on a 48 GB Mac).

Run:  lake exe diffusiongemma-load [path.gguf]
-/

open Hesper.WebGPU
open Hesper.Models.DiffusionGemma

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  IO.println "[diffusiongemma-load] init WebGPU (Metal)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let model ← DiffusionGemmaModel.fromGGUF (β := Device) device path
  let c := model.inner.config
  IO.println s!"✓ loaded on Metal: {model.inner.blocks.size} blocks, dim={c.hiddenSize}, experts={c.numExperts}/{c.numExpertsUsed}, canvas={model.dg.canvasLength}, maskTok={model.dg.maskTokenId}"
