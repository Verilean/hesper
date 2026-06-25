import Hesper.GGUF.Reader
import Hesper.GGUF.Loader
import Hesper.Models.DiffusionGemma.Config

/-!
# DiffusionGemma loader validation (Milestone 1)

Header-only validation that does NOT load the 16 GB tensor body or touch
the GPU.  It:
  1. parses the GGUF header via `loadGGUFHeaderMmap` (bounded prefix copy),
  2. parses `DiffusionConfig` and prints it,
  3. checks every tensor the verified architecture expects is present
     (and that no unexpected tensors are left over).

Run:  `lake exe diffusiongemma-validation [path-to.gguf]`
-/

open Hesper.Models.DiffusionGemma
open Hesper.Models.Gemma4 (Config LayerType)

/-- Per-layer tensor suffixes present in every block. `attn_v.weight` is
    intentionally excluded — global (full-attention) layers omit it
    (V reuses the K projection); it is added per-layer for SWA blocks. -/
def perLayerSuffixes : Array String := #[
  "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_output.weight",
  "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight",
  "layer_output_scale.weight", "enc_layer_output_scale.weight",
  "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
  "post_ffw_norm.weight",
  "ffn_gate_inp.weight", "ffn_gate_inp.scale",
  "ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_down_exps.scale",
  "pre_ffw_norm_2.weight", "post_ffw_norm_1.weight", "post_ffw_norm_2.weight"
]

/-- Global (non-block) tensors.  `self_cond_*` and `output.weight` are
    optional (self-conditioning is unused in the zero-SC baseline; the LM
    head is tied to `token_embd` when `output.weight` is absent). -/
def globalRequired : Array String := #["token_embd.weight", "output_norm.weight", "rope_freqs.weight"]
def globalOptional : Array String :=
  #["output.weight", "self_cond_pre_norm.weight", "self_cond_gate.weight",
    "self_cond_up.weight", "self_cond_down.weight"]

def main (args : List String) : IO Unit := do
  let path := args.head?.getD "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
  IO.println s!"[DiffusionGemma] Header-parsing {path} (no body load)..."
  let gguf ← Hesper.GGUF.loadGGUFHeader path

  let arch := (gguf.getMetadataString "general.architecture").getD "<none>"
  IO.println s!"  architecture : {arch}"
  IO.println s!"  tensors      : {gguf.tensors.size}"
  if arch != archName then
    IO.println s!"  ⚠ expected architecture '{archName}'"

  -- Parse config
  let cfg ← match DiffusionConfig.fromGGUF gguf with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"Config parse error: {e}")
  let b := cfg.base
  IO.println "── Parsed DiffusionConfig ──────────────────────────────"
  IO.println s!"  layers={b.numHiddenLayers} dim={b.hiddenSize} ffn={b.intermediateSize} heads={b.numAttentionHeads} vocab={b.vocabSize}"
  IO.println s!"  kv heads  : full={b.numKeyValueHeadsFull}  swa={b.numKeyValueHeadsSWA}"
  IO.println s!"  head dim  : full={b.headDimFull}  swa={b.headDimSWA}"
  IO.println s!"  rope theta: full={b.ropeTheta}  swa={b.ropeThetaSWA}  rmsEps={b.rmsNormEps}"
  IO.println s!"  MoE       : experts={b.numExperts} used={b.numExpertsUsed} expertFF={b.expertFFSize}"
  IO.println s!"  softcap={b.logitSoftcapScale} slidingWindow={b.slidingWindowSize}"
  IO.println s!"  diffusion : canvas={cfg.canvasLength} maskTok={cfg.maskTokenId} causal={cfg.causal} denoiseSteps={cfg.denoiseSteps}"
  let nFull := (List.range b.numHiddenLayers).countP (fun i => b.isFullAttention i)
  IO.println s!"  layer types: {b.numHiddenLayers - nFull} SWA + {nFull} full"

  -- Build the expected tensor set and check presence
  IO.println "── Tensor presence check ───────────────────────────────"
  let mut expected : Array String := #[]
  for n in globalRequired do
    expected := expected.push n
  for n in globalOptional do
    if (gguf.findTensor n).isSome then expected := expected.push n
  for li in [0:b.numHiddenLayers] do
    for s in perLayerSuffixes do
      expected := expected.push s!"blk.{li}.{s}"
    -- attn_v only on SWA layers
    if !b.isFullAttention li then
      expected := expected.push s!"blk.{li}.attn_v.weight"

  let mut missing : Array String := #[]
  for n in expected do
    if (gguf.findTensor n).isNone then missing := missing.push n

  let mut unexpected : Array String := #[]
  for ti in gguf.tensors do
    if !expected.contains ti.name then unexpected := unexpected.push ti.name

  IO.println s!"  expected tensors : {expected.size}"
  IO.println s!"  present          : {expected.size - missing.size}"
  IO.println s!"  missing          : {missing.size}"
  IO.println s!"  unexpected       : {unexpected.size}"
  if !missing.isEmpty then
    IO.println "  ── MISSING ──"
    for n in missing.toList.take 30 do IO.println s!"    - {n}"
  if !unexpected.isEmpty then
    IO.println "  ── UNEXPECTED (in file, not modelled) ──"
    for n in unexpected.toList.take 30 do IO.println s!"    + {n}"

  if missing.isEmpty && unexpected.isEmpty && gguf.tensors.size == expected.size then
    IO.println "✓ Milestone 1 PASS: config parsed and all tensors map exactly."
  else
    IO.println "✗ Milestone 1: mismatch — see lists above."
    throw (IO.userError "tensor map mismatch")
