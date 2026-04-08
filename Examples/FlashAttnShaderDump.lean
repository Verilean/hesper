import Hesper
import Hesper.WGSL.FlashAttention
import Hesper.WGSL.Execute

open Hesper.WGSL

/-- Dump the WGSL source of flashAttentionSWAKernel for a couple of (cacheLen, pos)
    values so we can inspect whether the generated loop bound follows cacheLen. -/
def main : IO Unit := do
  let numHeads := 4
  let numKVHeads := 2
  let maxSeqLen := 32768
  let headDim := 256
  let windowSize := 512
  let scale : Float := 1.0 / 16.0  -- 1/sqrt(256)

  for (cacheLen, pos) in [(1, 0), (2, 1), (3, 2)] do
    let src := Execute.compileToWGSL
      (FlashAttention.flashAttentionSWAKernel numHeads numKVHeads maxSeqLen headDim cacheLen windowSize pos scale)
    IO.println s!"===== SWA cacheLen={cacheLen} pos={pos} ====="
    -- print only the body around the outer attention loop for readability
    IO.println src
    IO.println ""
