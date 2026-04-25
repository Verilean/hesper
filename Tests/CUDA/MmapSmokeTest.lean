import Hesper.CUDA.FFI

def main (args : List String) : IO UInt32 := do
  IO.println "[A] enter main"
  (← IO.getStdout).flush
  let path := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  IO.println s!"[B] path={path}"
  (← IO.getStdout).flush
  let mmap ← Hesper.CUDA.mmapOpen path
  IO.println "[C] mmapOpen OK"
  (← IO.getStdout).flush
  let sz ← Hesper.CUDA.mmapSize mmap
  IO.println s!"[D] size={sz}"
  (← IO.getStdout).flush

  Hesper.CUDA.cuDriverInit
  IO.println "[E] cuDriverInit OK"
  (← IO.getStdout).flush
  let dev ← Hesper.CUDA.cuDeviceGet 0
  IO.println s!"[F] cuDeviceGet OK dev={dev}"
  (← IO.getStdout).flush
  let _ctx ← Hesper.CUDA.cuCtxCreate dev
  IO.println "[G] cuCtxCreate OK"
  (← IO.getStdout).flush
  let stream ← Hesper.CUDA.cuStreamCreate
  IO.println "[H] cuStreamCreate OK"
  (← IO.getStdout).flush

  let dst ← Hesper.CUDA.cuMalloc 4096
  IO.println "[I] cuMalloc OK"
  (← IO.getStdout).flush
  Hesper.CUDA.cuMemcpyHtoDFromMmap dst mmap 0 4096 0
  IO.println "[J] sync H2D OK"
  (← IO.getStdout).flush

  Hesper.CUDA.cuMemcpyHtoDFromMmap dst mmap 0 4096 stream
  IO.println "[K] async H2D submitted"
  (← IO.getStdout).flush
  Hesper.CUDA.cuStreamSynchronize stream
  IO.println "[L] streamSync OK"
  (← IO.getStdout).flush

  Hesper.CUDA.cuFree dst
  IO.println "[M] DONE"
  return 0
