import Hesper.GGUF.Reader
import Hesper.Tokenizer.SentencePiece

/-!
# BitNet Tokenizer Demo

Demonstrates the tokenizer working with GGUF models:
1. Load GGUF model
2. Create tokenizer from vocabulary
3. Encode input text to tokens
4. Decode tokens back to text
5. Test roundtrip

Note: Full inference requires WebGPU device initialization which
depends on the specific runtime environment.
-/

open Hesper.GGUF
open Hesper.Tokenizer.SentencePiece

def main (args : List String) : IO Unit := do
  -- Parse command line arguments
  if args.length < 2 then
    IO.println "Usage: bitnet_tokenizer <gguf_model> <text1> [text2] [text3] ..."
    IO.println ""
    IO.println "Arguments:"
    IO.println "  gguf_model  - Path to GGUF model file (for vocabulary)"
    IO.println "  text1...    - Text strings to tokenize"
    IO.println ""
    IO.println "Examples:"
    IO.println "  bitnet_tokenizer model.gguf \"Hello world\""
    IO.println "  bitnet_tokenizer model.gguf \"The quick brown fox\" \"jumps over the lazy dog\""
    return

  let ggufPath := args[0]!
  let texts := args.tail

  IO.println "═══════════════════════════════════════════════"
  IO.println "  BitNet Tokenizer Demo"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Model: {ggufPath}"
  IO.println s!"Number of texts: {texts.length}"
  IO.println ""

  -- Step 1: Load GGUF file
  IO.println "[1/3] Loading GGUF model..."
  let gguf ← loadGGUF ggufPath
  IO.println s!"  ✓ Loaded {gguf.tensors.size} tensors"
  IO.println s!"  ✓ Metadata keys: {gguf.metadata.size}"

  -- Step 2: Create tokenizer
  IO.println ""
  IO.println "[2/3] Creating tokenizer..."
  let tokenizer ← fromGGUF gguf true false
  printInfo tokenizer

  -- Step 3: Process each text
  IO.println ""
  IO.println "[3/3] Tokenizing inputs..."
  IO.println "─────────────────────────────────────────"

  for idx in [:texts.length] do
    let text := texts[idx]!
    IO.println ""
    IO.println s!"Text {idx + 1}: \"{text}\""

    -- Encode
    let tokens := encode tokenizer text
    IO.println s!"  Tokens: {tokens}"
    IO.println s!"  Count: {tokens.size}"

    -- Show token details
    IO.println "  Token details:"
    for i in [:tokens.size] do
      let tokenId := tokens[i]!
      if tokenId < tokenizer.vocab.tokens.size then
        let tokenInfo := tokenizer.vocab.tokens[tokenId]!
        IO.println s!"    [{i}] ID={tokenId} piece=\"{tokenInfo.piece}\""

    -- Decode
    let decoded := decode tokenizer tokens
    IO.println s!"  Decoded: \"{decoded}\""

    -- Check roundtrip
    if text.trim == decoded then
      IO.println "  ✓ Roundtrip successful"
    else
      IO.println "  ✗ Roundtrip failed"
      IO.println s!"    Expected: \"{text.trim}\""
      IO.println s!"    Got:      \"{decoded}\""

  IO.println ""
  IO.println "─────────────────────────────────────────"
  IO.println "✓ Tokenization demo complete!"
