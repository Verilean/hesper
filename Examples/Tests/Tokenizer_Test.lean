import Hesper.GGUF.Reader
import Hesper.Tokenizer.SentencePiece

/-!
# SentencePiece Tokenizer Test

Tests the tokenizer with actual GGUF vocab.
-/

open Hesper.GGUF
open Hesper.Tokenizer.SentencePiece

def main (args : List String) : IO Unit := do
  -- Check command line arguments
  if args.length < 1 then
    IO.println "Usage: tokenizer_test <gguf_file> [text]"
    IO.println "Example: tokenizer_test model.gguf \"Hello world\""
    return

  let ggufPath := args[0]!
  let testText := if args.length > 1 then args[1]! else "Hello world"

  IO.println "═══════════════════════════════════════════════"
  IO.println "  SentencePiece Tokenizer Test"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"GGUF file: {ggufPath}"
  IO.println s!"Test text: '{testText}'"
  IO.println ""

  -- Parse GGUF file
  IO.println "[1/4] Parsing GGUF file..."
  let gguf ← loadGGUF ggufPath
  IO.println s!"  Loaded {gguf.tensors.size} tensors"
  IO.println s!"  Metadata keys: {gguf.metadata.size}"

  -- Create tokenizer
  IO.println ""
  IO.println "[2/4] Creating tokenizer..."
  let tokenizer ← fromGGUF gguf true false

  -- Print tokenizer info
  IO.println ""
  IO.println "[3/4] Tokenizer information:"
  printInfo tokenizer

  -- Test tokenization
  IO.println ""
  IO.println "[4/4] Testing tokenization:"
  IO.println "─────────────────────────────────────────"

  -- Encode
  let tokens := encode tokenizer testText
  IO.println s!"Original text: '{testText}'"
  IO.println s!"Token IDs: {tokens}"
  IO.println s!"Number of tokens: {tokens.size}"

  -- Show individual tokens
  IO.println ""
  IO.println "Token details:"
  for i in [0:tokens.size] do
    let tokenId := tokens[i]!
    if tokenId < tokenizer.vocab.tokens.size then
      let tokenInfo := tokenizer.vocab.tokens[tokenId]!
      IO.println s!"  [{i}] ID={tokenId} piece=\"{tokenInfo.piece}\" score={tokenInfo.score}"

  -- Decode
  let decoded := decode tokenizer tokens
  IO.println ""
  IO.println s!"Decoded text: '{decoded}'"

  -- Check roundtrip
  IO.println ""
  if testText.trim == decoded then
    IO.println "✓ Roundtrip successful!"
  else
    IO.println "✗ Roundtrip mismatch"
    IO.println s!"  Expected: '{testText.trim}'"
    IO.println s!"  Got:      '{decoded}'"

  IO.println "─────────────────────────────────────────"
  IO.println ""
  IO.println "Test completed!"
