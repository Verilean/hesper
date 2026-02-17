import Hesper.GGUF.Types
import Hesper.GGUF.Parser

/-!
# SentencePiece Tokenizer

Implements SentencePiece tokenization for BitNet models.

## SentencePiece Algorithm

SentencePiece uses **subword tokenization** with unigram language model:

1. **Encoding** (Text → Tokens):
   ```
   Text: "Hello world"
   ↓ Normalize: "▁Hello▁world" (▁ = space marker)
   ↓ Segment: ["▁Hello", "▁world"]
   ↓ Lookup: [1234, 5678]
   ```

2. **Decoding** (Tokens → Text):
   ```
   Tokens: [1234, 5678]
   ↓ Lookup: ["▁Hello", "▁world"]
   ↓ Join: "▁Hello▁world"
   ↓ Replace ▁: "Hello world"
   ```

## Vocabulary Structure

From GGUF metadata:
```lean
tokenizer.ggml.tokens: Array String  -- ["<unk>", "<s>", "</s>", "▁the", ...]
tokenizer.ggml.scores: Array Float   -- [-inf, -inf, -inf, -2.5, ...]
tokenizer.ggml.token_type: Array UInt32  -- [2, 2, 2, 1, ...]
```

**Token Types**:
- 0: Normal
- 1: Unknown
- 2: Control (BOS, EOS, etc.)
- 3: User-defined
- 4: Unused
- 5: Byte

## References
- SentencePiece: https://github.com/google/sentencepiece
- GGUF Tokenizer: https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py
-/

namespace Hesper.Tokenizer.SentencePiece

open Hesper.GGUF

/-! ## Token Types -/

/-- SentencePiece token type -/
inductive TokenType where
  | Normal      -- Regular token
  | Unknown     -- Unknown token (fallback)
  | Control     -- Special tokens (BOS, EOS, PAD)
  | UserDefined -- User-defined tokens
  | Unused      -- Unused
  | Byte        -- Byte fallback
  deriving Repr, BEq

def TokenType.fromNat (n : Nat) : TokenType :=
  match n with
  | 0 => .Normal
  | 1 => .Unknown
  | 2 => .Control
  | 3 => .UserDefined
  | 4 => .Unused
  | 5 => .Byte
  | _ => .Normal

/-! ## Vocabulary -/

/-- Token information -/
structure TokenInfo where
  id : Nat
  piece : String       -- Token string (may contain ▁ for space)
  score : Float        -- Log probability
  tokenType : TokenType
  deriving Repr

instance : Inhabited TokenInfo where
  default := {
    id := 0
    piece := ""
    score := 0.0
    tokenType := .Normal
  }

/-- Vocabulary with token mappings -/
structure Vocab where
  tokens : Array TokenInfo
  pieceToId : List (String × Nat)  -- For fast lookup (simplified, would use HashMap)
  vocabSize : Nat
  bosToken : Option Nat  -- Begin of sequence
  eosToken : Option Nat  -- End of sequence
  unkToken : Option Nat  -- Unknown token
  padToken : Option Nat  -- Padding token
  usesBPE : Bool := false  -- BPE uses Ġ (U+0120), SentencePiece uses ▁ (U+2581)
  deriving Repr

/-! ## Tokenizer Structure -/

/-- SentencePiece tokenizer -/
structure Tokenizer where
  vocab : Vocab
  addBos : Bool := true        -- Add BOS token
  addEos : Bool := false       -- Add EOS token
  deriving Repr

/-! ## Vocabulary Construction -/

/-- Parse string array from GGUF metadata array -/
def parseStringArray (data : ByteArray) (_offset : Nat := 0) : IO (Array String) := do
  -- The Parser has already consumed the array type and count
  -- data contains just the string elements, each with 8-byte length prefix

  let mut result := #[]
  let mut currentOffset := 0

  -- Parse until we run out of data
  while currentOffset + 8 <= data.size do
    -- Read string length (8 bytes, UInt64 little-endian)
    let b0 := data.get! currentOffset |>.toNat
    let b1 := data.get! (currentOffset + 1) |>.toNat
    let b2 := data.get! (currentOffset + 2) |>.toNat
    let b3 := data.get! (currentOffset + 3) |>.toNat
    let strLen := b0 + b1 * 256 + b2 * 65536 + b3 * 16777216

    currentOffset := currentOffset + 8

    if strLen > 0 && currentOffset + strLen <= data.size then
      let strBytes := data.extract currentOffset (currentOffset + strLen)
      match String.fromUTF8? strBytes with
      | some str => result := result.push str
      | none => pure ()  -- Skip invalid UTF-8
      currentOffset := currentOffset + strLen
    else
      break  -- Invalid length or not enough data

  return result

/-- Parse Float32 array from GGUF metadata array -/
partial def parseFloatArray (data : ByteArray) : Array Float :=
  -- The Parser has already consumed type and count, data contains just Float32 elements
  let rec parseElements (offset : Nat) (result : Array Float) : Array Float :=
    if offset + 4 > data.size then
      result
    else
      -- Read Float32 (little endian)
      let b0 := data.get! offset |>.toUInt32
      let b1 := data.get! (offset + 1) |>.toUInt32
      let b2 := data.get! (offset + 2) |>.toUInt32
      let b3 := data.get! (offset + 3) |>.toUInt32
      let bits := (b0 + b1 * 256 + b2 * 65536 + b3 * 16777216).toUInt64
      parseElements (offset + 4) (result.push (Float.ofBits bits))

  parseElements 0 #[]

/-- Parse UInt32 array from GGUF metadata array -/
partial def parseUInt32Array (data : ByteArray) : Array UInt32 :=
  -- The Parser has already consumed type and count, data contains just UInt32 elements
  let rec parseElements (offset : Nat) (result : Array UInt32) : Array UInt32 :=
    if offset + 4 > data.size then
      result
    else
      let b0 := data.get! offset |>.toUInt32
      let b1 := data.get! (offset + 1) |>.toUInt32
      let b2 := data.get! (offset + 2) |>.toUInt32
      let b3 := data.get! (offset + 3) |>.toUInt32
      let val := b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
      parseElements (offset + 4) (result.push val)

  parseElements 0 #[]

/-- Build piece-to-id map for fast lookup -/
def buildPieceToIdMap (tokens : Array TokenInfo) : List (String × Nat) :=
  tokens.toList.map (fun t => (t.piece, t.id))

/-- Find special token by type and common names -/
def findSpecialToken (tokens : Array TokenInfo) (names : List String) : Option Nat :=
  tokens.findIdx? (fun t => names.contains t.piece)

/-- Create vocabulary from GGUF metadata -/
def createVocabFromGGUF (gguf : GGUFFile) : IO Vocab := do
  IO.println "[Tokenizer] Creating vocabulary from GGUF..."

  -- Extract token strings
  let tokenStrings ← match gguf.metadata.find? (·.1 == "tokenizer.ggml.tokens") with
    | some (_, mv) =>
      if mv.valueType == MetadataValueType.MArray then
        parseStringArray mv.data
      else
        throw $ IO.userError "tokenizer.ggml.tokens is not an array"
    | _ => throw $ IO.userError "tokenizer.ggml.tokens not found"

  if tokenStrings.isEmpty then
    throw $ IO.userError "No tokenizer.ggml.tokens found in GGUF metadata"

  -- Extract scores (optional)
  let scores := match gguf.metadata.find? (·.1 == "tokenizer.ggml.scores") with
    | some (_, mv) =>
      if mv.valueType == MetadataValueType.MArray then
        parseFloatArray mv.data
      else
        Array.mk (List.replicate tokenStrings.size 0.0)
    | _ => Array.mk (List.replicate tokenStrings.size 0.0)

  -- Extract token types (optional)
  let tokenTypes := match gguf.metadata.find? (·.1 == "tokenizer.ggml.token_type") with
    | some (_, mv) =>
      if mv.valueType == MetadataValueType.MArray then
        parseUInt32Array mv.data
      else
        Array.mk (List.replicate tokenStrings.size 0)
    | _ => Array.mk (List.replicate tokenStrings.size 0)

  -- Build token info array
  let tokens := List.range tokenStrings.size |>.toArray.map (fun idx =>
    { id := idx
      piece := tokenStrings[idx]!
      score := if idx < scores.size then scores[idx]! else 0.0
      tokenType := TokenType.fromNat (if idx < tokenTypes.size then tokenTypes[idx]!.toNat else 0)
    })

  -- Build lookup map
  let pieceToId := buildPieceToIdMap tokens

  -- Find special tokens: prefer explicit GGUF metadata IDs, fall back to name search
  let readTokenId (key : String) : Option Nat :=
    match gguf.metadata.find? (·.1 == key) with
    | some (_, mv) =>
      if mv.data.size >= 4 then
        let v := mv.data[0]!.toNat ||| (mv.data[1]!.toNat <<< 8) |||
                 (mv.data[2]!.toNat <<< 16) ||| (mv.data[3]!.toNat <<< 24)
        some v
      else none
    | none => none
  let bosToken := readTokenId "tokenizer.ggml.bos_token_id" |>.orElse
    (fun _ => findSpecialToken tokens ["<s>", "<bos>", "[BOS]", "<|begin_of_text|>"])
  let eosToken := readTokenId "tokenizer.ggml.eos_token_id" |>.orElse
    (fun _ => findSpecialToken tokens ["</s>", "<eos>", "[EOS]", "<|end_of_text|>"])
  let unkToken := findSpecialToken tokens ["<unk>", "[UNK]"]
  let padToken := findSpecialToken tokens ["<pad>", "[PAD]"]

  IO.println s!"  Vocabulary size: {tokens.size}"
  IO.println s!"  BOS token: {bosToken}"
  IO.println s!"  EOS token: {eosToken}"
  IO.println s!"  UNK token: {unkToken}"

  -- Detect BPE vs SentencePiece space encoding
  let usesBPE := tokens.any (fun t => t.piece.startsWith "Ġ")
  if usesBPE then
    IO.println "  Space encoding: BPE (Ġ)"
  else
    IO.println "  Space encoding: SentencePiece (▁)"

  return {
    tokens := tokens
    pieceToId := pieceToId
    vocabSize := tokens.size
    bosToken := bosToken
    eosToken := eosToken
    unkToken := unkToken
    padToken := padToken
    usesBPE := usesBPE
  }

/-! ## Tokenization (Encoding) -/

/-- Normalize text: add space marker, handle special cases -/
def normalizeText (vocab : Vocab) (text : String) : String :=
  -- Add space marker at beginning if not present
  let text := if text.startsWith " " then text else " " ++ text
  -- BPE uses Ġ (U+0120), SentencePiece uses ▁ (U+2581)
  let spaceMarker := if vocab.usesBPE then "Ġ" else "▁"
  text.replace " " spaceMarker

/-- Find longest matching token piece starting at position -/
partial def findLongestMatch (vocab : Vocab) (text : String) (startPos : Nat) : Option (Nat × Nat) :=
  -- Try from longest to shortest
  let remaining := text.drop startPos
  let maxLen := min remaining.length 20

  let rec tryLength (len : Nat) : Option (Nat × Nat) :=
    if len == 0 then
      none
    else
      let piece := remaining.take len
      match vocab.pieceToId.find? (·.1 == piece) with
      | some (_, id) => some (id, len)
      | none => tryLength (len - 1)

  tryLength maxLen

/-- Greedy tokenization (simplified, real SentencePiece uses Viterbi) -/
partial def tokenizeGreedy (vocab : Vocab) (text : String) : Array Nat :=
  let normalized := normalizeText vocab text
  let textLen := normalized.length

  let rec loop (pos : Nat) (acc : Array Nat) : Array Nat :=
    if pos >= textLen then
      acc
    else
      match findLongestMatch vocab normalized pos with
      | some (tokenId, len) =>
        loop (pos + len) (acc.push tokenId)
      | none =>
        -- Fallback: use UNK token or skip character
        match vocab.unkToken with
        | some unkId => loop (pos + 1) (acc.push unkId)
        | none => loop (pos + 1) acc

  loop 0 #[]

/-- Encode text to token IDs -/
def encode (tokenizer : Tokenizer) (text : String) : Array Nat :=
  let tokens := tokenizeGreedy tokenizer.vocab text

  -- Add BOS token
  let tokens := if tokenizer.addBos then
    match tokenizer.vocab.bosToken with
    | some bosId => #[bosId] ++ tokens
    | none => tokens
  else
    tokens

  -- Add EOS token
  let tokens := if tokenizer.addEos then
    match tokenizer.vocab.eosToken with
    | some eosId => tokens.push eosId
    | none => tokens
  else
    tokens

  tokens

/-! ## Detokenization (Decoding) -/

/-- Decode single token to string -/
def decodeToken (vocab : Vocab) (tokenId : Nat) : String :=
  if tokenId < vocab.tokens.size then
    vocab.tokens[tokenId]!.piece
  else
    "<unk>"

/-- Decode token IDs to text -/
def decode (tokenizer : Tokenizer) (tokens : Array Nat) : String :=
  let pieces := tokens.filterMap (fun tokenId =>
    -- Skip special tokens
    let isSpecial :=
      (some tokenId == tokenizer.vocab.bosToken) ||
      (some tokenId == tokenizer.vocab.eosToken) ||
      (some tokenId == tokenizer.vocab.padToken)

    if !isSpecial then
      some (decodeToken tokenizer.vocab tokenId)
    else
      none
  )

  -- Join and convert BPE byte markers back to characters
  -- GPT-2/Llama 3 BPE uses Unicode chars for non-printable bytes:
  --   Ġ (U+0120) = space, Ċ (U+010A) = newline, ĉ (U+0109) = tab
  let text := String.join pieces.toList
  let text := text.replace "▁" " "   -- SentencePiece space marker
  let text := text.replace "Ġ" " "   -- BPE space byte
  let text := text.replace "Ċ" "\n"  -- BPE newline byte
  let text := text.replace "ĉ" "\t"  -- BPE tab byte
  text.trim

/-! ## Tokenizer Creation -/

/-- Create tokenizer from GGUF file -/
def fromGGUF (gguf : GGUFFile) (addBos : Bool := true) (addEos : Bool := false)
    : IO Tokenizer := do
  let vocab ← createVocabFromGGUF gguf

  IO.println "[Tokenizer] Tokenizer created successfully"
  return {
    vocab := vocab
    addBos := addBos
    addEos := addEos
  }

/-! ## Utilities -/

/-- Get vocabulary size -/
def vocabSize (tokenizer : Tokenizer) : Nat :=
  tokenizer.vocab.vocabSize

/-- Get special token IDs -/
def bosToken (tokenizer : Tokenizer) : Option Nat :=
  tokenizer.vocab.bosToken

def eosToken (tokenizer : Tokenizer) : Option Nat :=
  tokenizer.vocab.eosToken

def unkToken (tokenizer : Tokenizer) : Option Nat :=
  tokenizer.vocab.unkToken

/-- Print tokenizer info -/
def printInfo (tokenizer : Tokenizer) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  SentencePiece Tokenizer Info"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Vocabulary size: {tokenizer.vocab.vocabSize}"
  IO.println s!"BOS token: {tokenizer.vocab.bosToken}"
  IO.println s!"EOS token: {tokenizer.vocab.eosToken}"
  IO.println s!"UNK token: {tokenizer.vocab.unkToken}"
  IO.println s!"Add BOS: {tokenizer.addBos}"
  IO.println s!"Add EOS: {tokenizer.addEos}"
  IO.println "═══════════════════════════════════════════════"

/-- Test tokenization roundtrip -/
def testRoundtrip (tokenizer : Tokenizer) (text : String) : IO Unit := do
  IO.println s!"Original: '{text}'"
  let tokens := encode tokenizer text
  IO.println s!"Tokens: {tokens}"
  let decoded := decode tokenizer tokens
  IO.println s!"Decoded: '{decoded}'"

  if text.trim == decoded then
    IO.println "✓ Roundtrip successful!"
  else
    IO.println "✗ Roundtrip mismatch"

end Hesper.Tokenizer.SentencePiece
