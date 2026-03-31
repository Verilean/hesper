import Lean.Data.Json

/-!
# Alpaca Dataset Loader

Parses Stanford Alpaca-format JSON datasets for instruction finetuning.

## Format

```json
[
  {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet..."
  },
  ...
]
```

## Prompt Template

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The model is trained with teacher forcing: the loss is computed only on
the output tokens (after "### Response:\n").
-/

namespace Hesper.Training.AlpacaDataset

/-- A single Alpaca training example -/
structure Example where
  instruction : String
  input : String     -- can be empty
  output : String
  deriving Repr

/-- A tokenized training example ready for the model -/
structure TokenizedExample where
  /-- Full token sequence (prompt + output + EOS) -/
  tokens : Array Nat
  /-- Index where the output starts (loss computed from here) -/
  promptLen : Nat
  /-- Total sequence length -/
  seqLen : Nat
  deriving Repr

/-- Format an Alpaca example into the standard prompt template -/
def formatPrompt (ex : Example) : String :=
  let base := "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" ++ ex.instruction
  let withInput := if ex.input.isEmpty then base
    else base ++ "\n\n### Input:\n" ++ ex.input
  withInput ++ "\n\n### Response:\n"

/-- Format the full sequence (prompt + output) for training -/
def formatFullSequence (ex : Example) : String :=
  formatPrompt ex ++ ex.output

/-- Parse a single JSON object into an Alpaca Example -/
def parseExample (json : Lean.Json) : Except String Example := do
  let instruction ← json.getObjValAs? String "instruction"
  let input ← match json.getObjValAs? String "input" with
    | .ok s => pure s
    | .error _ => pure ""
  let output ← json.getObjValAs? String "output"
  pure { instruction, input, output }

/-- Load an Alpaca dataset from a JSON file.
    The file should contain a JSON array of objects. -/
def loadDataset (path : String) : IO (Array Example) := do
  let contents ← IO.FS.readFile path
  match Lean.Json.parse contents with
  | .error msg => throw (IO.userError s!"Failed to parse JSON: {msg}")
  | .ok json =>
    match json.getArr? with
    | .error msg => throw (IO.userError s!"Expected JSON array: {msg}")
    | .ok arr =>
      let mut examples := #[]
      for item in arr do
        match parseExample item with
        | .ok ex => examples := examples.push ex
        | .error msg =>
          IO.eprintln s!"[AlpacaDataset] Skipping malformed example: {msg}"
      IO.println s!"[AlpacaDataset] Loaded {examples.size} examples from {path}"
      pure examples

/-- Tokenize an Alpaca example using the provided encode function.

    @param encode Tokenizer encode function (String → Array Nat)
    @param example The Alpaca example
    @param eosToken End-of-sequence token ID
    @param maxSeqLen Maximum sequence length (truncate if longer)
    @return TokenizedExample with prompt boundary marked -/
def tokenizeExample (encode : String → Array Nat) (ex : Example)
    (eosToken : Nat) (maxSeqLen : Nat := 512) : TokenizedExample :=
  let prompt := formatPrompt ex
  let promptTokens := encode prompt
  let outputTokens := encode ex.output
  let fullTokens := promptTokens ++ outputTokens ++ #[eosToken]
  -- Truncate if needed
  let tokens := if fullTokens.size > maxSeqLen then
    fullTokens.extract 0 maxSeqLen
  else fullTokens
  { tokens, promptLen := promptTokens.size, seqLen := tokens.size }

/-- Tokenize an entire dataset -/
def tokenizeDataset (encode : String → Array Nat) (examples : Array Example)
    (eosToken : Nat) (maxSeqLen : Nat := 512) : Array TokenizedExample :=
  examples.map (tokenizeExample encode · eosToken maxSeqLen)

/-- Print dataset statistics -/
def printStats (examples : Array TokenizedExample) : IO Unit := do
  if examples.isEmpty then
    IO.println "[AlpacaDataset] No examples"
    return
  let totalTokens := examples.foldl (fun acc ex => acc + ex.seqLen) 0
  let totalOutputTokens := examples.foldl (fun acc ex => acc + (ex.seqLen - ex.promptLen)) 0
  let avgLen := totalTokens / examples.size
  let avgOutputLen := totalOutputTokens / examples.size
  IO.println s!"[AlpacaDataset] {examples.size} examples"
  IO.println s!"[AlpacaDataset] Avg sequence length: {avgLen} tokens"
  IO.println s!"[AlpacaDataset] Avg output length: {avgOutputLen} tokens"
  IO.println s!"[AlpacaDataset] Total training tokens: {totalOutputTokens}"

end Hesper.Training.AlpacaDataset
