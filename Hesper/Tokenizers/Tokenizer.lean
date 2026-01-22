/-!
# Tokenizer Interface
-/

namespace Hesper.Tokenizers

structure Tokenizer where
  encode : String → List Nat
  decode : List Nat → String
  vocabSize : Nat

namespace Tokenizer

  def dummy : Tokenizer := {
    encode := fun _ => [],
    decode := fun _ => "",
    vocabSize := 0
  }

end Tokenizer

end Hesper.Tokenizers
