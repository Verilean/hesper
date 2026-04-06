import Hesper.AD.Verified
open Hesper.AD.Verified

def main : IO Unit := do
  IO.println "=== Testing that wrong backwards are detected ==="
  IO.println ""

  -- Correct softmax backward
  let correctOp := softmaxOp
  let (p1, e1) := verifyOp correctOp
  IO.println s!"Correct backward:  {if p1 then "PASS" else "FAIL"} (err={e1})"

  -- WRONG backward: return zeros
  let wrongOp1 : DiffOp := { softmaxOp with
    backward := fun _ _ => #[0.0, 0.0, 0.0, 0.0]
  }
  let (p2, e2) := verifyOp wrongOp1
  IO.println s!"Zero backward:     {if p2 then "PASS" else "FAIL"} (err={e2})"

  -- WRONG backward: return dy unchanged (identity)
  let wrongOp2 : DiffOp := { softmaxOp with
    backward := fun _ dy => dy
  }
  let (p3, e3) := verifyOp wrongOp2
  IO.println s!"Identity backward: {if p3 then "PASS" else "FAIL"} (err={e3})"

  -- WRONG backward: negate the correct answer
  let wrongOp3 : DiffOp := { softmaxOp with
    backward := fun x dy => (softmaxBwd x dy).map (· * (-1.0))
  }
  let (p4, e4) := verifyOp wrongOp3
  IO.println s!"Negated backward:  {if p4 then "PASS" else "FAIL"} (err={e4})"

  -- WRONG RoPE: forget to negate sin
  let wrongRope : DiffOp := { (ropeOp 0.7) with
    backward := fun _ dy =>
      let dy0 := dy.getD 0 0.0
      let dy1 := dy.getD 1 0.0
      -- BUG: should be +sin for dy1, -sin for dy0 component
      #[dy0 * Float.cos 0.7 - dy1 * Float.sin 0.7,   -- wrong sign!
        dy0 * Float.sin 0.7 + dy1 * Float.cos 0.7]
  }
  let (p5, e5) := verifyOp wrongRope
  IO.println s!"Wrong RoPE sign:   {if p5 then "PASS" else "FAIL"} (err={e5})"

  IO.println ""
  IO.println "Expected: Correct=PASS, all others=FAIL"
  if p1 && !p2 && !p3 && !p4 && !p5 then
    IO.println "✓ Checker correctly detects wrong backwards!"
  else
    IO.println "✗ Checker may have bugs — investigate!"
