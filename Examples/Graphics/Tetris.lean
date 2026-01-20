import Hesper
import Hesper.GLFW

/-!
# WebGPU Tetris

A Tetris implementation using GLFW and WebGPU.

Controls:
- A/D: Move left/right
- S: Soft drop
- W: Rotate counter-clockwise
- Space: Rotate clockwise / hard drop
- ESC: Exit
-/

namespace Examples.Tetris

open Hesper.GLFW

-- Constants
def windowWidth : Nat := 540
def windowHeight : Nat := 800
def boardWidth : Nat := 10
def boardHeight : Nat := 20

-- Types
abbrev Point := Int × Int
abbrev Color := Float × Float × Float

inductive TetrominoKind where
  | I | O | T | S | Z | J | L
  deriving Repr, BEq, Inhabited

structure Tetromino where
  kind : TetrominoKind
  rotation : Nat
  position : Point
  deriving Repr, Inhabited

-- Game state
structure GameState where
  board : List (List (Option TetrominoKind))  -- height × width
  currentPiece : Tetromino
  dropTimer : Float
  gameOver : Bool
  deriving Inhabited

-- Helper functions
def emptyRow : List (Option TetrominoKind) :=
  List.replicate boardWidth none

def emptyBoard : List (List (Option TetrominoKind)) :=
  List.replicate boardHeight emptyRow

def pieceColor (kind : TetrominoKind) : Color :=
  match kind with
  | .I => (0.0, 0.8, 0.9)
  | .O => (0.95, 0.85, 0.15)
  | .T => (0.75, 0.25, 0.85)
  | .S => (0.2, 0.8, 0.2)
  | .Z => (0.9, 0.2, 0.2)
  | .J => (0.2, 0.35, 0.95)
  | .L => (0.95, 0.55, 0.2)

def baseShape (kind : TetrominoKind) : List Point :=
  match kind with
  | .I => [(0, 1), (1, 1), (2, 1), (3, 1)]
  | .O => [(1, 1), (1, 2), (2, 1), (2, 2)]
  | .T => [(1, 0), (0, 1), (1, 1), (2, 1)]
  | .S => [(1, 1), (2, 1), (0, 2), (1, 2)]
  | .Z => [(0, 1), (1, 1), (1, 2), (2, 2)]
  | .J => [(0, 0), (0, 1), (1, 1), (2, 1)]
  | .L => [(2, 0), (0, 1), (1, 1), (2, 1)]

def rotatePoint (p : Point) : Point :=
  (3 - p.2, p.1)

def rotateShapeTimes (shape : List Point) : Nat → List Point
  | 0 => shape
  | n + 1 => rotateShapeTimes (shape.map rotatePoint) n

def tetrominoBlocks (piece : Tetromino) : List Point :=
  let shape := rotateShapeTimes (baseShape piece.kind) (piece.rotation % 4)
  shape.map fun p => (piece.position.1 + p.1, piece.position.2 + p.2)

-- Collision detection
def cellFree (board : List (List (Option TetrominoKind))) (p : Point) : Bool :=
  if p.2 < 0 then true
  else if p.2 >= boardHeight then false
  else if p.1 < 0 || p.1 >= boardWidth then false
  else
    match board[p.2.toNat]? with
    | none => false
    | some row =>
      match row[p.1.toNat]? with
      | none => false
      | some cell => cell.isNone

def validPosition (board : List (List (Option TetrominoKind))) (piece : Tetromino) : Bool :=
  (tetrominoBlocks piece).all (cellFree board)

-- Movement
def tryMove (board : List (List (Option TetrominoKind))) (delta : Point) (piece : Tetromino) : Option Tetromino :=
  let moved := { piece with position := (piece.position.1 + delta.1, piece.position.2 + delta.2) }
  if validPosition board moved then some moved else none

-- Rotation
def tryRotate (board : List (List (Option TetrominoKind))) (piece : Tetromino) (delta : Nat) : Option Tetromino :=
  let rotated := { piece with rotation := (piece.rotation + delta) % 4 }
  -- Try wall kicks
  let kicks : List Point := [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)]
  kicks.findSome? fun kick =>
    let candidate := { rotated with position := (rotated.position.1 + kick.1, rotated.position.2 + kick.2) }
    if validPosition board candidate then some candidate else none

-- Place piece on board
partial def placePiece (board : List (List (Option TetrominoKind))) (piece : Tetromino) : List (List (Option TetrominoKind)) :=
  let blocks := tetrominoBlocks piece
  blocks.foldl (fun acc p =>
    if p.2 < 0 || p.2 >= (boardHeight : Int) then acc
    else if p.1 < 0 || p.1 >= (boardWidth : Int) then acc
    else
      let yIdx := p.2.toNat
      let xIdx := p.1.toNat
      acc.mapIdx fun idx row =>
        if idx == yIdx then
          row.set xIdx (some piece.kind)
        else
          row
  ) board

-- Line clearing
def rowFull (row : List (Option TetrominoKind)) : Bool :=
  row.all Option.isSome

def clearLines (board : List (List (Option TetrominoKind))) : List (List (Option TetrominoKind)) × Nat :=
  let remaining := board.filter (fun row => !rowFull row)
  let cleared := board.length - remaining.length
  let newRows := List.replicate cleared emptyRow
  (newRows ++ remaining, cleared)

-- Game logic
def spawnPosition : Point := (boardWidth / 2 - 2, -1)

def dropInterval : Float := 0.5

partial def initialPiece : Tetromino :=
  { kind := TetrominoKind.I, rotation := 0, position := spawnPosition }

partial def lockPiece (state : GameState) : GameState :=
  let boardWithPiece := placePiece state.board state.currentPiece
  let (cleansed, _cleared) := clearLines boardWithPiece
  let newPiece := initialPiece  -- Simplified: always spawn I piece
  let gameOverNow := !validPosition cleansed newPiece
  { state with
    board := cleansed
    currentPiece := newPiece
    dropTimer := 0
    gameOver := gameOverNow }

partial def dropStep (state : GameState) : GameState :=
  match tryMove state.board (0, 1) state.currentPiece with
  | some moved => { state with currentPiece := moved }
  | none => lockPiece state

def moveHorizontal (dx : Int) (state : GameState) : GameState :=
  match tryMove state.board (dx, 0) state.currentPiece with
  | some moved => { state with currentPiece := moved }
  | none => state

def rotatePiece (state : GameState) : GameState :=
  match tryRotate state.board state.currentPiece 1 with
  | some rotated => { state with currentPiece := rotated }
  | none => state

-- Update with timer
partial def updateGame (dt : Float) (leftPressed rightPressed downPressed rotatePressed : Bool) (state : GameState) : GameState :=
  if state.gameOver then state
  else
    let s1 := if leftPressed then moveHorizontal (-1) state else state
    let s2 := if rightPressed then moveHorizontal 1 s1 else s1
    let s3 := if rotatePressed then rotatePiece s2 else s2
    let s4 := if downPressed then dropStep s3 else s3
    -- Auto-drop with timer
    let newTimer := s4.dropTimer + dt
    if newTimer >= dropInterval then
      { (dropStep s4) with dropTimer := 0 }
    else
      { s4 with dropTimer := newTimer }

-- Rendering
structure Square where
  x : Nat
  y : Nat
  color : Color

def cellWidth : Float := 2.0 / boardWidth.toFloat
def cellHeight : Float := 2.0 / boardHeight.toFloat

partial def settledSquares (board : List (List (Option TetrominoKind))) : List Square :=
  (board.mapIdx fun y row =>
    row.mapIdx fun x cell =>
      cell.map (fun kind => Square.mk x y (pieceColor kind))
  ).flatten.filterMap id

partial def activeSquares (piece : Tetromino) : List Square :=
  let color := pieceColor piece.kind
  (tetrominoBlocks piece).filterMap fun p =>
    let (px, py) := p
    if py < 0 then none
    else if py >= (boardHeight : Int) then none
    else if px < 0 then none
    else if px >= (boardWidth : Int) then none
    else some (Square.mk px.toNat py.toNat color)

structure VertexData where
  pos : Float × Float
  color : Color

def squareToVertices (sq : Square) : List VertexData :=
  let x0 := -1.0 + sq.x.toFloat * cellWidth
  let x1 := x0 + cellWidth
  let y0 := 1.0 - sq.y.toFloat * cellHeight
  let y1 := y0 - cellHeight
  let (r, g, b) := sq.color
  let v1 := VertexData.mk (x0, y0) (r, g, b)
  let v2 := VertexData.mk (x0, y1) (r, g, b)
  let v3 := VertexData.mk (x1, y1) (r, g, b)
  let v4 := VertexData.mk (x0, y0) (r, g, b)
  let v5 := VertexData.mk (x1, y1) (r, g, b)
  let v6 := VertexData.mk (x1, y0) (r, g, b)
  v1 :: v2 :: v3 :: v4 :: v5 :: v6 :: []

def squaresToVertices (squares : List Square) : List VertexData :=
  squares.flatMap squareToVertices

-- WGSL shader generation
def formatFloat (f : Float) : String :=
  let s := toString f
  if s.contains '.' then s else s ++ ".0"

def formatVec2 (p : Float × Float) : String :=
  "vec2f(" ++ formatFloat p.1 ++ ", " ++ formatFloat p.2 ++ ")"

def formatVec3 (c : Color) : String :=
  "vec3f(" ++ formatFloat c.1 ++ ", " ++ formatFloat c.2.1 ++ ", " ++ formatFloat c.2.2 ++ ")"

def shaderFromVertices (verts : List VertexData) : String :=
  let vertexCount := verts.length
  let positions := String.intercalate ",\n    " (verts.map (formatVec2 ∘ VertexData.pos))
  let colors := String.intercalate ",\n    " (verts.map (formatVec3 ∘ VertexData.color))
  "struct VertexOutput {\n" ++
  "  @builtin(position) position : vec4f,\n" ++
  "  @location(0) color : vec3f,\n" ++
  "};\n\n" ++
  "const POSITIONS : array<vec2f," ++ toString vertexCount ++ "> = array<vec2f," ++ toString vertexCount ++ ">(\n" ++
  "    " ++ positions ++ "\n" ++
  ");\n\n" ++
  "const COLORS : array<vec3f," ++ toString vertexCount ++ "> = array<vec3f," ++ toString vertexCount ++ ">(\n" ++
  "    " ++ colors ++ "\n" ++
  ");\n\n" ++
  "@vertex\n" ++
  "fn vertexMain(@builtin(vertex_index) idx : u32) -> VertexOutput {\n" ++
  "  var out : VertexOutput;\n" ++
  "  out.position = vec4f(POSITIONS[idx], 0.0, 1.0);\n" ++
  "  out.color = COLORS[idx];\n" ++
  "  return out;\n" ++
  "}\n\n" ++
  "@fragment\n" ++
  "fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {\n" ++
  "  return vec4f(input.color, 1.0);\n" ++
  "}"

-- Custom render frame with dynamic vertices
def renderGameFrame (device : Hesper.WebGPU.Device) (surface : Surface) (format : Nat) (state : GameState) : IO Unit := do
  let squares := settledSquares state.board ++ activeSquares state.currentPiece
  let vertices := squaresToVertices squares

  if vertices.isEmpty then
    -- Just clear the screen
    let texture ← getCurrentTexture surface
    let view ← createTextureView texture
    let encoder ← createCommandEncoder device
    let pass ← beginRenderPass encoder view
    endRenderPass pass
    let cmd ← finishEncoder encoder
    submit device cmd
    present surface
  else
    -- Render with dynamic shader
    let shaderCode := shaderFromVertices vertices
    let texture ← getCurrentTexture surface
    let view ← createTextureView texture
    let shader ← createShaderModule device shaderCode
    let pipeline ← createRenderPipeline device shader format
    let encoder ← createCommandEncoder device
    let pass ← beginRenderPass encoder view
    setPipeline pass pipeline
    drawVertices pass vertices.length
    endRenderPass pass
    let cmd ← finishEncoder encoder
    submit device cmd
    present surface

-- Game loop
partial def gameLoop (device : Hesper.WebGPU.Device) (window : Window) (surface : Surface)
    (format : Nat) (state : GameState) (prevLeft prevRight prevDown prevRotate : Bool) : IO Unit := do
  pollEvents
  let shouldClose ← windowShouldClose window
  unless shouldClose do
    -- Read input
    let escKey ← getKey window Key.escape
    if escKey == KeyAction.press then
      return ()

    let leftKey ← getKey window Key.a
    let rightKey ← getKey window Key.d
    let downKey ← getKey window Key.s
    let rotateKey ← getKey window Key.space

    let leftNow := leftKey == KeyAction.press || leftKey == KeyAction.repeated
    let rightNow := rightKey == KeyAction.press || rightKey == KeyAction.repeated
    let downNow := downKey == KeyAction.press || downKey == KeyAction.repeated
    let rotateNow := rotateKey == KeyAction.press

    let leftPressed := leftNow && !prevLeft
    let rightPressed := rightNow && !prevRight
    let downPressed := downNow && !prevDown
    let rotatePressed := rotateNow && !prevRotate

    -- Update game (assuming 60 FPS = ~0.016s per frame)
    let nextState := updateGame 0.016 leftPressed rightPressed downPressed rotatePressed state

    -- Render
    renderGameFrame device surface format nextState

    -- Check game over
    if nextState.gameOver && !state.gameOver then
      IO.println "Game Over! Press ESC to exit."

    -- Continue
    gameLoop device window surface format nextState leftNow rightNow downNow rotateNow

def initialGameState : GameState :=
  { board := emptyBoard
    currentPiece := initialPiece
    dropTimer := 0
    gameOver := false }

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║         WebGPU Tetris (Lean 4)               ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""
  IO.println "Controls:"
  IO.println "  A/D - Move left/right"
  IO.println "  S - Soft drop"
  IO.println "  Space - Rotate"
  IO.println "  ESC - Exit"
  IO.println ""

  Hesper.init

  withGLFW do
    IO.println "✓ GLFW initialized"

    let window ← createWindow windowWidth windowHeight "Hesper Tetris"
    IO.println "✓ Window created"

    let device ← Hesper.WebGPU.getDevice
    IO.println "✓ Device obtained"

    let surface ← createSurface device window
    IO.println "✓ Surface created"

    let format ← getSurfacePreferredFormat surface
    configureSurface surface windowWidth windowHeight format
    IO.println s!"✓ Surface configured (format: {format})"

    IO.println ""
    IO.println "Starting game..."
    IO.println ""

    gameLoop device window surface format initialGameState false false false false

    IO.println "Game loop returned"

  IO.println ""
  IO.println "✓ Thanks for playing!"

end Examples.Tetris

def main : IO Unit := Examples.Tetris.main
