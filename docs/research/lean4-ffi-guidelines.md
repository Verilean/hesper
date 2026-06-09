# Lean 4 FFI Guidelines for Hesper

C/C++ FFI ブリッジを書くときの注意点。
`native/bridge.cpp`（Dawn）と `native/cuda_bridge.cpp`（CUDA）の実装経験に基づく。

## 1. Box/Unbox は Lean 型に正確に対応させる

**これが最重要。間違えるとメモリ破壊 → segfault（module init 時に発生し、原因特定が極めて困難）。**

| Lean 型 | C → Lean（返り値） | Lean → C（引数） |
|---------|-------------------|-----------------|
| `Unit` | `lean_box(0)` | — |
| `Nat` | `lean_box(n)` | `lean_unbox(o)` |
| `UInt32` | `lean_box_uint32(v)` | `lean_unbox_uint32(o)` |
| **`USize`** | **`lean_box_usize(v)`** | **`lean_unbox_usize(o)`** |
| `UInt64` | `lean_box_uint64(v)` | `lean_unbox_uint64(o)` |
| `Float` | `lean_box_float(v)` | `lean_unbox_float(o)` |
| `String` | `lean_mk_string(s)` | `lean_string_cstr(o)` |
| `ByteArray` | `lean_alloc_sarray(...)` | `lean_sarray_cptr(o)` |

### 致命的な間違い

```cpp
// ❌ USize を lean_box で返す → Nat エンコーディングで返るが
//    Lean 側は lean_unbox_usize でデコードしようとしてメモリ破壊
return lean_io_result_mk_ok(lean_box((size_t)ptr));

// ✅ 正しい
return lean_io_result_mk_ok(lean_box_usize((size_t)ptr));
```

`lean_box` と `lean_box_usize` は**エンコーディングが異なる**。
`lean_box` は Nat 用（small nat optimization + mpz fallback）、
`lean_box_usize` は USize 用（常に `lean_object` に格納）。
混同すると、Lean の GC やパターンマッチが不正なメモリを辿ってクラッシュする。

### `Array USize` 内の要素

```cpp
// ❌
ptrs[i] = (CUdeviceptr)lean_unbox(lean_array_get_core(arr, i));

// ✅
ptrs[i] = (CUdeviceptr)lean_unbox_usize(lean_array_get_core(arr, i));
```

## 2. IO 関数は world token を受け取らない

Lean 4 の FFI では、`IO α` を返す `opaque` 関数の C 側実装に
**world token 引数は渡されない**。

```lean
-- Lean 側
@[extern "lean_hesper_cuda_init"]
opaque cuDriverInit : IO Unit
```

```cpp
// ❌ world token を受け取る（引数がずれてスタック破壊）
extern "C" lean_obj_res lean_hesper_cuda_init(lean_obj_arg world) { ... }

// ✅ 引数なし
extern "C" lean_obj_res lean_hesper_cuda_init() { ... }
```

Lean コンパイラが生成する C コードを見れば確認できる:
```bash
grep "lean_hesper_cuda_init" .lake/build/ir/Hesper/CUDA/FFI.c
# → lean_object* lean_hesper_cuda_init();  ← 引数なし
```

## 3. `@& T`（borrowed reference）は `b_lean_obj_arg`

```lean
opaque cuModuleLoadData (ptxSource : @& String) : IO CUmodule
```

```cpp
// @& → b_lean_obj_arg（参照カウントを減らさない）
extern "C" lean_obj_res lean_hesper_cuda_module_load_data(b_lean_obj_arg ptx_str) {
    const char* ptx = lean_string_cstr(ptx_str);
    // ptx_str の lean_dec は不要（borrowed）
}
```

## 4. エラーハンドリング

```cpp
// IO.userError を作って IO result に包む
return lean_io_result_mk_error(
    lean_mk_io_user_error(lean_mk_string("error message")));

// 成功
return lean_io_result_mk_ok(lean_box(0));  // Unit
return lean_io_result_mk_ok(lean_box_usize(ptr));  // USize
```

## 5. `extern "C"` を忘れない（C++ の場合）

C++ でコンパイルする場合、全 FFI 関数に `extern "C"` が必要。
なければ C++ name mangling でシンボル名が変わり、リンクエラーになる。

## 6. デバッグ方法

FFI の型不整合は module init 時の segfault として現れることが多い。
`main` の `IO.println` すら出力されない場合、FFI の型を疑う。

```bash
# Lean が生成した C コードでシグネチャを確認
grep "lean_hesper_cuda_" .lake/build/ir/Hesper/CUDA/FFI.c

# 引数の数・型が C++ 実装と一致しているか確認
```

## 7. Lean の abbrev 型と C 型の対応

```lean
abbrev CUdevice := UInt32    -- C: uint32_t, box: lean_box_uint32
abbrev CUcontext := USize    -- C: size_t,   box: lean_box_usize  ← 要注意
abbrev CUmodule := USize     -- C: size_t,   box: lean_box_usize
abbrev CUfunction := USize   -- C: size_t,   box: lean_box_usize
abbrev CUdeviceptr := USize  -- C: size_t,   box: lean_box_usize
```

`USize` を `abbrev` で定義したオペークハンドルは全て `lean_box_usize` で返す。
`lean_box` で返すと **確実にクラッシュする**。
