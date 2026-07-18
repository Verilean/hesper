// metal_backend.mm — M-Metal Stage 2: thin Metal executor (HESPER_BACKEND=metal).
//
// A minimal WebGPU-shaped compute backend that runs hesper's generated WGSL
// kernels directly on Metal, bypassing Dawn in the hot path:
//   WGSL --(pinned May tint CLI, robustness OFF)--> MSL --> MTLLibrary --> PSO
// with a persistent on-disk MSL cache keyed by WGSL content hash (this also
// deduplicates the 30 byte-identical per-layer kernels that Dawn compiled as
// 30 separate pipelines).
//
// Ordering model = Dawn parity: one MTLCommandBuffer per batch, ONE serial
// compute encoder per command buffer (serial dispatch type ≈ Dawn's implicit
// per-dispatch barriers), hazard-tracked resources. flushBatch = commit
// without wait; endBatch = commit + waitUntilCompleted.
//
// Buffers are MTLResourceStorageModeShared (unified memory) and explicitly
// zero-filled on creation — Dawn zero-initializes, Metal does not, and the
// uninitialized-read bug class is one we've already paid for once.
//
// Pure C ABI, no Lean dependency: bridge.cpp owns all lean_object wrapping and
// registers finalizers that call the hm_free_* functions below.

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <sys/stat.h>

namespace {

struct HMPipe {
    id<MTLComputePipelineState> pso;   // retained
    uint32_t tgBytes;                  // threadgroup memory at index 0 (0 = none)
    uint32_t wgX, wgY, wgZ;            // threads per threadgroup (from @workgroup_size)
    std::vector<uint8_t> writeMask;    // per BINDING index: 1 = kernel writes this buffer
};

struct HMCtx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> lastCB;       // retained; last committed batch (for read fences)
    std::unordered_map<uint64_t, HMPipe*> psoCache;  // wgsl-hash -> pipeline
    std::mutex mu;
    std::string tintPath;
    std::string cacheDir;
};

struct HMBuf {
    id<MTLBuffer> buf;                 // retained (new* = +1)
    size_t size;
};

struct HMShader {
    std::string wgsl;
    uint32_t tgBytes;
    uint32_t wgX, wgY, wgZ;
    std::vector<uint8_t> writeMask;    // per binding: 1 = written (declared rw AND stored-to)
};

struct HMBind {
    std::vector<std::pair<uint32_t, id<MTLBuffer>>> entries;  // (binding, buffer), buffers retained
};

struct HMEnc {
    id<MTLCommandBuffer> cb;           // retained
    id<MTLComputeCommandEncoder> enc;  // retained; nil after end
    bool concurrent;
    std::unordered_set<void*> readSet;   // buffers read since the last barrier
    std::unordered_set<void*> writeSet;  // buffers written since the last barrier
};

HMCtx* g_ctx = nullptr;
std::mutex g_ctx_mu;

uint64_t fnv64(const char* s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

// Parse "@workgroup_size(X[, Y[, Z]])" (integer literals — the DSL always
// emits literals).
void parseWorkgroupSize(const std::string& wgsl, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = 1; y = 1; z = 1;
    size_t p = wgsl.find("@workgroup_size(");
    if (p == std::string::npos) return;
    p += strlen("@workgroup_size(");
    uint32_t vals[3] = {1, 1, 1};
    int vi = 0;
    while (p < wgsl.size() && vi < 3) {
        while (p < wgsl.size() && (wgsl[p] == ' ' || wgsl[p] == ',')) p++;
        if (!isdigit((unsigned char)wgsl[p])) break;
        uint32_t v = 0;
        while (p < wgsl.size() && isdigit((unsigned char)wgsl[p])) { v = v * 10 + (wgsl[p] - '0'); p++; }
        vals[vi++] = v;
        while (p < wgsl.size() && wgsl[p] == 'u') p++;
        if (p < wgsl.size() && wgsl[p] == ')') break;
    }
    x = vals[0]; y = vals[1]; z = vals[2];
}

// Total threadgroup memory: sum var<workgroup> declarations. Tint packs them
// into a single [[threadgroup(0)]] struct; we compute a safe upper bound with
// per-member element alignment and 16-byte final rounding.
uint32_t parseWorkgroupBytes(const std::string& wgsl) {
    uint64_t total = 0;
    size_t p = 0;
    while ((p = wgsl.find("var<workgroup>", p)) != std::string::npos) {
        size_t line_end = wgsl.find(';', p);
        if (line_end == std::string::npos) break;
        std::string decl = wgsl.substr(p, line_end - p);
        p = line_end;
        uint64_t elem = 4, count = 1;
        size_t ap = decl.find("array<");
        if (ap != std::string::npos) {
            if (decl.find("f16", ap) != std::string::npos) elem = 2;
            else elem = 4;  // u32 / i32 / f32
            size_t comma = decl.find(',', ap);
            if (comma != std::string::npos) {
                count = strtoull(decl.c_str() + comma + 1, nullptr, 10);
            }
        } else {
            if (decl.find(": f16") != std::string::npos) elem = 2;
        }
        // align member offset to element size
        if (elem > 0 && total % elem) total += elem - (total % elem);
        total += elem * count;
    }
    return (uint32_t)((total + 15) & ~15ULL);
}

bool runTint(HMCtx* ctx, const std::string& wgsl, uint64_t h, std::string& mslOut, std::string& err) {
    char mslPath[1024], wgslPath[1024];
    snprintf(mslPath, sizeof(mslPath), "%s/%016llx.msl", ctx->cacheDir.c_str(), (unsigned long long)h);
    snprintf(wgslPath, sizeof(wgslPath), "%s/%016llx.wgsl", ctx->cacheDir.c_str(), (unsigned long long)h);

    // cache hit?
    if (FILE* f = fopen(mslPath, "rb")) {
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        mslOut.resize(n);
        size_t rd = fread(&mslOut[0], 1, n, f);
        fclose(f);
        if ((long)rd == n && n > 0) return true;
    }

    if (FILE* f = fopen(wgslPath, "wb")) {
        fwrite(wgsl.data(), 1, wgsl.size(), f);
        fclose(f);
    } else { err = "cannot write wgsl temp"; return false; }

    char cmd[2600];
    snprintf(cmd, sizeof(cmd),
        "'%s' --format msl --disable-robustness true --msl-version 3.2 '%s' > '%s.tmp' 2> '%s.err'",
        ctx->tintPath.c_str(), wgslPath, mslPath, mslPath);
    int rc = system(cmd);
    if (rc != 0) {
        char errPath[1060]; snprintf(errPath, sizeof(errPath), "%s.err", mslPath);
        if (FILE* f = fopen(errPath, "rb")) {
            char buf[512]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); buf[n] = 0; fclose(f);
            err = std::string("tint failed: ") + buf;
        } else err = "tint failed (no stderr)";
        return false;
    }
    char tmpPath[1060]; snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", mslPath);
    rename(tmpPath, mslPath);
    if (FILE* f = fopen(mslPath, "rb")) {
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        mslOut.resize(n);
        size_t rd = fread(&mslOut[0], 1, n, f);
        fclose(f);
        return (long)rd == n && n > 0;
    }
    err = "tint produced no output";
    return false;
}

// Per-binding write masks. Declared var<storage, read> is definitively read-only.
// Declared read_write is REFINED by scanning the body for actual store sites
// ("name[...] =", "subgroupMatrixStore(&name", "atomicXxx(&name") because the
// generated WMMA kernels declare every buffer read_write — without refinement
// no matmul fan-out could ever overlap. HESPER_METAL_NOREFINE=1 disables the
// refinement (all declared-rw buffers count as writers). A wrong demotion is a
// missed hazard and fails the bit-identity gate — that is the contract.
std::vector<uint8_t> parseWriteMask(const std::string& wgsl) {
    static const bool noRefine = getenv("HESPER_METAL_NOREFINE") != nullptr;
    std::vector<uint8_t> mask;
    size_t p = 0;
    while ((p = wgsl.find("@binding(", p)) != std::string::npos) {
        p += strlen("@binding(");
        uint32_t binding = (uint32_t)strtoul(wgsl.c_str() + p, nullptr, 10);
        size_t v = wgsl.find("var<", p);
        if (v == std::string::npos) break;
        size_t declEnd = wgsl.find(';', v);
        if (declEnd == std::string::npos) break;
        std::string decl = wgsl.substr(v, declEnd - v);
        bool rw = decl.find("read_write") != std::string::npos;
        bool isWrite = rw;
        if (rw && !noRefine) {
            // extract the variable name: "var<...> NAME:"
            size_t gt = decl.find('>');
            size_t colon = decl.find(':', gt);
            if (gt != std::string::npos && colon != std::string::npos) {
                std::string name = decl.substr(gt + 1, colon - gt - 1);
                // trim
                while (!name.empty() && (name.front() == ' ')) name.erase(name.begin());
                while (!name.empty() && (name.back() == ' ')) name.pop_back();
                if (!name.empty()) {
                    bool stored = false;
                    // "name[" ... "]" "=" (assignment, not ==) within one statement
                    size_t q = 0;
                    std::string pat = name + "[";
                    while (!stored && (q = wgsl.find(pat, q)) != std::string::npos) {
                        // preceding char must not be identifier (avoid suffix matches)
                        if (q > 0 && (isalnum((unsigned char)wgsl[q-1]) || wgsl[q-1] == '_')) { q += pat.size(); continue; }
                        size_t stmtEnd = wgsl.find(';', q);
                        if (stmtEnd == std::string::npos) stmtEnd = wgsl.size();
                        // find the matching close bracket then check for '=' (not '==', '<=', '>=', '!=')
                        int depth = 0; size_t r = q + name.size();
                        for (; r < stmtEnd; r++) {
                            if (wgsl[r] == '[') depth++;
                            else if (wgsl[r] == ']') { depth--; if (depth == 0) { r++; break; } }
                        }
                        while (r < stmtEnd && wgsl[r] == ' ') r++;
                        if (r < stmtEnd && wgsl[r] == '=' && (r + 1 >= stmtEnd || wgsl[r+1] != '=')) {
                            char prev = (r > 0) ? wgsl[r-1] : ' ';
                            if (prev != '<' && prev != '>' && prev != '!') stored = true;
                        }
                        q += pat.size();
                    }
                    if (!stored && wgsl.find("subgroupMatrixStore(&" + name) != std::string::npos) stored = true;
                    if (!stored && wgsl.find("subgroupMatrixStore(&(" + name) != std::string::npos) stored = true;
                    if (!stored) {
                        // atomic RMW/store through &name[
                        size_t a = 0;
                        while ((a = wgsl.find("(&" + name + "[", a)) != std::string::npos) {
                            size_t as = wgsl.rfind("atomic", a > 40 ? a - 40 : 0);
                            if (as != std::string::npos && a - as < 40) { stored = true; break; }
                            a += 2;
                        }
                    }
                    isWrite = stored;
                }
            }
        }
        if (mask.size() <= binding) mask.resize(binding + 1, 0);
        mask[binding] = isWrite ? 1 : 0;
    }
    return mask;
}

std::string parseEntryPoint(const std::string& msl) {
    size_t p = msl.find("kernel void ");
    if (p == std::string::npos) return "main";
    p += strlen("kernel void ");
    size_t e = p;
    while (e < msl.size() && (isalnum((unsigned char)msl[e]) || msl[e] == '_')) e++;
    return msl.substr(p, e - p);
}

} // namespace

extern "C" {

int hm_available(void) { return 1; }

void* hm_get_ctx(void) {
    std::lock_guard<std::mutex> lk(g_ctx_mu);
    if (g_ctx) return g_ctx;
    HMCtx* ctx = new HMCtx();
    ctx->dev = MTLCreateSystemDefaultDevice();  // +1
    if (!ctx->dev) { delete ctx; return nullptr; }
    ctx->queue = [ctx->dev newCommandQueue];    // +1
    ctx->lastCB = nil;
    const char* tp = getenv("HESPER_TINT");
    if (tp) ctx->tintPath = tp;
    else {
        const char* home = getenv("HOME");
        ctx->tintPath = std::string(home ? home : "") + "/git/verilean/hesper/.lake/build/tint-cli/tint";
    }
    const char* home = getenv("HOME");
    ctx->cacheDir = std::string(home ? home : "/tmp") + "/.cache/hesper-msl";
    mkdir((std::string(home ? home : "/tmp") + "/.cache").c_str(), 0755);
    mkdir(ctx->cacheDir.c_str(), 0755);
    fprintf(stderr, "[metal-backend] device=%s tint=%s cache=%s\n",
            [[ctx->dev name] UTF8String], ctx->tintPath.c_str(), ctx->cacheDir.c_str());
    g_ctx = ctx;
    return ctx;
}

const char* hm_device_name(void* ctxp) {
    HMCtx* ctx = (HMCtx*)ctxp;
    return [[ctx->dev name] UTF8String];
}

// ---- buffers ---------------------------------------------------------------

void* hm_create_buffer(void* ctxp, size_t size) {
    HMCtx* ctx = (HMCtx*)ctxp;
    @autoreleasepool {
        size_t sz = size < 4 ? 4 : size;
        id<MTLBuffer> b = [ctx->dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
        if (!b) return nullptr;
        memset([b contents], 0, sz);   // Dawn zero-init parity — CRITICAL
        HMBuf* hb = new HMBuf{b, sz};
        return hb;
    }
}

void hm_write_buffer(void* /*ctxp*/, void* bufp, size_t offset, const void* data, size_t n) {
    HMBuf* hb = (HMBuf*)bufp;
    if (offset + n > hb->size) n = (offset < hb->size) ? hb->size - offset : 0;
    if (n) memcpy((uint8_t*)[hb->buf contents] + offset, data, n);
}

// Read AFTER fencing on the last committed batch (unified memory: direct copy).
int hm_read_buffer(void* ctxp, void* bufp, size_t offset, void* dst, size_t n) {
    HMCtx* ctx = (HMCtx*)ctxp;
    HMBuf* hb = (HMBuf*)bufp;
    id<MTLCommandBuffer> last = nil;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        last = ctx->lastCB;
        if (last) [last retain];
    }
    if (last) { [last waitUntilCompleted]; [last release]; }
    if (offset + n > hb->size) return 0;
    memcpy(dst, (uint8_t*)[hb->buf contents] + offset, n);
    return 1;
}

uint64_t hm_buffer_id(void* bufp) {
    HMBuf* hb = (HMBuf*)bufp;
    return (uint64_t)(uintptr_t)hb->buf;
}

void hm_free_buffer(void* bufp) {
    HMBuf* hb = (HMBuf*)bufp;
    [hb->buf release];
    delete hb;
}

// ---- shaders / pipelines ---------------------------------------------------

void* hm_create_shader(void* /*ctxp*/, const char* wgsl) {
    HMShader* sh = new HMShader();
    sh->wgsl = wgsl;
    sh->tgBytes = parseWorkgroupBytes(sh->wgsl);
    parseWorkgroupSize(sh->wgsl, sh->wgX, sh->wgY, sh->wgZ);
    sh->writeMask = parseWriteMask(sh->wgsl);
    return sh;
}

void hm_free_shader(void* shp) { delete (HMShader*)shp; }

// Returns HMPipe* or nullptr; err_out (static buffer) describes failures.
static char g_pipe_err[2048];
const char* hm_last_pipeline_error(void) { return g_pipe_err; }

void* hm_create_pipeline(void* ctxp, void* shp) {
    HMCtx* ctx = (HMCtx*)ctxp;
    HMShader* sh = (HMShader*)shp;
    uint64_t h = fnv64(sh->wgsl.data(), sh->wgsl.size());
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        auto it = ctx->psoCache.find(h);
        if (it != ctx->psoCache.end()) return it->second;
    }
    @autoreleasepool {
        std::string msl, err;
        if (!runTint(ctx, sh->wgsl, h, msl, err)) {
            snprintf(g_pipe_err, sizeof(g_pipe_err), "%s", err.c_str());
            return nullptr;
        }
        std::string entry = parseEntryPoint(msl);

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_2;
        // fast math = native Dawn parity (Dawn default compiles fastMathEnabled)
#if defined(__MAC_15_0)
        opts.mathMode = MTLMathModeFast;
#else
        opts.fastMathEnabled = YES;
#endif
        NSError* nserr = nil;
        NSString* src = [[NSString alloc] initWithBytes:msl.data() length:msl.size() encoding:NSUTF8StringEncoding];
        id<MTLLibrary> lib = [ctx->dev newLibraryWithSource:src options:opts error:&nserr];
        [src release];
        [opts release];
        if (!lib) {
            snprintf(g_pipe_err, sizeof(g_pipe_err), "MSL compile: %s",
                     nserr ? [[nserr localizedDescription] UTF8String] : "?");
            return nullptr;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:
            [NSString stringWithUTF8String:entry.c_str()]];
        if (!fn) {
            snprintf(g_pipe_err, sizeof(g_pipe_err), "entry '%s' not found", entry.c_str());
            [lib release];
            return nullptr;
        }
        id<MTLComputePipelineState> pso = [ctx->dev newComputePipelineStateWithFunction:fn error:&nserr];
        [fn release];
        [lib release];
        if (!pso) {
            snprintf(g_pipe_err, sizeof(g_pipe_err), "PSO: %s",
                     nserr ? [[nserr localizedDescription] UTF8String] : "?");
            return nullptr;
        }
        HMPipe* hp = new HMPipe{pso, sh->tgBytes, sh->wgX, sh->wgY, sh->wgZ, sh->writeMask};
        std::lock_guard<std::mutex> lk(ctx->mu);
        auto it = ctx->psoCache.find(h);
        if (it != ctx->psoCache.end()) { [pso release]; delete hp; return it->second; }
        ctx->psoCache[h] = hp;
        return hp;
    }
}

// pipelines are cache-owned: Lean-side finalizer is a no-op
void hm_free_pipeline(void* /*p*/) {}

// ---- bind groups -----------------------------------------------------------

void* hm_create_bindgroup(void* /*ctxp*/, uint32_t n, const uint32_t* bindings, void** bufs) {
    HMBind* bg = new HMBind();
    bg->entries.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        HMBuf* hb = (HMBuf*)bufs[i];
        [hb->buf retain];
        bg->entries.emplace_back(bindings[i], hb->buf);
    }
    return bg;
}

void hm_free_bindgroup(void* bgp) {
    HMBind* bg = (HMBind*)bgp;
    for (auto& e : bg->entries) [e.second release];
    delete bg;
}

// ---- encoding / submission -------------------------------------------------

static std::atomic<uint64_t> g_hm_dispatches{0};
static std::atomic<uint64_t> g_hm_barriers{0};

// ---- tagged GPU-time attribution (DG_MOEISO-class isolation measurements) --
// A command buffer's GPU busy time (GPUEnd-GPUStart) is accumulated into the
// tag that was current at commit. The Lean side flushes at tag switches so a
// CB never spans two tags. No waits added — attribution is completion-handler
// based, so it does not inflate the measured range (unlike pmark/DG_PROF).
static std::atomic<int> g_hm_tag{0};
static std::atomic<uint64_t> g_hm_tag_ns[8] = {};
extern "C" void hm_tag_set(int t) { g_hm_tag.store(t & 7, std::memory_order_relaxed); }
extern "C" int hm_tag_get(void) { return g_hm_tag.load(std::memory_order_relaxed); }
extern "C" void hm_tag_account_ns(uint64_t ns, int tag) {
    g_hm_tag_ns[tag & 7].fetch_add(ns, std::memory_order_relaxed);
}
extern "C" uint64_t hm_tag_read_ns(int t) {
    return g_hm_tag_ns[t & 7].load(std::memory_order_relaxed);
}

void* hm_encoder_new(void* ctxp) {
    HMCtx* ctx = (HMCtx*)ctxp;
    static const bool serial = getenv("HESPER_METAL_SERIAL") != nullptr;
    @autoreleasepool {
        HMEnc* e = new HMEnc();
        e->cb = [[ctx->queue commandBuffer] retain];
        e->concurrent = !serial;
        e->enc = [[e->cb computeCommandEncoderWithDispatchType:
                   (serial ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent)] retain];
        return e;
    }
}

void hm_record(void* /*ctxp*/, void* encp, void* pipep, void* bgp,
               uint32_t wx, uint32_t wy, uint32_t wz) {
    HMEnc* e = (HMEnc*)encp;
    HMPipe* p = (HMPipe*)pipep;
    HMBind* bg = (HMBind*)bgp;
    if (!e->enc) return;
    if (e->concurrent) {
        // static hazard analysis: full barrier only on RAW / WAW / WAR at buffer
        // granularity; independent dispatches between barriers overlap.
        bool conflict = false;
        for (auto& en : bg->entries) {
            void* key = (void*)en.second;
            bool w = en.first < p->writeMask.size() && p->writeMask[en.first];
            if (w) {
                if (e->writeSet.count(key) || e->readSet.count(key)) { conflict = true; break; }
            } else {
                if (e->writeSet.count(key)) { conflict = true; break; }
            }
        }
        if (conflict) {
            [e->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            e->readSet.clear();
            e->writeSet.clear();
            g_hm_barriers.fetch_add(1, std::memory_order_relaxed);
        }
        for (auto& en : bg->entries) {
            void* key = (void*)en.second;
            bool w = en.first < p->writeMask.size() && p->writeMask[en.first];
            (w ? e->writeSet : e->readSet).insert(key);
        }
        g_hm_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    [e->enc setComputePipelineState:p->pso];
    for (auto& en : bg->entries)
        [e->enc setBuffer:en.second offset:0 atIndex:en.first];
    if (p->tgBytes)
        [e->enc setThreadgroupMemoryLength:p->tgBytes atIndex:0];
    [e->enc dispatchThreadgroups:MTLSizeMake(wx, wy, wz)
           threadsPerThreadgroup:MTLSizeMake(p->wgX, p->wgY, p->wgZ)];
}

void hm_submit(void* ctxp, void* encp, int wait) {
    HMCtx* ctx = (HMCtx*)ctxp;
    HMEnc* e = (HMEnc*)encp;
    if (e->enc) { [e->enc endEncoding]; [e->enc release]; e->enc = nil; }
    {
        int tag = g_hm_tag.load(std::memory_order_relaxed);
        [e->cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            if (c.GPUEndTime > c.GPUStartTime)
                hm_tag_account_ns((uint64_t)((c.GPUEndTime - c.GPUStartTime) * 1e9), tag);
        }];
    }
    [e->cb commit];
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        if (ctx->lastCB) [ctx->lastCB release];
        ctx->lastCB = [e->cb retain];
    }
    if (wait) {
        [e->cb waitUntilCompleted];
        static const bool stats = getenv("HESPER_METAL_STATS") != nullptr;
        if (stats) {
            fprintf(stderr, "[metal-stats] dispatches=%llu barriers=%llu (%.1f%%)\n",
                    (unsigned long long)g_hm_dispatches.load(),
                    (unsigned long long)g_hm_barriers.load(),
                    g_hm_dispatches.load() ? 100.0 * g_hm_barriers.load() / g_hm_dispatches.load() : 0.0);
        }
    }
}

void hm_free_encoder(void* encp) {
    HMEnc* e = (HMEnc*)encp;
    if (e->enc) { [e->enc endEncoding]; [e->enc release]; }
    if (e->cb) [e->cb release];
    delete e;
}

// One-shot dispatch (parity-exe path): encode, commit, wait.
int hm_dispatch_once(void* ctxp, void* pipep, void* bgp,
                     uint32_t wx, uint32_t wy, uint32_t wz) {
    void* e = hm_encoder_new(ctxp);
    hm_record(ctxp, e, pipep, bgp, wx, wy, wz);
    hm_submit(ctxp, e, 1);
    hm_free_encoder(e);
    return 1;
}

// ---- accessors for metal_replace.mm (hand-MSL kernels on this backend) ----

id<MTLDevice> hm_mtl_device(void) {
    HMCtx* ctx = (HMCtx*)hm_get_ctx();
    return ctx ? ctx->dev : nil;
}

id<MTLCommandQueue> hm_mtl_queue(void) {
    HMCtx* ctx = (HMCtx*)hm_get_ctx();
    return ctx ? ctx->queue : nil;
}

id<MTLBuffer> hm_mtl_buffer_of(void* hmbuf) {
    return hmbuf ? ((HMBuf*)hmbuf)->buf : nil;
}

void hm_wait_idle(void* ctxp) {
    HMCtx* ctx = (HMCtx*)ctxp;
    id<MTLCommandBuffer> last = nil;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        last = ctx->lastCB;
        if (last) [last retain];
    }
    if (last) { [last waitUntilCompleted]; [last release]; }
}

} // extern "C"

#endif // __APPLE__
