// webml decode-token replayer (Exp 1 / DEVPLAN §9): reads out/manifest.json (produced
// by convert.py from a browser WebGPU trace), compiles the Tint-MSL kernels, allocates
// buffers (small param/uniform buffers get their REAL traced contents; big weight
// buffers are zero-filled — timing-only), then encodes the token's dispatch sequence
// serial and concurrent-no-barrier, 20 iters each, reporting GPU wall min/avg.
//
//   clang++ -fobjc-arc -std=c++17 replayer.mm -framework Metal -framework Foundation -o replayer
//   ./replayer out/manifest.json
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:(argc > 1 ? argv[1] : "out/manifest.json")];
        NSData* data = [NSData dataWithContentsOfFile:path];
        if (!data) { fprintf(stderr, "no manifest at %s\n", path.UTF8String); return 1; }
        NSError* err = nil;
        NSDictionary* mf = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!mf) { fprintf(stderr, "bad manifest: %s\n", err.description.UTF8String); return 1; }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString* dir = [path stringByDeletingLastPathComponent];

        // pipelines
        NSArray* pipes = mf[@"pipes"];
        std::vector<id<MTLComputePipelineState>> psos;
        std::vector<MTLSize> wgs;
        std::vector<NSUInteger> tgBytes;
        for (NSDictionary* p in pipes) {
            NSString* msl = [NSString stringWithContentsOfFile:
                [dir stringByAppendingPathComponent:p[@"msl"]] encoding:NSUTF8StringEncoding error:&err];
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [dev newLibraryWithSource:msl options:opts error:&err];
            if (!lib) { fprintf(stderr, "compile failed %s: %s\n",
                [p[@"msl"] UTF8String], err.description.UTF8String); return 1; }
            id<MTLFunction> fn = [lib newFunctionWithName:p[@"entry"]];
            if (!fn) { fprintf(stderr, "no entry %s in %s\n",
                [p[@"entry"] UTF8String], [p[@"msl"] UTF8String]); return 1; }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "pso failed %s: %s\n",
                [p[@"msl"] UTF8String], err.description.UTF8String); return 1; }
            psos.push_back(pso);
            NSArray* w = p[@"wg"];
            wgs.push_back(MTLSizeMake([w[0] unsignedIntegerValue], [w[1] unsignedIntegerValue], [w[2] unsignedIntegerValue]));
            tgBytes.push_back([p[@"tgBytes"] unsignedIntegerValue]);
        }

        // buffers (zero-filled; small ones get traced contents)
        NSMutableDictionary<NSNumber*, id<MTLBuffer>>* bufs = [NSMutableDictionary new];
        unsigned long long total = 0;
        for (NSDictionary* b in mf[@"buffers"]) {
            NSUInteger size = [b[@"size"] unsignedIntegerValue];
            id<MTLBuffer> mb = [dev newBufferWithLength:MAX(size, (NSUInteger)4)
                                                options:MTLResourceStorageModeShared];
            memset([mb contents], 0, mb.length);
            bufs[b[@"id"]] = mb;
            total += size;
        }
        NSDictionary* contents = mf[@"contents"];
        for (NSString* key in contents) {
            NSData* raw = [[NSData alloc] initWithBase64EncodedString:contents[key] options:0];
            id<MTLBuffer> mb = bufs[@([key integerValue])];
            if (mb && raw.length <= mb.length) memcpy([mb contents], raw.bytes, raw.length);
        }

        NSArray* ops = mf[@"ops"];
        fprintf(stderr, "[replayer] pipes=%zu ops=%lu buffers=%lu total=%.2fGB contents=%lu\n",
                psos.size(), (unsigned long)ops.count, (unsigned long)bufs.count,
                total / 1e9, (unsigned long)contents.count);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        for (int mode = 0; mode < 2; mode++) {
            bool concurrent = (mode == 1);
            double best = 1e30, sum = 0;
            const int iters = 20;
            for (int r = 0; r < iters; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc;
                if (concurrent) {
                    MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
                    d.dispatchType = MTLDispatchTypeConcurrent;
                    enc = [cb computeCommandEncoderWithDescriptor:d];
                } else {
                    enc = [cb computeCommandEncoder];
                }
                for (NSDictionary* op in ops) {
                    NSUInteger pi = [op[@"p"] unsignedIntegerValue];
                    [enc setComputePipelineState:psos[pi]];
                    for (NSArray* bind in op[@"binds"]) {
                        [enc setBuffer:bufs[bind[1]]
                                offset:[bind[2] unsignedIntegerValue]
                               atIndex:[bind[0] unsignedIntegerValue]];
                    }
                    if (tgBytes[pi] > 0)
                        [enc setThreadgroupMemoryLength:((tgBytes[pi] + 15) & ~15ul) atIndex:0];
                    NSArray* g = op[@"grid"];
                    [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntegerValue],
                                                          [g[1] unsignedIntegerValue],
                                                          [g[2] unsignedIntegerValue])
                        threadsPerThreadgroup:wgs[pi]];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError) {
                    fprintf(stderr, "cb error: %s\n", cb.error.description.UTF8String);
                    return 1;
                }
                double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                if (ms < best) best = ms;
                sum += ms;
            }
            printf("[webml-replay] ops=%lu mode=%s min=%.4f avg=%.4f ms\n",
                   (unsigned long)ops.count, concurrent ? "concurrent-nobarrier" : "serial",
                   best, sum / iters);
        }
        return 0;
    }
}
