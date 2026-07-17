// metal_backend_stub.cpp — non-Apple stand-in for metal_backend.mm.
// hm_available() = 0 keeps hesper_metal_mode() permanently false, so none of
// the other entry points can be reached; they abort defensively if they are.
#include <cstdint>
#include <cstdlib>
#include <cstdio>

extern "C" {
static void* die() { fprintf(stderr, "metal backend stub called\n"); abort(); }
int hm_available(void) { return 0; }
void* hm_get_ctx(void) { return die(); }
void* hm_create_buffer(void*, size_t) { return die(); }
void hm_write_buffer(void*, void*, size_t, const void*, size_t) { die(); }
int hm_read_buffer(void*, void*, size_t, void*, size_t) { die(); return 0; }
uint64_t hm_buffer_id(void*) { die(); return 0; }
void hm_free_buffer(void*) {}
void* hm_create_shader(void*, const char*) { return die(); }
void hm_free_shader(void*) {}
void* hm_create_pipeline(void*, void*) { return die(); }
const char* hm_last_pipeline_error(void) { return "stub"; }
void* hm_create_bindgroup(void*, uint32_t, const uint32_t*, void**) { return die(); }
void hm_free_bindgroup(void*) {}
void* hm_encoder_new(void*) { return die(); }
void hm_record(void*, void*, void*, void*, uint32_t, uint32_t, uint32_t) { die(); }
void hm_submit(void*, void*, int) { die(); }
void hm_free_encoder(void*) {}
int hm_dispatch_once(void*, void*, void*, uint32_t, uint32_t, uint32_t) { die(); return 0; }
void hm_wait_idle(void*) { die(); }
}
