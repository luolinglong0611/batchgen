/* Scoped Linux/x86_64 NVSHMEM 3.3.24 IBGDA DCT initialization workaround.
 *
 * Build on the target against the SAME verbs headers as the working CPU probe:
 *   cc -std=c11 -O2 -fPIC -shared nvshmem3324_dct_ah_init_fix_v2.c \
 *      -o nvshmem3324_dct_ah_init_fix_v2.so -ldl -pthread
 *
 * NVSHMEM resolves ibv_create_ah with dlsym(an explicit libibverbs handle),
 * which bypasses ordinary LD_PRELOAD interposition of ibv_create_ah alone.
 * This narrowly substitutes that explicit-handle dlsym result. Non-target
 * calls tail-transfer to real dlsym so RTLD_NEXT keeps the original caller;
 * verify that sibling tail jump in the target compiler's objdump before use.
 * Default is observe-only. The launcher MUST pin the exact NVSHMEM 3.3.24
 * IBGDA plugin hash and review the generated dispatcher tail jump. Enable:
 *   BATCHGEN_NVSHMEM3324_DCT_AH_INIT_FIX=1
 *   BATCHGEN_NVSHMEM3324_DCT_AH_CALLER_RVA=0x<observed DCT return offset>
 * Optional exact file gate:
 *   BATCHGEN_NVSHMEM3324_DCT_AH_LIBRARY=/absolute/plugin/path
 * Both dlsym and AH caller must be the IBGDA plugin; an exact DCT caller RVA
 * is mandatory. The single static_rate field is deterministically set to
 * zero BEFORE the one real call, including when the random byte happens to
 * be a valid nonzero rate. At most eight DCT init calls per PID are allowed;
 * exceeding this bound fails closed. No GID/dlid/route/payload changes.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void *(*dlsym_fn)(void *, const char *);
typedef struct ibv_ah *(*create_ah_fn)(struct ibv_pd *, struct ibv_ah_attr *);
static dlsym_fn real_dlsym;
static create_ah_fn real_create_ah;
static pthread_once_t resolver_once = PTHREAD_ONCE_INIT;
static unsigned int init_repairs;

static int is_ibgda_plugin(const Dl_info *info) {
    if (!info->dli_fname) return 0;
    const char *base = strrchr(info->dli_fname, '/');
    base = base ? base + 1 : info->dli_fname;
    const char prefix[] = "nvshmem_transport_ibgda.so";
    const char libprefix[] = "libnvshmem_transport_ibgda.so";
    return strncmp(base, prefix, sizeof(prefix)-1) == 0 ||
           strncmp(base, libprefix, sizeof(libprefix)-1) == 0;
}

static int scoped_init_fix(const Dl_info *caller, unsigned long long rva) {
    const char *enabled = getenv("BATCHGEN_NVSHMEM3324_DCT_AH_INIT_FIX");
    const char *offset = getenv("BATCHGEN_NVSHMEM3324_DCT_AH_CALLER_RVA");
    const char *library = getenv("BATCHGEN_NVSHMEM3324_DCT_AH_LIBRARY");
    if (!enabled || strcmp(enabled, "1") != 0 || !offset || !is_ibgda_plugin(caller)) return 0;
    if (library && strcmp(library, caller->dli_fname) != 0) return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long wanted = strtoull(offset, &end, 0);
    if (errno || !end || *end != '\0' || wanted != rva) return 0;
    return 1;
}

static void initialize_resolver(void) {
    /* dlvsym is deliberately NOT interposed. Both versions are public glibc
     * symbols on the target's x86_64 ABI; no private __libc_dlsym is used. */
    real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym)
        real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!real_dlsym) {
        const char message[] = "NVSHMEM_AH_DIAG: cannot resolve real glibc dlsym\n";
        (void)write(2, message, sizeof(message) - 1);
        _exit(127);
    }
}

static void describe(const char *phase, struct ibv_pd *pd,
                     const struct ibv_ah_attr *attr, struct ibv_ah *result,
                     int before_errno, int after_errno) {
    char gid[33];
    for (int i = 0; i < 16; ++i)
        (void)snprintf(gid + i * 2, 3, "%02x", attr->grh.dgid.raw[i]);
    gid[32] = '\0';
    const char *rank = getenv("RANK");
    fprintf(stderr,
            "NVSHMEM_AH_DIAG phase=%s pid=%ld rank=%s pd=%p context=%p "
            "ah=%p errno_before=%d errno_after=%d errno_text=%s "
            "attr_size=%zu dlid=%u sl=%u src_path_bits=%u static_rate=%u "
            "is_global=%u port=%u sgid_index=%u hop_limit=%u traffic_class=%u "
            "flow_label=%u dgid=%s\n",
            phase, (long)getpid(), rank ? rank : "unset", (void *)pd,
            pd ? (void *)pd->context : NULL, (void *)result,
            before_errno, after_errno, strerror(after_errno),
            sizeof(*attr), attr->dlid, attr->sl, attr->src_path_bits,
            attr->static_rate, attr->is_global, attr->port_num,
            attr->grh.sgid_index, attr->grh.hop_limit,
            attr->grh.traffic_class, attr->grh.flow_label, gid);
}

static struct ibv_ah *diagnostic_create_ah(struct ibv_pd *pd,
                                         struct ibv_ah_attr *attr) {
    create_ah_fn original = __atomic_load_n(&real_create_ah, __ATOMIC_ACQUIRE);
    const int before_errno = errno;
    Dl_info caller = {0};
    void *site = __builtin_return_address(0);
    (void)dladdr(site, &caller);
    unsigned long long rva = caller.dli_fbase ?
        (unsigned long long)((uintptr_t)site - (uintptr_t)caller.dli_fbase) : 0;
    const int scoped = scoped_init_fix(&caller, rva);
    if (is_ibgda_plugin(&caller))
        fprintf(stderr, "NVSHMEM_AH_DIAG callsite pid=%ld library=%s caller_RVA=0x%llx fix_armed=%d\n",
                (long)getpid(), caller.dli_fname, rva, scoped);
    if (scoped) {
        unsigned int attempt = __atomic_fetch_add(&init_repairs, 1, __ATOMIC_RELAXED);
        if (attempt >= 8) {
            fprintf(stderr, "NVSHMEM_AH_DIAG init_call_limit pid=%ld attempts=%u fail_closed=true\n",
                    (long)getpid(), attempt + 1);
            errno = EOVERFLOW;
            return NULL;
        }
        describe("NVSHMEM3324_DCT_INIT_BEFORE", pd, attr, NULL, before_errno, before_errno);
        /* This is the entire repair: one declared uint8_t member. Preserve
         * every other byte, including dlid, GID, hop limit and traffic class. */
        attr->static_rate = 0;
    }
    errno = before_errno;
    struct ibv_ah *result = original(pd, attr);
    int after_errno = errno;
    if (is_ibgda_plugin(&caller))
        describe(scoped ? "NVSHMEM3324_DCT_INIT_AFTER_static_rate_zero" : "original",
                 pd, attr, result, before_errno, after_errno);
    errno = after_errno;
    return result;
}

static __attribute__((noinline)) void *resolve_diagnostic_ah(
        void *handle, const char *symbol, void *original_caller) {
    void *address = real_dlsym(handle, symbol);
    if (!address) return address;
    const int saved_errno = errno;
    Dl_info binding_caller = {0};
    (void)dladdr(original_caller, &binding_caller);
    if (!is_ibgda_plugin(&binding_caller)) {
        errno = saved_errno;
        return address;
    }
    create_ah_fn expected = NULL;
    create_ah_fn target = (create_ah_fn)address;
    if (!__atomic_compare_exchange_n(&real_create_ah, &expected, target, 0,
                                     __ATOMIC_RELEASE, __ATOMIC_ACQUIRE) &&
        expected != target) {
        /* Do not send a PD from another loaded verbs library into the first
         * library. Observe the conflict and leave that binding untouched. */
        fprintf(stderr, "NVSHMEM_AH_DIAG symbol_conflict pid=%ld previous=%p next=%p\n",
                (long)getpid(), (void *)expected, address);
        errno = saved_errno;
        return address;
    }
    Dl_info library = {0}, caller = {0};
    (void)dladdr(address, &library);
    (void)dladdr(original_caller, &caller);
    fprintf(stderr,
            "NVSHMEM_AH_DIAG bound pid=%ld function=%p library=%s caller=%s "
            "AH_size=%zu offsets_dlid=%zu_static_rate=%zu_global=%zu_port=%zu\n",
            (long)getpid(), address,
            library.dli_fname ? library.dli_fname : "unknown",
            caller.dli_fname ? caller.dli_fname : "unknown", sizeof(struct ibv_ah_attr),
            offsetof(struct ibv_ah_attr, dlid), offsetof(struct ibv_ah_attr, static_rate),
            offsetof(struct ibv_ah_attr, is_global), offsetof(struct ibv_ah_attr, port_num));
    errno = saved_errno;
    return (void *)diagnostic_create_ah;
}

/* Keep this dispatcher small so GCC -O2 emits sibling tail jumps. RTLD_NEXT
 * and RTLD_DEFAULT must retain the original caller/namespace, hence are never
 * intercepted. objdump must show a tail jmp for the real_dlsym branch; do not
 * use an unoptimized build that performs call+return on that branch. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O2", "optimize-sibling-calls")))
#endif
void *dlsym(void *handle, const char *symbol) {
    pthread_once(&resolver_once, initialize_resolver);
    if (handle != RTLD_NEXT && handle != RTLD_DEFAULT &&
        strcmp(symbol, "ibv_create_ah") == 0)
        return resolve_diagnostic_ah(handle, symbol, __builtin_return_address(0));
    return real_dlsym(handle, symbol);
}
