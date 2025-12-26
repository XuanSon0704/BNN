#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>


char LICENSE[] SEC("license") = "GPL";

#define ETH_P_IP 0x0800
#define N_FEATURES 10
#define N_HIDDEN 64

#define N_OUTPUT 1

#define N_FEATURES_PACKED 1
#define N_HIDDEN_PACKED 1   

// --- MAPS ---
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __u64); 
    __uint(max_entries, N_FEATURES);
} threshold_map SEC(".maps");


struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_FEATURES_PACKED]);
    __uint(max_entries, N_HIDDEN);
} w1_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_HIDDEN_PACKED]);
    __uint(max_entries, N_HIDDEN);
} w2_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_HIDDEN_PACKED]);
    __uint(max_entries, N_OUTPUT);
} w_out_map SEC(".maps");

struct { // Thresholds (Folded BN: Mean/Var/Gamma/Beta -> s64)
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __s64); 
    __uint(max_entries, N_HIDDEN);
} t1_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __s64);
    __uint(max_entries, N_HIDDEN);
} t2_map SEC(".maps");


// scratch per-cpu arrays for packed activations (avoid races)
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_FEATURES_PACKED]);
    __uint(max_entries, 1);
} scratch_a0_packed SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_HIDDEN_PACKED]);
    __uint(max_entries, 1);
} scratch_a1_packed SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, u64[N_HIDDEN_PACKED]);
    __uint(max_entries, 1);
} scratch_a2_packed SEC(".maps");

// Metrics map: index 0 -> struct metrics
struct metrics {
    __u64 tp;
    __u64 fp;
    __u64 tn;
    __u64 fn;
    __u64 pkt_count;
    __u64 drop_count;
    __u64 total_proc_ns; 
};
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, struct metrics);
    __uint(max_entries, 1);
} metrics_map SEC(".maps");

static __always_inline int sign_s64(s64 x) {
    return (x >= 0) ? 1 : -1;
}

static __always_inline s64 xnor_popcount(const u64 *a, const u64 *b, int packed_len, int total_bits) {
    s64 sum = 0;
    int remainder = total_bits % 64;
    u64 last_mask = (remainder == 0) ? ~0ULL : ((1ULL << remainder) - 1);

    #pragma unroll
    for (int i = 0; i < packed_len; i++) {
        u64 xnor_val = ~(a[i] ^ b[i]);
        if (i == packed_len - 1) {
            xnor_val &= last_mask;
        }
        s64 ones = __builtin_popcountll(xnor_val);
        s64 current_bits = (i == packed_len - 1 && remainder != 0) ? remainder : 64;
        sum += (2 * ones - current_bits);
    }
    return sum;
}

static __always_inline void binarize_pack_features(const s64 features[N_FEATURES], u64 packed_out[N_FEATURES_PACKED]) {
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PACKED; i++) packed_out[i] = 0ULL;

    #pragma unroll
    for (int i = 0; i < N_FEATURES; i++) {
        int idx = i;
        u64 *thresh_ptr = bpf_map_lookup_elem(&threshold_map, &idx);
        u64 thresh = 0;
        if (thresh_ptr) thresh = *thresh_ptr;

        if (features[i] >= thresh) {
            int pi = i / 64;
            int bp = i % 64;
            packed_out[pi] |= (1ULL << bp);
        }
    }
}


static __always_inline int extract_features(struct xdp_md *ctx, s64 features[N_FEATURES]) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void*)(eth + 1) > data_end) return -1;
    if (bpf_ntohs(eth->h_proto) != ETH_P_IP) return -1;

    struct iphdr *iph = data + sizeof(*eth);
    if ((void*)(iph + 1) > data_end) return -1;

    s64 pkt_len = (s64)bpf_ntohs(iph->tot_len);
    s64 init_win_bytes = 0;

    if (iph->protocol == IPPROTO_TCP) {
        u32 ip_hdr_len = iph->ihl * 4;
        struct tcphdr *tcph = (void*)iph + ip_hdr_len;
        if ((void*)(tcph + 1) <= data_end) {
            init_win_bytes = (s64)bpf_ntohs(tcph->window);
        }
    }

    // Mapping Features 
    features[0] = pkt_len;      // Avg Fwd Segment Size
    features[1] = init_win_bytes; // Init_Win_bytes_forward
    features[2] = 0;            // Flow Packets/s
    features[3] = 0;            // Fwd Packets/s
    features[4] = pkt_len;      // Average Packet Size
    features[5] = pkt_len;      // Total Length of Fwd Packets
    features[6] = pkt_len;      // Subflow Fwd Bytes
    features[7] = pkt_len;      // Max
    features[8] = pkt_len;      // Min
    features[9] = pkt_len;      // Mean

    return 0;
}

SEC("xdp")
int bnn_xdp(struct xdp_md *ctx) {
    __u32 zero = 0;
    struct metrics *m = bpf_map_lookup_elem(&metrics_map, &zero);
    if (!m) return XDP_PASS;

    // timing start
    u64 t0 = bpf_ktime_get_ns();

    // 1. scratch
    u64 *a0_pack = bpf_map_lookup_elem(&scratch_a0_packed, &zero);
    u64 *a1_pack = bpf_map_lookup_elem(&scratch_a1_packed, &zero);
    u64 *a2_pack = bpf_map_lookup_elem(&scratch_a2_packed, &zero);
    if (!a0_pack || !a1_pack || !a2_pack) return XDP_PASS;


    // 2. Extract & Binarize Input
    s64 features[N_FEATURES];
    if (extract_features(ctx, features) < 0) return XDP_PASS;
    
    binarize_pack_features(features, a0_pack);

    // Reset next layers
    #pragma unroll
    for (int i=0; i<N_HIDDEN_PACKED; i++) { a1_pack[i] = 0; a2_pack[i] = 0; }

    // 3. Hidden Layer 1
    for (int i = 0; i < N_HIDDEN; i++) {
        __u32 idx = i;
        u64 *w1_row = bpf_map_lookup_elem(&w1_map, &idx);
        s64 *thresh1 = bpf_map_lookup_elem(&t1_map, &idx);
        
        if (w1_row && thresh1) {
            // XNOR với input layer
            s64 dot = xnor_popcount(a0_pack, w1_row, N_FEATURES_PACKED, N_FEATURES);
            
            // So sánh threshold (đã fold BN)
            if (dot > *thresh1) {
                int pi = i / 64;
                int bp = i % 64;
                a1_pack[pi] |= (1ULL << bp); // Set bit 1
            }
        }
    }

    // 4. Hidden Layer 2
    for (int i = 0; i < N_HIDDEN; i++) {
        __u32 idx = i;
        u64 *w2_row = bpf_map_lookup_elem(&w2_map, &idx);
        s64 *thresh2 = bpf_map_lookup_elem(&t2_map, &idx);

        if (w2_row && thresh2) {
            // XNOR với hidden layer 1
            s64 dot = xnor_popcount(a1_pack, w2_row, N_HIDDEN_PACKED, N_HIDDEN);
            
            if (dot > *thresh2) {
                int pi = i / 64;
                int bp = i % 64;
                a2_pack[pi] |= (1ULL << bp);
            }
        }
    }

    // 5. Output Layer 
    // Tính 1 neuron duy nhất
    u64 *wout_row = bpf_map_lookup_elem(&w_out_map, &zero);
    if (wout_row) {
        // XNOR giữa hidden layer 2 và weight output
        s64 score = xnor_popcount(a2_pack, wout_row, N_HIDDEN_PACKED, N_HIDDEN);
        
        int pred = (score >= 0) ? 1 : 0;

        // Metrics Update
        __sync_fetch_and_add(&m->pkt_count, 1);
        u64 t1_time = bpf_ktime_get_ns();
        __sync_fetch_and_add(&m->total_proc_ns, (t1_time - t0));

        // Decision
        if (pred == 1) {
            __sync_fetch_and_add(&m->drop_count, 1);
            return XDP_DROP;
        }
    }

    return XDP_PASS;
}