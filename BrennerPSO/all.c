#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <float.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_WIN32)
#include <malloc.h>
#endif

#ifndef CACHE_BLOCK
#define CACHE_BLOCK 64
#endif
#ifndef ALIGN_BYTES
#define ALIGN_BYTES 64
#endif
#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define THREAD_LOCAL __thread
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define THREAD_LOCAL _Thread_local
#else
#define THREAD_LOCAL
#endif

const int n = N;
int n_threads = NTHREADS;

const float R1 = 1.7f, R2 = 2.0f, Re = 1.315f;
const float a0 = 0.011304f, c0 = 19.0f, d0 = 2.5f;
const float delta = 0.80469f, beta = 1.5f, S = 1.29f, De = 6.325f;
const float De_S_1 = De / (S - 1.0f);
const float DexS_S_1 = (De / (S - 1.0f)) * S;
const float c0sq = c0 * c0, d0sq = d0 * d0;
const float c0sq_d0sq_1 = 1.0f + (c0sq / d0sq);
const float pi = 3.14159265359f;
const float pi_DeltaR = pi / (R2 - R1);

static inline void* aligned_malloc_bytes(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, ALIGN_BYTES);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGN_BYTES, size) != 0) return NULL;
    return ptr;
#endif
}
static inline void aligned_free_bytes(void* ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
static inline float* aligned_calloc_float(size_t count) {
    float* ptr = (float*)aligned_malloc_bytes(count * sizeof(float));
    if (!ptr) { fprintf(stderr, "aligned allocation failed\n"); exit(1); }
    memset(ptr, 0, count * sizeof(float));
    return ptr;
}
static inline int* aligned_calloc_int(size_t count) {
    int* ptr = (int*)aligned_malloc_bytes(count * sizeof(int));
    if (!ptr) { fprintf(stderr, "aligned allocation failed\n"); exit(1); }
    memset(ptr, 0, count * sizeof(int));
    return ptr;
}
static inline float asm_sqrtf(float x) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    float y;
    __asm__ __volatile__("sqrtss %1, %0" : "=x"(y) : "x"(x));
    return y;
#else
    return sqrtf(x);
#endif
}
static inline float cutoff_fij(float rij) {
    if (rij > R2) return 0.0f;
    if (rij < R1) return 1.0f;
    return 0.5f * (1.0f + cosf(pi_DeltaR * (rij - R1)));
}
float f_ij(float rij) { return cutoff_fij(rij); }
float F(float xik) {
    if (xik <= 2.0f) return 1.0f;
    if (xik > 3.0f) return 0.0f;
    return 0.5f * (1.0f + cosf(pi * (xik - 2.0f)));
}

typedef struct {
    int capacity;
    float *x, *y, *z, *dist, *inv_dist, *cutoff;
    int *neighbors, *neighbor_count;
} GeometryCache;
static THREAD_LOCAL GeometryCache geom_cache = {0};

static void ensure_geometry_cache(void) {
    const size_t nn = (size_t)n * (size_t)n;
    if (geom_cache.capacity == n) return;
    aligned_free_bytes(geom_cache.x); aligned_free_bytes(geom_cache.y); aligned_free_bytes(geom_cache.z);
    aligned_free_bytes(geom_cache.dist); aligned_free_bytes(geom_cache.inv_dist); aligned_free_bytes(geom_cache.cutoff);
    aligned_free_bytes(geom_cache.neighbors); aligned_free_bytes(geom_cache.neighbor_count);
    geom_cache.x = aligned_calloc_float(n); geom_cache.y = aligned_calloc_float(n); geom_cache.z = aligned_calloc_float(n);
    geom_cache.dist = aligned_calloc_float(nn); geom_cache.inv_dist = aligned_calloc_float(nn); geom_cache.cutoff = aligned_calloc_float(nn);
    geom_cache.neighbors = aligned_calloc_int(nn); geom_cache.neighbor_count = aligned_calloc_int(n);
    geom_cache.capacity = n;
}

static void precompute_geometry(const float* restrict r) {
    ensure_geometry_cache();
    float *restrict x = geom_cache.x, *restrict y = geom_cache.y, *restrict z = geom_cache.z;
    float *restrict dist = geom_cache.dist, *restrict inv_dist = geom_cache.inv_dist, *restrict cutoff = geom_cache.cutoff;
    int *restrict neighbors = geom_cache.neighbors, *restrict neighbor_count = geom_cache.neighbor_count;
    for (int i = 0; i < n; ++i) { x[i] = r[(size_t)i*3u]; y[i] = r[(size_t)i*3u+1u]; z[i] = r[(size_t)i*3u+2u]; }
    for (int ib = 0; ib < n; ib += CACHE_BLOCK) {
        const int i_end = (ib + CACHE_BLOCK < n) ? ib + CACHE_BLOCK : n;
        for (int i = ib; i < i_end; ++i) {
            const float xi = x[i], yi = y[i], zi = z[i];
            const size_t base = (size_t)i * (size_t)n;
            int j = 0;
#if defined(__AVX2__)
            const __m256 vxi = _mm256_set1_ps(xi), vyi = _mm256_set1_ps(yi), vzi = _mm256_set1_ps(zi);
            for (; j + 7 < n; j += 8) {
                const __m256 dx = _mm256_sub_ps(vxi, _mm256_loadu_ps(x + j));
                const __m256 dy = _mm256_sub_ps(vyi, _mm256_loadu_ps(y + j));
                const __m256 dz = _mm256_sub_ps(vzi, _mm256_loadu_ps(z + j));
                const __m256 d2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz));
                _mm256_storeu_ps(dist + base + (size_t)j, _mm256_sqrt_ps(d2));
            }
#endif
            for (; j < n; ++j) {
                const float dx = xi - x[j], dy = yi - y[j], dz = zi - z[j];
                dist[base + (size_t)j] = asm_sqrtf(dx*dx + dy*dy + dz*dz);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const size_t base = (size_t)i * (size_t)n;
        int count = 0;
        for (int jb = 0; jb < n; jb += CACHE_BLOCK) {
            const int j_end = (jb + CACHE_BLOCK < n) ? jb + CACHE_BLOCK : n;
            for (int j = jb; j < j_end; ++j) {
                const float d = dist[base + (size_t)j];
                const float c = cutoff_fij(d);
                cutoff[base + (size_t)j] = c;
                inv_dist[base + (size_t)j] = (d > FLT_MIN) ? (1.0f / d) : 0.0f;
                if (i != j && c > 0.0f) neighbors[base + (size_t)count++] = j;
            }
        }
        neighbor_count[i] = count;
    }
}
static inline float angular_g(float cos_term) {
    const float shifted = cos_term + 1.0f, sq = shifted * shifted;
    return a0 * (c0sq_d0sq_1 - (c0sq / (d0sq + sq)));
}

float BrennerPotential(float* r) {
    precompute_geometry(r);
    const float *restrict x = geom_cache.x, *restrict y = geom_cache.y, *restrict z = geom_cache.z;
    const float *restrict dist = geom_cache.dist, *restrict inv_dist = geom_cache.inv_dist, *restrict cutoff = geom_cache.cutoff;
    const int *restrict neighbors = geom_cache.neighbors, *restrict neighbor_count = geom_cache.neighbor_count;
    const float sqrt_2_Sxbeta = sqrtf(2.0f / S) * beta;
    const float sqrt_2xSxbeta = sqrtf(2.0f * S) * beta;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float xi = x[i], yi = y[i], zi = z[i];
        const int *restrict neigh_i = neighbors + (size_t)i * (size_t)n;
        const int count_i = neighbor_count[i];
        for (int jj = 0; jj < count_i; ++jj) {
            const int j = neigh_i[jj];
            if (j <= i) continue;
            const size_t ij = (size_t)i * (size_t)n + (size_t)j;
            const float fij = cutoff[ij], rij = dist[ij], inv_rij = inv_dist[ij];
            if (fij <= 0.0f || inv_rij == 0.0f) continue;
            const float rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
            float Bij = 1.0f, Bji = 1.0f;
            for (int kk = 0; kk < count_i; ++kk) {
                const int k = neigh_i[kk];
                if (k == j) continue;
                const size_t ik = (size_t)i * (size_t)n + (size_t)k;
                const float inv_rik = inv_dist[ik];
                if (inv_rik == 0.0f) continue;
                const float dx = x[k] - xi, dy = y[k] - yi, dz = z[k] - zi;
                const float cos_ijk = -(rx*dx + ry*dy + rz*dz) * inv_rik * inv_rij;
                Bij += angular_g(cos_ijk) * cutoff[ik];
            }
            const int *restrict neigh_j = neighbors + (size_t)j * (size_t)n;
            const int count_j = neighbor_count[j];
            for (int kk = 0; kk < count_j; ++kk) {
                const int k = neigh_j[kk];
                if (k == i) continue;
                const size_t jk = (size_t)j * (size_t)n + (size_t)k;
                const float inv_rjk = inv_dist[jk];
                if (inv_rjk == 0.0f) continue;
                const float dx = x[k] - x[j], dy = y[k] - y[j], dz = z[k] - z[j];
                const float cos_jik = (rx*dx + ry*dy + rz*dz) * inv_rjk * inv_rij;
                Bji += angular_g(cos_jik) * cutoff[jk];
            }
            const float B_ij = 0.5f * (powf(Bij, -delta) + powf(Bji, -delta));
            const float rij_Re = rij - Re;
            const float Vr = expf(-sqrt_2xSxbeta * rij_Re) * De_S_1 * fij;
            const float Va = expf(-sqrt_2_Sxbeta * rij_Re) * DexS_S_1 * fij;
            s += Vr - B_ij * Va;
        }
    }
    return s;
}

long gettime(){ struct timeval currentTime; gettimeofday(&currentTime, NULL); return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec; }

typedef struct {
    size_t swarm_size, input_dim;
    float upper_bound, lower_bound;
    float **swarm, **vel;
    float global_best;
    float *Gbest_vector, *personal_best;
    float **Pbest_vector;
    float (*f)(float*);
} Swarm;

static inline uint32_t rng_next(uint32_t* state) { uint32_t x=*state; x^=x<<13; x^=x>>17; x^=x<<5; *state=x?x:0x9e3779b9u; return *state; }
static inline float rng_unit(uint32_t* state) { return (float)(rng_next(state) >> 8) * (1.0f / 16777216.0f); }
static inline float clampf_fast(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

Swarm init(float (*optim_func)(float*), size_t swarm_size, size_t input_vec_dim, float upper_bound, float lower_bound){
    float** swarm = (float**)calloc(swarm_size, sizeof(float*));
    float** velocity = (float**)calloc(swarm_size, sizeof(float*));
    float** Pbest_vector = (float**)calloc(swarm_size, sizeof(float*));
    float* personal_best = aligned_calloc_float(swarm_size);
    float* Gbest_vector = aligned_calloc_float(input_vec_dim);
    if(!swarm || !velocity || !Pbest_vector){ fprintf(stderr,"allocation failed\n"); exit(1); }
    const uint32_t base_seed = (uint32_t)time(NULL) ^ 0xa5a5a5a5u;
    srand(base_seed);
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for(int i=0; i<(int)swarm_size; ++i){
        uint32_t seed = base_seed ^ (uint32_t)(i + 1) * 0x9e3779b9u;
        swarm[i] = aligned_calloc_float(input_vec_dim);
        velocity[i] = aligned_calloc_float(input_vec_dim);
        Pbest_vector[i] = aligned_calloc_float(input_vec_dim);
        for(size_t j=0; j<input_vec_dim; ++j){
            swarm[i][j] = lower_bound + rng_unit(&seed) * (upper_bound - lower_bound);
            Pbest_vector[i][j] = swarm[i][j];
        }
        personal_best[i] = optim_func(swarm[i]);
    }
    float global_best = INFINITY;
    size_t best_index = 0;
    for(size_t i=0; i<swarm_size; ++i) if(personal_best[i] < global_best){ global_best = personal_best[i]; best_index = i; }
    memcpy(Gbest_vector, swarm[best_index], input_vec_dim * sizeof(float));
    Swarm s;
    s.input_dim=input_vec_dim; s.swarm_size=swarm_size; s.swarm=swarm; s.Pbest_vector=Pbest_vector;
    s.upper_bound=upper_bound; s.lower_bound=lower_bound; s.vel=velocity; s.global_best=global_best;
    s.personal_best=personal_best; s.f=optim_func; s.Gbest_vector=Gbest_vector;
    return s;
}

void optimize(Swarm* swarm, int n_iter, float c1, float c2, float w, float chi){
    const size_t dim = swarm->input_dim;
    const float delV = swarm->upper_bound - swarm->lower_bound;
    float* gbests = aligned_calloc_float(n_threads);
    float* gbest_vec = aligned_calloc_float((size_t)n_threads * dim);
    for(int iter=0; iter<n_iter; ++iter){
        if(iter%100==0){ printf(" %d of %d with Global best of %f ", iter, n_iter, swarm->global_best); fputc('\r', stdout); }
        const float max_step = delV * (float)(iter + 1) / (float)n_iter;
        const float u1 = ((float)rand()) / (float)RAND_MAX, u2 = ((float)rand()) / (float)RAND_MAX;
        const float c1u1 = c1 * u1, c2u2 = c2 * u2;
        for(int t=0; t<n_threads; ++t) gbests[t] = INFINITY;
#pragma omp parallel num_threads(n_threads)
        {
            const int tid = omp_get_thread_num();
            float local_best = INFINITY;
            float* restrict local_best_vec = gbest_vec + (size_t)tid * dim;
#pragma omp for schedule(static)
            for(int i=0; i<(int)swarm->swarm_size; ++i){
                float *restrict pos = swarm->swarm[i], *restrict vel = swarm->vel[i];
                const float *restrict pbest = swarm->Pbest_vector[i], *restrict gbest = swarm->Gbest_vector;
                for(size_t j=0; j<dim; ++j){
                    float v = chi * (vel[j] * w + c1u1 * (pbest[j] - pos[j]) + c2u2 * (gbest[j] - pos[j]));
                    v = clampf_fast(v, -max_step, max_step);
                    vel[j] = v;
                    pos[j] = clampf_fast(pos[j] + v, swarm->lower_bound, swarm->upper_bound);
                }
                const float val = swarm->f(pos);
                if(val < swarm->personal_best[i]){
                    swarm->personal_best[i] = val;
                    memcpy(swarm->Pbest_vector[i], pos, dim * sizeof(float));
                    if(val < local_best){ local_best = val; memcpy(local_best_vec, pos, dim * sizeof(float)); }
                }
            }
            gbests[tid] = local_best;
        }
        int best_thread = -1;
        float best_value = swarm->global_best;
        for(int t=0; t<n_threads; ++t) if(gbests[t] < best_value){ best_value = gbests[t]; best_thread = t; }
        if(best_thread >= 0){ swarm->global_best = best_value; memcpy(swarm->Gbest_vector, gbest_vec + (size_t)best_thread * dim, dim * sizeof(float)); }
    }
    aligned_free_bytes(gbests); aligned_free_bytes(gbest_vec);
}

void save(Swarm swarm){
    char str[128];
    snprintf(str, sizeof(str), "opt/OptimizedSwarm_S%d_%f.bin", n, swarm.global_best);
    printf("Swarm saved to file :%s", str);
    FILE* file = fopen(str,"wb");
    if(file==NULL){ printf("ERROR in opening file"); exit(1); }
    size_t nwritten = fwrite(&swarm, sizeof(swarm), 1, file);
    fclose(file);
    if(nwritten<1){ printf("ERROR in writing to file"); exit(1); }
}

Swarm open(char* filename){
    FILE* file = fopen(filename,"rb");
    if(file==NULL){ printf("ERROR in opening file"); exit(1); }
    Swarm swarm;
    size_t nread = fread(&swarm,sizeof(swarm),1,file);
    (void)nread;
    fclose(file);
    return swarm;
}

int get_n(){ return (int)n; }

float* GetGlobalBestCandidate(float c1, float c2, float w, float chi, size_t SwarmSize, float upper, float lower, int n_iter){
    printf("%f %f %f %f %zu  %f %f %d\n", c1, c2, w, chi, SwarmSize, upper, lower, n_iter);
    size_t dim = (size_t)n * 3u;
    Swarm swarm = init(BrennerPotential, SwarmSize, dim, upper, lower);
    printf("initial global minima start :%f \n", swarm.global_best);
    float t1 = (float)gettime();
    optimize(&swarm,n_iter,c1,c2,w,chi);
    float t2 = (float)gettime();
    printf("final best %f \n",swarm.global_best);
    printf("\n total time elapsed %f ms\n",(double)(t2-t1)/1000.0);
    return swarm.Gbest_vector;
}
