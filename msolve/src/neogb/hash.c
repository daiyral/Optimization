/* This file is part of msolve.
 *
 * msolve is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * msolve is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with msolve.  If not, see <https://www.gnu.org/licenses/>
 *
 * Authors:
 * Jérémy Berthomieu
 * Christian Eder
 * Mohab Safey El Din */


#include "hash.h"

/* Performance optimization: Add branch prediction hints */
#ifdef __GNUC__
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define RESTRICT       __restrict__
#define PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
#define PURE           __attribute__((pure))
#define HOT            __attribute__((hot))
#define INLINE         __attribute__((always_inline)) inline
#else
#define likely(x)      (x)
#define unlikely(x)    (x)
#define RESTRICT
#define PREFETCH(addr, rw, locality) ((void)0)
#define PURE
#define HOT
#define INLINE         inline
#endif

/* Performance optimization: Memory alignment for better cache performance */
#ifdef __GNUC__
#define CACHE_ALIGNED __attribute__((aligned(64)))
#else
#define CACHE_ALIGNED
#endif

/* Performance optimization: SIMD includes */
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define USE_SSE4 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#endif

/* Performance optimization: Fast power of 2 using bit shifts instead of pow() */
static INLINE uint32_t fast_pow2(uint32_t n) {
    return (n < 32) ? (1U << n) : ((uint32_t)1U << 31);
}

/* Performance optimization: Fast log2 for hash table sizing */
static INLINE uint32_t fast_log2(uint32_t x) {
#ifdef __GNUC__
    return (x == 0) ? 0 : (31U - (uint32_t)__builtin_clz(x));
#else
    uint32_t result = 0;
    while (x >>= 1) result++;
    return result;
#endif
}

/* Performance optimization: Optimized modulo for power of 2 */
static INLINE hi_t fast_mod_pow2(hi_t x, hi_t mask) {
    return x & mask;
}

/* we have three different hash tables:
 * 1. one hash table for elements in the basis (bht)
 * 2. one hash table for the spairs during the update process (uht)
 * 3. one hash table for the multiplied elements during symbolic
 *    preprocessing (sht) */

/* The idea of the structure of the hash table is taken from an
 * implementation by Roman Pearce and Michael Monagan in Maple. */

/* Performance optimization: Improved pseudo random number generator with better distribution */
static INLINE val_t pseudo_random_number_generator(uint32_t * RESTRICT seed) {
    uint32_t rseed = *seed;
    /* Xorshift32 algorithm - faster and better distribution than original */
    rseed ^= rseed << 13;
    rseed ^= rseed >> 17;
    rseed ^= rseed << 5;
    *seed = rseed;
    return (val_t)rseed;
}

/* Performance optimization: SIMD-optimized hash computation */
static HOT INLINE val_t compute_hash_value_simd(
    const exp_t * RESTRICT a,
    const val_t * RESTRICT rn,
    const len_t evl
) {
    val_t hash = 0;
    len_t i = 0;

#if defined(USE_AVX2) && defined(__x86_64__)
    /* AVX2 vectorized hash computation for x86_64 */
    __m256i hash_vec = _mm256_setzero_si256();
    const len_t simd_end = evl & ~7U; /* Process 8 elements at a time */
    
    for (; i < simd_end; i += 8) {
        PREFETCH(&a[i + 8], 0, 3);
        PREFETCH(&rn[i + 8], 0, 3);
        
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i rn_vec = _mm256_loadu_si256((__m256i*)&rn[i]);
        __m256i prod = _mm256_mullo_epi32(a_vec, rn_vec);
        hash_vec = _mm256_add_epi32(hash_vec, prod);
    }
    
    /* Horizontal sum of AVX2 vector */
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(hash_vec), 
                                   _mm256_extracti128_si256(hash_vec, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    hash = _mm_cvtsi128_si32(sum128);
    
#elif defined(USE_SSE4)
    /* SSE4 vectorized hash computation */
    __m128i hash_vec = _mm_setzero_si128();
    const len_t simd_end = evl & ~3U; /* Process 4 elements at a time */
    
    for (; i < simd_end; i += 4) {
        PREFETCH(&a[i + 4], 0, 3);
        PREFETCH(&rn[i + 4], 0, 3);
        
        __m128i a_vec = _mm_loadu_si128((__m128i*)&a[i]);
        __m128i rn_vec = _mm_loadu_si128((__m128i*)&rn[i]);
        __m128i prod = _mm_mullo_epi32(a_vec, rn_vec);
        hash_vec = _mm_add_epi32(hash_vec, prod);
    }
    
    /* Horizontal sum of SSE vector */
    hash_vec = _mm_hadd_epi32(hash_vec, hash_vec);
    hash_vec = _mm_hadd_epi32(hash_vec, hash_vec);
    hash = _mm_cvtsi128_si32(hash_vec);
    
#elif defined(USE_NEON)
    /* NEON vectorized hash computation for ARM */
    uint32x4_t hash_vec = vdupq_n_u32(0);
    const len_t simd_end = evl & ~3U; /* Process 4 elements at a time */
    
    for (; i < simd_end; i += 4) {
        __builtin_prefetch(&a[i + 4], 0, 3);
        __builtin_prefetch(&rn[i + 4], 0, 3);
        
        uint32x4_t a_vec = vld1q_u32((uint32_t*)&a[i]);
        uint32x4_t rn_vec = vld1q_u32((uint32_t*)&rn[i]);
        uint32x4_t prod = vmulq_u32(a_vec, rn_vec);
        hash_vec = vaddq_u32(hash_vec, prod);
    }
    
    /* Horizontal sum of NEON vector */
    uint32x2_t sum = vadd_u32(vget_low_u32(hash_vec), vget_high_u32(hash_vec));
    sum = vpadd_u32(sum, sum);
    hash = vget_lane_u32(sum, 0);
    
#else
    /* Scalar fallback with manual unrolling */
    const len_t unroll_end = evl & ~3U;
    for (; i < unroll_end; i += 4) {
        PREFETCH(&a[i + 4], 0, 3);
        PREFETCH(&rn[i + 4], 0, 3);
        
        hash += ((val_t)a[i] * rn[i]) + 
                ((val_t)a[i+1] * rn[i+1]) + 
                ((val_t)a[i+2] * rn[i+2]) + 
                ((val_t)a[i+3] * rn[i+3]);
    }
#endif
    
    /* Handle remaining elements */
    for (; i < evl; ++i) {
        hash += (val_t)a[i] * rn[i];
    }
    
    return hash;
}

/* Performance optimization: Optimized memory allocation with alignment */
static INLINE void* aligned_malloc(size_t size, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__GNUC__)
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return malloc(size);
#else
    return malloc(size);
#endif
}

static INLINE void aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

ht_t *initialize_basis_hash_table(
    md_t *st
    )
{
    len_t i;
    hl_t j;

    const len_t nv  = st->nvars;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t));
    if (unlikely(ht == NULL)) {
        fprintf(stderr, "Failed to allocate hash table\n");
        return NULL;
    }
    
    ht->nv    = nv;
    /* generate map */
    ht->bpv = (len_t)((CHAR_BIT * sizeof(sdm_t)) / (unsigned long)nv);
    if (ht->bpv == 0) {
        ht->bpv++;
    }
    ht->ndv = (unsigned long)nv < (CHAR_BIT * sizeof(sdm_t)) ?
        nv : (len_t)((CHAR_BIT * sizeof(sdm_t)));
    ht->dv  = (len_t *)calloc((unsigned long)ht->ndv, sizeof(len_t));

    /* Performance optimization: Use fast_pow2 instead of pow() */
    ht->hsz   = (hl_t)fast_pow2(st->init_hts);
    ht->esz   = ht->hsz / 2;
    
    /* Performance optimization: Use aligned allocation for better cache performance */
    ht->hmap = (hi_t *)aligned_malloc(ht->hsz * sizeof(hi_t), 64);
    if (unlikely(ht->hmap == NULL)) {
        ht->hmap = (hi_t *)calloc(ht->hsz, sizeof(hi_t));
    } else {
        memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));
    }

    if (st->nev == 0) {
        ht->evl = nv + 1; /* store also degree at first position */
        ht->ebl = 0;
        for (i = 1; i <= ht->ndv; ++i) {
            ht->dv[i-1] = i;
        }
    } else {
        ht->evl = nv + 2; /* store also degrees for both blocks, see
                           * data.h for more on exponent vector structure */
        ht->ebl = st->nev + 1; /* store also degree at first position */
        if (st->nev >= ht->ndv) {
            for (i = 1; i <= ht->ndv; ++i) {
                ht->dv[i-1] = i;
            }
        } else {
            len_t ctr = 0;
            for (i = 1; i <= st->nev; ++i) {
                ht->dv[ctr++] = i;
            }
            for (i = ht->ebl+1; i < ht->ndv+2; ++i) {
                ht->dv[ctr++] = i;
            }
        }

    }
    /* generate divmask map */
    ht->dm  = (sdm_t *)calloc(
            (unsigned long)(ht->ndv * ht->bpv), sizeof(sdm_t));

    /* generate random values */
    ht->rsd = 2463534242;
    ht->rn  = (val_t *)aligned_malloc((unsigned long)ht->evl * sizeof(val_t), 64);
    if (ht->rn == NULL) {
        ht->rn = (val_t *)calloc((unsigned long)ht->evl, sizeof(val_t));
    } else {
        memset(ht->rn, 0, (unsigned long)ht->evl * sizeof(val_t));
    }
    
    for (i = ht->evl; i > 0; --i) {
        /* random values should not be zero */
        ht->rn[i-1] = pseudo_random_number_generator(&(ht->rsd)) | 1;
    }
    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->eld = 1;
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (unlikely(ht->ev == NULL)) {
        fprintf(stderr, "Computation needs too much memory on this machine,\n");
        fprintf(stderr, "could not initialize exponent vector for hash table,\n");
        fprintf(stderr, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    
    /* Performance optimization: Align memory for better cache performance */
    exp_t *tmp = (exp_t *)aligned_malloc((unsigned long)ht->evl * ht->esz * sizeof(exp_t), 64);
    if (unlikely(tmp == NULL)) {
        tmp = (exp_t *)malloc((unsigned long)ht->evl * ht->esz * sizeof(exp_t));
        if (unlikely(tmp == NULL)) {
            fprintf(stderr, "Exponent storage needs too much memory on this machine,\n");
            fprintf(stderr, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
            fprintf(stderr, "segmentation fault will follow.\n");
        }
    }
    
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    st->max_bht_size  = ht->esz;
    return ht;
}

ht_t *copy_hash_table(
    const ht_t *bht
    )
{
    hl_t j;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t));

    ht->nv    = bht->nv;
    ht->evl   = bht->evl;
    ht->ebl   = bht->ebl;
    ht->hsz   = bht->hsz;
    ht->esz   = bht->esz;

    ht->hmap  = calloc(ht->hsz, sizeof(hi_t));
    memcpy(ht->hmap, bht->hmap, (unsigned long)ht->hsz * sizeof(hi_t));

    ht->ndv = bht->ndv;
    ht->bpv = bht->bpv;
    ht->dm  = bht->dm;
    ht->rn  = bht->rn;

    ht->dv  = (len_t *)calloc((unsigned long)ht->ndv, sizeof(len_t));
    memcpy(ht->dv, bht->dv, (unsigned long)ht->ndv * sizeof(len_t));

    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));

    memcpy(ht->hd, bht->hd, (unsigned long)ht->esz * sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(stderr, "Computation needs too much memory on this machine,\n");
        fprintf(stderr, "could not initialize exponent vector for hash table,\n");
        fprintf(stderr, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(stderr, "Exponent storage needs too much memory on this machine,\n");
        fprintf(stderr, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
        fprintf(stderr, "segmentation fault will follow.\n");
    }
    memcpy(tmp, bht->ev[0], (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    ht->eld = bht->eld;
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    return ht;
}


ht_t *initialize_secondary_hash_table(
    const ht_t * const bht,
    const md_t * const md
    )
{
    hl_t j;

    ht_t *ht  = (ht_t *)malloc(sizeof(ht_t)); 
    ht->nv    = bht->nv;
    ht->evl   = bht->evl;
    ht->ebl   = bht->ebl;

    /* generate map */
    int32_t min = 3 > md->init_hts-5 ? 3 : md->init_hts-5;
    ht->hsz   = (hl_t)fast_pow2(min);
    ht->esz   = ht->hsz / 2;
    ht->hmap  = calloc(ht->hsz, sizeof(hi_t));

    /* divisor mask and random number seeds from basis hash table */
    ht->ndv = bht->ndv;
    ht->bpv = bht->bpv;
    ht->dm  = bht->dm;
    ht->rn  = bht->rn;
    ht->dv  = bht->dv;

    /* generate exponent vector */
    /* keep first entry empty for faster divisibility checks */
    ht->eld = 1;
    ht->hd  = (hd_t *)calloc(ht->esz, sizeof(hd_t));
    ht->ev  = (exp_t **)malloc(ht->esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(stderr, "Computation needs too much memory on this machine,\n");
        fprintf(stderr, "could not initialize exponent vector for hash table,\n");
        fprintf(stderr, "esz = %lu, segmentation fault will follow.\n", (unsigned long)ht->esz);
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)ht->evl * ht->esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(stderr, "Exponent storage needs too much memory on this machine,\n");
        fprintf(stderr, "initialization failed, esz = %lu,\n", (unsigned long)ht->esz);
        fprintf(stderr, "segmentation fault will follow.\n");
    }
    const hl_t esz  = ht->esz;
    for (j = 0; j < esz; ++j) {
        ht->ev[j]  = tmp + (j*ht->evl);
    }
    return ht;
}

void free_shared_hash_data(
    ht_t *ht
    )
{
    if (ht != NULL) {
        if (ht->rn) {
            aligned_free(ht->rn);
            ht->rn = NULL;
        }
        if (ht->dv) {
            free(ht->dv);
            ht->dv = NULL;
        }
        if (ht->dm) {
            free(ht->dm);
            ht->dm = NULL;
        }
    }
}

void free_hash_table(
    ht_t **htp
    )
{
    ht_t *ht  = *htp;
    if (ht->hmap) {
        aligned_free(ht->hmap);
        ht->hmap = NULL;
    }
    if (ht->hd) {
        free(ht->hd);
        ht->hd  = NULL;
    }
    if (ht->ev) {
        /* note: memory is allocated as one big block,
            *       so freeing ev[0] is enough */
        aligned_free(ht->ev[0]);
        free(ht->ev);
        ht->ev  = NULL;
    }
    if (ht->rn) {
        aligned_free(ht->rn);
        ht->rn = NULL;
    }
    free(ht);
    ht    = NULL;
    *htp  = ht;
}

void full_free_hash_table(
                     ht_t **htp
                     )
{
  ht_t *ht  = *htp;
  if (ht->hmap) {
    aligned_free(ht->hmap);
    ht->hmap = NULL;
  }
  if (ht->hd) {
    free(ht->hd);
    ht->hd  = NULL;
  }
  if (ht->ev) {
    /* note: memory is allocated as one big block,
     *       so freeing ev[0] is enough */
    aligned_free(ht->ev[0]);
    free(ht->ev);
    ht->ev  = NULL;
  }
  if (ht != NULL) {
    if (ht->rn) {
      aligned_free(ht->rn);
      ht->rn = NULL;
    }
    if (ht->dv) {
      free(ht->dv);
      ht->dv = NULL;
    }
    if (ht->dm) {
      free(ht->dm);
      ht->dm = NULL;
    }
  }
  free(ht);
  ht    = NULL;
  *htp  = ht;
}

/* we just double the hash table size */
static void enlarge_hash_table(
    ht_t *ht
    )
{
    hl_t i, j;
    val_t h, k;

    ht->esz = 2 * ht->esz;
    const hl_t esz  = ht->esz;
    const hi_t eld  = ht->eld;

    ht->hd    = realloc(ht->hd, esz * sizeof(hd_t));
    memset(ht->hd+eld, 0, (esz-eld) * sizeof(hd_t));
    ht->ev    = realloc(ht->ev, esz * sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(stderr, "Enlarging hash table failed for esz = %lu,\n", (unsigned long)esz);
        fprintf(stderr, "segmentation fault will follow.\n");
    }
    /* note: memory is allocated as one big block, so reallocating
     *       memory from ev[0] is enough    */
    ht->ev[0] = realloc(ht->ev[0],
            esz * (unsigned long)ht->evl * sizeof(exp_t));
    if (ht->ev[0] == NULL) {
        fprintf(stderr, "Enlarging exponent vector for hash table failed\n");
        fprintf(stderr, "for esz = %lu, segmentation fault will follow.\n", (unsigned long)esz);
    }
    /* due to realloc we have to reset ALL ev entries,
     * memory might have been moved */
    for (i = 1; i < esz; ++i) {
        ht->ev[i] = ht->ev[0] + (i*ht->evl);
    }

    /* The hash table should be double the size of the exponent space in
     * order to never get a fill in over 50%. If the exponent size is now
     * enlarge to 2^31 elements that's the limit we can go. Thus we cannot
     * enlarge the hash table size any further and have to live with more
     * than 50% fill in. */
    if (ht->hsz < fast_pow2(32)) {
        ht->hsz = 2 * ht->hsz;
        const hl_t hsz  = ht->hsz;
        
        /* Performance optimization: Use aligned realloc */
        hi_t *new_hmap = (hi_t *)aligned_malloc(hsz * sizeof(hi_t), 64);
        if (new_hmap == NULL) {
            new_hmap = (hi_t *)realloc(ht->hmap, hsz * sizeof(hi_t));
        } else {
            aligned_free(ht->hmap);
        }
        ht->hmap = new_hmap;
        
        if (ht->hmap == NULL) {
            fprintf(stderr, "Enlarging hash table failed for hsz = %lu,\n", (unsigned long)hsz);
            fprintf(stderr, "segmentation fault will follow.\n");
        }
        memset(ht->hmap, 0, hsz * sizeof(hi_t));
        const hi_t mod = (hi_t)(hsz-1);

        /* Performance optimization: Reinsert with better cache locality */
        for (i = 1; i < eld; ++i) {
            h = ht->hd[i].val;
            k = fast_mod_pow2(h, mod);
            
            /* Linear probing with better cache behavior */
            for (j = 0; j < hsz; ++j) {
                const hi_t pos = fast_mod_pow2(k + j, mod);
                if (likely(!ht->hmap[pos])) {
                    ht->hmap[pos] = i;
                    break;
                }
            }
        }
    } else {
        if (ht->hsz == fast_pow2(32)) {
          printf("Exponent space is now 2^32 elements wide, we cannot\n");
          printf("enlarge the hash table any further, thus fill in gets\n");
          printf("over 50%% and performance of hashing may get worse.\n");
        } else {
          printf("Hash table is full, we can no longer enlarge\n");
          printf("Segmentation fault will follow.\n");
          aligned_free(ht->hmap);
          ht->hmap  = NULL;
        }
    }
}

static inline sdm_t generate_short_divmask(
    const exp_t * const a,
    const ht_t *ht
    )
{
  len_t i, j;
  int32_t res = 0;
  int32_t ctr = 0;
  const len_t ndv         = ht->ndv;
  const len_t * const dv  = ht->dv;
  const len_t bpv         = ht->bpv;

  for (i = 0; i < ndv; ++i) {
    for (j = 0; j < bpv; ++j) {
      if ((sdm_t)a[dv[i]] >= ht->dm[ctr]) {
        res |= 1 << ctr;
      }
      ctr++;
    }
  }
 
  return res;
}

/* note: we calculate the divmask after reading in the input generators. thoseV
 * are first stored in the local hash table. thus we use the local exponents to
 * generate the divmask */
void calculate_divmask(
    ht_t *ht
    )
{
  hi_t i;
  hl_t k;
  len_t j, steps;
  int32_t ctr = 0;
  const len_t * const dv  = ht->dv;
  exp_t **ev  = ht->ev;

  deg_t *max_exp  = (deg_t *)malloc((unsigned long)ht->ndv * sizeof(deg_t));
  deg_t *min_exp  = (deg_t *)malloc((unsigned long)ht->ndv * sizeof(deg_t));

  exp_t *e  = ev[1];

  /* get initial values from first hash table entry */
  for (i = 0; i < ht->ndv; ++i) {
    max_exp[i]  = min_exp[i]  = e[dv[i]];
  }

  /* get maximal and minimal exponent element entries in hash table */
  for (i = 2; i < ht->eld; ++i) {
    e = ev[i];
    for (j = 0; j < ht->ndv; ++j) {
      if (e[dv[j]] > max_exp[j]) {
        max_exp[j]  = e[dv[j]];
        continue;
      }
      if (e[dv[j]] < min_exp[j]) {
        min_exp[j]  = e[dv[j]];
      }
    }
  }

  /* calculate average values for generating divmasks */
  for (i = 0; i < ht->ndv; ++i) {
    steps = (max_exp[i] - min_exp[i]) / ht->bpv;
    if (steps == 0)
      steps++;
    for (j = 0; j < ht->bpv; ++j) {
      ht->dm[ctr++] = (sdm_t)steps++;
    }
  }

  /* initialize divmasks for elements already added to hash table */
  for (k = 1; k < ht->eld; k++) {
    ht->hd[k].sdm = generate_short_divmask(ev[k], ht);
  }

  free(max_exp);
  free(min_exp);
}

/* returns zero if a is not divisible by b, else 1 is returned */
static inline hi_t check_monomial_division(
    const hi_t a,
    const hi_t b,
    const ht_t *ht
    )
{
  len_t i;

  /* short divisor mask check */
  if (ht->hd[b].sdm & ~ht->hd[a].sdm) {
    return 0;
  }

  const len_t evl = ht->evl;

  const exp_t *const ea = ht->ev[a];
  const exp_t *const eb = ht->ev[b];

  /* printf("! no sdm decision !\n"); */
      /* Performance optimization: Fast divisibility check */
    return fast_divisibility_check(ea, eb, evl);
}

static inline void check_monomial_division_in_update(
    hi_t *a,
    const len_t start,
    const len_t end,
    const hi_t b,
    const ht_t *ht
    )
{
    len_t i, j;
    const len_t evl = ht->evl;

    const sdm_t sb        = ht->hd[b].sdm;
    const exp_t *const eb = ht->ev[b];
    /* pairs are sorted, we only have to search entries
     * above the starting point */
        j = start+1;
restart:
    for (; j < end; ++j) {
        if (a[j] == 0) {
            continue;
        }
        /* short divisor mask check */
        if (~ht->hd[a[j]].sdm & sb) {
            continue;
        }
        const exp_t *const ea = ht->ev[a[j]];
        /* Performance optimization: Fast divisibility check */
        if (!fast_divisibility_check(ea, eb, evl)) {
            j++;
            goto restart;
        }
        a[j]  = 0;
    }
}

static inline hi_t check_lm_divisibility_and_insert_in_hash_table(
    const exp_t *a,
    ht_t *ht,
    const bs_t * const bs
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    hd_t *d;
    const len_t lml   = bs->lml;

    const sdm_t * const lms = bs->lm;
    const bl_t * const lmps = bs->lmps;

    const sdm_t nsdm  = ~generate_short_divmask(a, ht);

    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz  = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(ht->hsz - 1);

    /* check divisibility w.r.t. current lead monomials */
    i = 0;
start:
    while (i < lml && lms[i] & nsdm) {
        i++;
    }
    if (i < lml) {
        e = ht->ev[bs->hm[lmps[i]][OFFSET]];
        for (j = 0; j < evl; ++j) {
            if (e[j] > a[j]) {
                i++;
                goto start;
            }
        }
        /* divisible by lm */
        return 0;
    }
    /* if we are here then a is not divisible by a current
     * lead monomial and we can add it to the hash table */

    /* Performance optimization: Use SIMD hash computation */
    h = compute_hash_value_simd(a, ht->rn, evl);
    /* probing */
    k = h;
    i = 0;
restart:
    for (; i < hsz; ++i) {
        k = (hi_t)((k+i) & mod);
        const hi_t hm = ht->hmap[k];
        if (!hm) {
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        const exp_t * const ehm = ht->ev[hm];
        for (j = 0; j < evl-1; j += 2) {
            if (a[j] != ehm[j] || a[j+1] != ehm[j+1]) {
                i++;
                goto restart;
            }
        }
        if (a[evl-1] != ehm[evl-1]) {
            i++;
            goto restart;
        }
        return hm;
    }

    /* add element to hash table */
    ht->hmap[k]  = pos = (hi_t)ht->eld;
    e   = ht->ev[pos];
    d   = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm  =   generate_short_divmask(e, ht);
    d->deg  =   e[0];
    d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val  =   h;

    ht->eld++;

    return pos;
}

static HOT inline hi_t insert_multiplied_signature_in_hash_table(
    const hm_t h1,
    const hm_t h2,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    exp_t * RESTRICT a = ht->ev[0];
    hd_t *d;
    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz = ht->hsz;
    const hi_t mod = (hi_t)(hsz - 1);

    h = h1 + h2;

    /* Performance optimization: Vectorized exponent vector generation */
    const exp_t * RESTRICT ev1 = ht->ev[h1];
    const exp_t * RESTRICT ev2 = ht->ev[h2];
    
#if defined(USE_AVX2)
    /* AVX2 vectorized addition */
    const len_t simd_end = evl & ~7U;
    for (j = 0; j < simd_end; j += 8) {
        __m256i a1 = _mm256_loadu_si256((__m256i*)&ev1[j]);
        __m256i a2 = _mm256_loadu_si256((__m256i*)&ev2[j]);
        __m256i sum = _mm256_add_epi32(a1, a2);
        _mm256_storeu_si256((__m256i*)&a[j], sum);
    }
    for (; j < evl; ++j) {
        a[j] = ev1[j] + ev2[j];
    }
#elif defined(USE_SSE4)
    /* SSE4 vectorized addition */
    const len_t simd_end = evl & ~3U;
    for (j = 0; j < simd_end; j += 4) {
        __m128i a1 = _mm_loadu_si128((__m128i*)&ev1[j]);
        __m128i a2 = _mm_loadu_si128((__m128i*)&ev2[j]);
        __m128i sum = _mm_add_epi32(a1, a2);
        _mm_storeu_si128((__m128i*)&a[j], sum);
    }
    for (; j < evl; ++j) {
        a[j] = ev1[j] + ev2[j];
    }
#else
    /* Unrolled scalar addition */
    const len_t unroll_end = evl & ~3U;
    for (j = 0; j < unroll_end; j += 4) {
        a[j] = ev1[j] + ev2[j];
        a[j+1] = ev1[j+1] + ev2[j+1];
        a[j+2] = ev1[j+2] + ev2[j+2];
        a[j+3] = ev1[j+3] + ev2[j+3];
    }
    for (; j < evl; ++j) {
        a[j] = ev1[j] + ev2[j];
    }
#endif

    /* Performance optimization: Use optimized lookup */
    const int lookup_result = optimized_hash_lookup(a, h, ht, &k);
    if (lookup_result == 1) {
        return k; /* Found existing */
    } else if (lookup_result == 0) {
        /* Not found, k contains insertion position */
        goto insert_element;
    }
    
    /* Fallback to linear probing */
    k = fast_mod_pow2(h, mod);
    for (i = 0; i < hsz; ++i) {
        const hi_t pos_probe = fast_mod_pow2(k + i, mod);
        const hi_t hm = ht->hmap[pos_probe];
        if (!hm) {
            k = pos_probe;
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        if (fast_exponent_compare(a, ht->ev[hm], evl)) {
            return hm;
        }
    }

insert_element:
    /* add element to hash table */
    ht->hmap[k] = pos = (hi_t)ht->eld;
    e = ht->ev[pos];
    d = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm = generate_short_divmask(e, ht);
    d->deg = e[0];
    d->deg += ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val = h;

    ht->eld++;

    return pos;
}

/* Performance optimization: Fast comparison with prefetching */
static HOT INLINE int fast_exponent_compare(
    const exp_t * RESTRICT a,
    const exp_t * RESTRICT b,
    const len_t evl
) {
    len_t i = 0;
    
#if defined(USE_AVX2)
    /* AVX2 vectorized comparison */
    const len_t simd_end = evl & ~7U;
    for (; i < simd_end; i += 8) {
        PREFETCH(&a[i + 8], 0, 3);
        PREFETCH(&b[i + 8], 0, 3);
        
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i b_vec = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i cmp = _mm256_cmpeq_epi32(a_vec, b_vec);
        
        if (_mm256_movemask_epi8(cmp) != 0xFFFFFFFF) {
            /* Not all equal, do scalar comparison for this chunk */
            for (len_t j = i; j < i + 8 && j < evl; ++j) {
                if (a[j] != b[j]) return 0;
            }
        }
    }
#elif defined(USE_SSE4)
    /* SSE4 vectorized comparison */
    const len_t simd_end = evl & ~3U;
    for (; i < simd_end; i += 4) {
        PREFETCH(&a[i + 4], 0, 3);
        PREFETCH(&b[i + 4], 0, 3);
        
        __m128i a_vec = _mm_loadu_si128((__m128i*)&a[i]);
        __m128i b_vec = _mm_loadu_si128((__m128i*)&b[i]);
        __m128i cmp = _mm_cmpeq_epi32(a_vec, b_vec);
        
        if (_mm_movemask_epi8(cmp) != 0xFFFF) {
            for (len_t j = i; j < i + 4 && j < evl; ++j) {
                if (a[j] != b[j]) return 0;
            }
        }
    }
#else
    /* Unrolled scalar comparison */
    const len_t unroll_end = evl & ~3U;
    for (; i < unroll_end; i += 4) {
        PREFETCH(&a[i + 4], 0, 3);
        PREFETCH(&b[i + 4], 0, 3);
        
        if (unlikely(a[i] != b[i] || a[i+1] != b[i+1] || 
                     a[i+2] != b[i+2] || a[i+3] != b[i+3])) {
            return 0;
        }
    }
#endif
    
    /* Handle remaining elements */
    for (; i < evl; ++i) {
        if (unlikely(a[i] != b[i])) return 0;
    }
    
    return 1;
}

/* Performance optimization: Fast divisibility check using SIMD */
static HOT INLINE int fast_divisibility_check(
    const exp_t * RESTRICT ea,
    const exp_t * RESTRICT eb,
    const len_t evl
) {
    len_t i = 0;
    
#if defined(USE_AVX2)
    /* AVX2 vectorized divisibility check */
    const len_t simd_end = evl & ~7U;
    for (; i < simd_end; i += 8) {
        PREFETCH(&ea[i + 8], 0, 3);
        PREFETCH(&eb[i + 8], 0, 3);
        
        __m256i ea_vec = _mm256_loadu_si256((__m256i*)&ea[i]);
        __m256i eb_vec = _mm256_loadu_si256((__m256i*)&eb[i]);
        __m256i cmp = _mm256_cmpgt_epi32(eb_vec, ea_vec);
        
        if (_mm256_movemask_epi8(cmp) != 0) {
            return 0; /* eb[i] > ea[i] for some i, not divisible */
        }
    }
#elif defined(USE_SSE4)
    /* SSE4 vectorized divisibility check */
    const len_t simd_end = evl & ~3U;
    for (; i < simd_end; i += 4) {
        PREFETCH(&ea[i + 4], 0, 3);
        PREFETCH(&eb[i + 4], 0, 3);
        
        __m128i ea_vec = _mm_loadu_si128((__m128i*)&ea[i]);
        __m128i eb_vec = _mm_loadu_si128((__m128i*)&eb[i]);
        __m128i cmp = _mm_cmpgt_epi32(eb_vec, ea_vec);
        
        if (_mm_movemask_epi8(cmp) != 0) {
            return 0;
        }
    }
#else
    /* Unrolled scalar divisibility check */
    const len_t unroll_end = evl & ~3U;
    for (; i < unroll_end; i += 4) {
        PREFETCH(&ea[i + 4], 0, 3);
        PREFETCH(&eb[i + 4], 0, 3);
        
        if (unlikely(ea[i] < eb[i] || ea[i+1] < eb[i+1] || 
                     ea[i+2] < eb[i+2] || ea[i+3] < eb[i+3])) {
            return 0;
        }
    }
#endif
    
    /* Handle remaining elements */
    for (; i < evl; ++i) {
        if (unlikely(ea[i] < eb[i])) {
            return 0;
        }
    }
    
    return 1;
}

/* Performance optimization: Optimized hash table lookup */
static HOT INLINE hi_t optimized_hash_lookup(
    const exp_t * RESTRICT a,
    const val_t h,
    const ht_t * RESTRICT ht,
    hi_t * RESTRICT kp
) {
    const hi_t mod = (hi_t)(ht->hsz - 1);
    const len_t evl = ht->evl;
    hi_t k = fast_mod_pow2(h, mod);
    
    /* Performance optimization: Quadratic probing for better distribution */
    for (hl_t i = 0; i < ht->hsz; ++i) {
        const hi_t pos = fast_mod_pow2(k + (i * i), mod);
        const hi_t hm = ht->hmap[pos];
        
        if (likely(!hm)) {
            *kp = pos;
            return 0;
        }
        
        if (likely(ht->hd[hm].val == h)) {
            PREFETCH(ht->ev[hm], 0, 3);
            if (likely(fast_exponent_compare(a, ht->ev[hm], evl))) {
                *kp = hm;
                return 1;
            }
        }
    }
    return -1;
}

/* Performance optimization: Replace original lookup with optimized version */
static inline int32_t is_contained_in_hash_table(
        const exp_t *a,
        const ht_t * const ht,
        const val_t h,
        hi_t *kp
        )
{
    return optimized_hash_lookup(a, h, ht, kp);
}

/* This function assumes that is_contained_in_hash_table() was
 * called beforehand such that the values for h and k are already
 * precomputed. */
static inline len_t add_to_hash_table(
    const exp_t * const a,
    const val_t h,
    const hi_t k,
    ht_t *ht
    )
{
    /* add element to hash table */
    hi_t pos;
    ht->hmap[k] = pos = (hi_t)ht->eld;
    exp_t *e    = ht->ev[pos];
    hd_t *d     = ht->hd + pos;
    memcpy(e, a, (unsigned long)ht->evl * sizeof(exp_t));
    d->sdm  =   generate_short_divmask(e, ht);
    d->deg  =   e[0];
    d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val  =   h;

    ht->eld++;

    return pos;
}

static inline len_t check_insert_in_hash_table(
        const exp_t *a,
        val_t h,
        ht_t *ht
        )
{
    if (h == 0) {
        /* generate hash value */
        for (len_t j = 0; j < ht->evl; ++j) {
            h +=  ht->rn[j] * a[j];
        }
    }

    hi_t k  = 0;

#if 1
    len_t ld = 0;
    while (1) {
        ld = ht->eld;
        if (is_contained_in_hash_table(a, ht, h, &k)) {
            return k;
        } else {
            if (ht->eld == ld) {
#pragma omp critical
                ld = add_to_hash_table(a, h, k, ht);
                return ld;
            }
        }
    }
#else
    return is_contained_in_hash_table(a, ht, h, &k) ?
        k : add_to_hash_table(a, h, k, ht);
#endif
}

static HOT inline hi_t insert_in_hash_table(
    const exp_t *a,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j;
    exp_t *e;
    hd_t *d;
    val_t h = 0;
    const len_t evl = ht->evl;
    const hl_t hsz = ht->hsz;
    const hi_t mod = (hi_t)(hsz - 1);

    /* Performance optimization: Use SIMD hash computation */
    h = compute_hash_value_simd(a, ht->rn, evl);
    
    /* Performance optimization: Use optimized lookup */
    const int lookup_result = optimized_hash_lookup(a, h, ht, &k);
    if (lookup_result == 1) {
        return k; /* Found existing */
    } else if (lookup_result == 0) {
        /* Not found, k contains insertion position */
        goto insert_element;
    }
    
    /* Fallback to linear probing if quadratic failed */
    k = fast_mod_pow2(h, mod);
    for (i = 0; i < hsz; ++i) {
        const hi_t pos = fast_mod_pow2(k + i, mod);
        const hi_t hm = ht->hmap[pos];
        if (!hm) {
            k = pos;
            break;
        }
        if (ht->hd[hm].val != h) {
            continue;
        }
        if (fast_exponent_compare(a, ht->ev[hm], evl)) {
            return hm;
        }
    }

insert_element:
    /* add element to hash table */
    ht->hmap[k] = pos = (hi_t)ht->eld;
    e = ht->ev[pos];
    d = ht->hd + pos;
    memcpy(e, a, (unsigned long)evl * sizeof(exp_t));
    d->sdm = generate_short_divmask(e, ht);
    d->deg = e[0];
    d->deg += ht->ebl > 0 ? e[ht->ebl] : 0;
    d->val = h;

    ht->eld++;

    return pos;
}

static inline void reinitialize_hash_table(
    ht_t *ht,
    const hl_t size
    )
{
    hl_t i;
    /* is there still enough space in the local table? */
    if (size >= (ht->esz)) {
        while (size >= ht->esz) {
            ht->esz = 2 * ht->esz;
            ht->hsz = 2 * ht->hsz;
        }
        const hl_t esz  = ht->esz;
        const hl_t hsz  = ht->hsz;
        const len_t evl = ht->evl;
        ht->hd  = realloc(ht->hd, esz * sizeof(hd_t));
        ht->ev  = realloc(ht->ev, esz * sizeof(exp_t *));
        if (ht->ev == NULL) {
            fprintf(stderr, "Computation needs too much memory on this machine,\n");
            fprintf(stderr, "could not reinitialize exponent vector for hash table,\n");
            fprintf(stderr, "esz = %lu, segmentation fault will follow.\n", (unsigned long)esz);
        }
        /* note: memory is allocated as one big block, so reallocating
         *       memory from evl[0] is enough    */
        ht->ev[0]  = realloc(ht->ev[0],
                esz * (unsigned long)evl * sizeof(exp_t));
        if (ht->ev[0] == NULL) {
            fprintf(stderr, "Exponent storage needs too much memory on this machine,\n");
            fprintf(stderr, "reinitialization failed, esz = %lu\n", (unsigned long)esz);
            fprintf(stderr, "segmentation fault will follow.\n");
        }
        /* due to realloc we have to reset ALL evl entries, memory might be moved */
        for (i = 1; i < esz; ++i) {
            ht->ev[i] = ht->ev[0] + (i*evl);
        }
        ht->hmap  = realloc(ht->hmap, hsz * sizeof(hi_t));
    }
    memset(ht->hd, 0, ht->esz * sizeof(hd_t));
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));

    ht->eld  = 1;
}

static inline void clean_hash_table(
        ht_t *ht
    )
{
    memset(ht->hd, 0, ht->esz * sizeof(hd_t));
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));

    ht->eld  = 1;
}

static inline int prime_monomials(
    const hi_t a,
    const hi_t b,
    const ht_t *ht
    )
{
    len_t i;

    const exp_t * const ea = ht->ev[a];
    const exp_t * const eb = ht->ev[b];

    const len_t evl = ht->evl;
    const len_t ebl = ht->ebl;

    for (i = 1; i < ebl; ++i) {
        if (ea[i] != 0 && eb[i] != 0) {
            return 0;
        }
    }
    for (i = ebl+1; i < evl; ++i) {
        if (ea[i] != 0 && eb[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static inline void insert_plcms_in_basis_hash_table(
    ps_t *psl,
    spair_t *pp,
    ht_t *bht,
    const ht_t *uht,
    const bs_t * const bs,
    const hi_t * const lcms,
    const len_t start,
    const len_t end
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j, l, m;
    hd_t *d;

    spair_t *ps     = psl->p;
    const len_t evl = bht->evl;
    const hl_t hsz  = bht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod = (hi_t)(hsz - 1);
    hm_t * const * const hm = bs->hm;
    m = start;
    l = 0;
letsgo:
    for (; l < end; ++l) {
        if (lcms[l] == 0) {
            continue;
        }
        if (prime_monomials(
                    hm[pp[l].gen1][OFFSET], hm[pp[0].gen2][OFFSET], bht)) {
            continue;
        }
        ps[m] = pp[l];
        const val_t h = uht->hd[lcms[l]].val;
        memcpy(bht->ev[bht->eld], uht->ev[lcms[l]],
                (unsigned long)evl * sizeof(exp_t));
        const exp_t * const n = bht->ev[bht->eld];
        k = h;
        i = 0;
restart:
        for (; i < hsz; ++i) {
            k = (hi_t)(k+i) & mod;
            const hi_t hm = bht->hmap[k];
            if (!hm) {
                break;
            }
            if (bht->hd[hm].val != h) {
                continue;
            }
            const exp_t * const ehm = bht->ev[hm];
            for (j = 0; j < evl-1; j += 2) {
                if (n[j] != ehm[j] || n[j+1] != ehm[j+1]) {
                    i++;
                    goto restart;
                }
            }
            if (n[evl-1] != ehm[evl-1]) {
                i++;
                goto restart;
            }
            ps[m++].lcm = hm;
            l++;
            goto letsgo;
        }

        /* add element to hash table */
        bht->hmap[k] = pos = (hi_t)bht->eld;
        d = bht->hd + bht->eld;
        d->sdm  = uht->hd[lcms[l]].sdm;
        d->deg  = uht->hd[lcms[l]].deg;
        d->val  = h;

        bht->eld++;
        ps[m++].lcm =  pos;
    }
    psl->ld = m;
}

static inline void switch_hcm_data_to_basis_hash_table(
    hi_t *hcm,
    ht_t *bht,
    const mat_t *mat,
    const ht_t * const sht
    )
{
    const len_t start = mat->ncl;
    const len_t end   = mat->nc;

    while (bht->esz - bht->eld < mat->ncr) {
        enlarge_hash_table(bht);
    }

    for (len_t i = start; i < end; ++i) {
#if PARALLEL_HASHING
        hcm[i] = check_insert_in_hash_table(
                sht->ev[hcm[i]], sht->hd[hcm[i]].val, bht);
#else
        hcm[i] = insert_in_hash_table(sht->ev[hcm[i]], bht);
#endif
    }
}

static inline void insert_in_basis_hash_table_pivots(
    hm_t *row,
    ht_t *bht,
    const ht_t * const sht,
    const hi_t * const hcm,
    const md_t * const st
    )
{
    len_t l;

    /* while (bht->esz - bht->eld < row[LENGTH]) {
        enlarge_hash_table(bht);
    } */

    const len_t len = row[LENGTH]+OFFSET;
    const len_t evl = bht->evl;

    const hd_t * const hds    = sht->hd;
    exp_t * const * const evs = sht->ev;
    
    exp_t *evt  = (exp_t *)malloc((unsigned long)evl * sizeof(exp_t));
    for (l = OFFSET; l < len; ++l) {
        memcpy(evt, evs[hcm[row[l]]],
                (unsigned long)evl * sizeof(exp_t));
        row[l] = insert_in_hash_table(evt, bht);
    }
    free(evt);
}

static inline void insert_poly_in_hash_table(
    hm_t *row,
    const hm_t * const b,
    const ht_t * const ht1,
    ht_t *ht2
    )
{
    len_t l;

    const len_t len = b[LENGTH]+OFFSET;

    l = OFFSET;

    for (; l < len; ++l) {
#if PARALLEL_HASHING
        row[l] = check_insert_in_hash_table(ht1->ev[b[l]], ht1->hd[b[l]].val, ht2);
#else
        row[l] = insert_in_hash_table(ht1->ev[b[l]], ht2);
#endif
    }
}

static inline void insert_multiplied_poly_in_hash_table(
    hm_t *row,
    const val_t h1,
    const exp_t * const ea,
    const hm_t * const b,
    const ht_t * const ht1,
    ht_t *ht2
    )
{
    len_t j, l;
    exp_t *n;

    const len_t len = b[LENGTH]+OFFSET;
    const len_t evl = ht1->evl;

    exp_t * const *ev1      = ht1->ev;
    const hd_t * const hd1  = ht1->hd;
    
    exp_t **ev2     = ht2->ev;

    l = OFFSET;

    for (; l < len; ++l) {
        const exp_t * const eb = ev1[b[l]];

        n = ev2[ht2->eld];
        for (j = 0; j < evl; ++j) {
            n[j]  = (exp_t)(ea[j] + eb[j]);
        }

#if PARALLEL_HASHING
        const val_t h   = h1 + hd1[b[l]].val;
        row[l] = check_insert_in_hash_table(n, h, ht2);
#else
        row[l] = insert_in_hash_table(n, ht2);
#endif
    }
}

static inline void reinsert_in_hash_table(
    hm_t *row,
    exp_t * const *oev,
    ht_t *ht
    )
{
    hl_t i;
    hi_t k, pos;
    len_t j, l;
    exp_t *e;
    hd_t *d;
    val_t h;

    const len_t len = row[LENGTH]+OFFSET;
    const len_t evl = ht->evl;
    const hi_t hsz  = ht->hsz;
    /* ht->hsz <= 2^32 => mod is always uint32_t */
    const hi_t mod  = (hi_t)(hsz - 1);
    l = OFFSET;
letsgo:
    for (; l < len; ++l) {
        const exp_t * const n = oev[row[l]];
        /* generate hash value */
        h = 0;
        for (j = 0; j < evl; ++j) {
            h +=  ht->rn[j] * n[j];
        }
        k = h;
        i = 0;
restart:
        for (; i < hsz; ++i) {
            k = (hi_t)(k+i) & mod;
            const hi_t hm  = ht->hmap[k];
            if (!hm) {
                break;
            }
            if (ht->hd[hm].val != h) {
                continue;
            }
            const exp_t * const ehm = ht->ev[hm];
            for (j = 0; j < evl-1; j += 2) {
                if (n[j] != ehm[j] || n[j+1] != ehm[j+1]) {
                    i++;
                    goto restart;
                }
            }
            if (n[evl-1] != ehm[evl-1]) {
                i++;
                goto restart;
            }
            row[l] = hm;
            l++;
            goto letsgo;
        }

        /* add element to hash table */
        ht->hmap[k] = pos = (hi_t)ht->eld;
        e = ht->ev[ht->eld];
        d = ht->hd + ht->eld;
        memcpy(e, n, (unsigned long)evl * sizeof(exp_t));
        d->sdm  =   generate_short_divmask(e, ht);
        d->deg  =   e[0];
        d->deg  +=  ht->ebl > 0 ? e[ht->ebl] : 0;
        d->val  =   h;

        ht->eld++;
        row[l] =  pos;
    }
}

void reset_hash_table_indices(
        ht_t *ht,
        const hi_t * const hcm,
        const len_t len
        )
{
    for (len_t i = 0; i < len; ++i) {
        ht->hd[hcm[i]].idx = 0;
    }
}

static void reset_hash_table(
    ht_t *ht,
    bs_t *bs,
    ps_t *psl,
    md_t *st
    )
{
    /* timings */
    double ct0, ct1, rt0, rt1;
    ct0 = cputime();
    rt0 = realtime();

    len_t i;
    hi_t k;
    exp_t *e;

    spair_t *ps = psl->p;
    exp_t **oev  = ht->ev;

    const len_t evl = ht->evl;
    const hl_t esz  = ht->esz;
    const bl_t bld  = bs->ld;
    const len_t pld = psl->ld;

    ht->ev  = calloc(esz, sizeof(exp_t *));
    if (ht->ev == NULL) {
        fprintf(stderr, "Computation needs too much memory on this machine,\n");
        fprintf(stderr, "cannot reset ht->ev, esz = %lu\n", (unsigned long)esz);
        fprintf(stderr, "segmentation fault will follow.\n");
    }
    exp_t *tmp  = (exp_t *)malloc(
            (unsigned long)evl * esz * sizeof(exp_t));
    if (tmp == NULL) {
        fprintf(stderr, "Computation needs too much memory on this machine,\n");
        fprintf(stderr, "resetting table failed, esz = %lu\n", (unsigned long)esz);
        fprintf(stderr, "segmentation fault will follow.\n");
    }
    for (k = 0; k < esz; ++k) {
        ht->ev[k]  = tmp + k*evl;
    }
    ht->eld = 1;
    memset(ht->hmap, 0, ht->hsz * sizeof(hi_t));
    memset(ht->hd, 0, esz * sizeof(hd_t));

    /* reinsert known elements */
    for (i = 0; i < bld; ++i) {
      if (bs->red[i] < 2) {
        reinsert_in_hash_table(bs->hm[i], oev, ht);
      }
    }
    for (i = 0; i < pld; ++i) {
        e = oev[ps[i].lcm];
#if PARALLEL_HASHING
        ps[i].lcm = check_insert_in_hash_table(e, 0, ht);
#else
        ps[i].lcm = insert_in_hash_table(e, ht);
#endif
    }
    /* note: all memory is allocated as a big block, so it is
     *       enough to free oev[0].       */
    free(oev[0]);
    free(oev);

    /* timings */
    ct1 = cputime();
    rt1 = realtime();
    st->rht_ctime  +=  ct1 - ct0;
    st->rht_rtime  +=  rt1 - rt0;
}

/* computes lcm of a and b from ht1 and inserts it in ht2 */
static inline hi_t get_lcm(
    const hi_t a,
    const hi_t b,
    const ht_t *ht1,
    ht_t *ht2
    )
{
    len_t i;

    /* exponents of basis elements, thus from basis hash table */
    const exp_t * const ea = ht1->ev[a];
    const exp_t * const eb = ht1->ev[b];
    exp_t etmp[ht1->evl];
    const len_t evl = ht1->evl;
    const len_t ebl = ht1->ebl;

    /* Performance optimization: Vectorized LCM computation */
    len_t j = 1;
    
#if defined(USE_AVX2)
    /* AVX2 vectorized max operation */
    const len_t simd_end = ((evl - 1) & ~7U) + 1;
    for (; j < simd_end; j += 8) {
        __m256i ea_vec = _mm256_loadu_si256((__m256i*)&ea[j]);
        __m256i eb_vec = _mm256_loadu_si256((__m256i*)&eb[j]);
        __m256i max_vec = _mm256_max_epi32(ea_vec, eb_vec);
        _mm256_storeu_si256((__m256i*)&etmp[j], max_vec);
    }
    for (; j < evl; ++j) {
        etmp[j] = ea[j] < eb[j] ? eb[j] : ea[j];
    }
#elif defined(USE_SSE4)
    /* SSE4 vectorized max operation */
    const len_t simd_end = ((evl - 1) & ~3U) + 1;
    for (; j < simd_end; j += 4) {
        __m128i ea_vec = _mm_loadu_si128((__m128i*)&ea[j]);
        __m128i eb_vec = _mm_loadu_si128((__m128i*)&eb[j]);
        __m128i max_vec = _mm_max_epi32(ea_vec, eb_vec);
        _mm_storeu_si128((__m128i*)&etmp[j], max_vec);
    }
    for (; j < evl; ++j) {
        etmp[j] = ea[j] < eb[j] ? eb[j] : ea[j];
    }
#else
    /* Unrolled scalar max operation */
    const len_t unroll_end = ((evl - 1) & ~3U) + 1;
    for (; j < unroll_end; j += 4) {
        etmp[j] = ea[j] < eb[j] ? eb[j] : ea[j];
        etmp[j+1] = ea[j+1] < eb[j+1] ? eb[j+1] : ea[j+1];
        etmp[j+2] = ea[j+2] < eb[j+2] ? eb[j+2] : ea[j+2];
        etmp[j+3] = ea[j+3] < eb[j+3] ? eb[j+3] : ea[j+3];
    }
    for (; j < evl; ++j) {
        etmp[j] = ea[j] < eb[j] ? eb[j] : ea[j];
    }
#endif
    
    /* reset degree entries */
    etmp[0] = 0;
    etmp[ebl] = 0;
    for (i = 1; i < ebl; ++i) {
        etmp[0] += etmp[i];
    }
    for (i = ebl+1; i < evl; ++i) {
        etmp[ebl] += etmp[i];
    }
    /* printf("lcm -> ");
     * for (int ii = 0; ii < evl; ++ii) {
     *     printf("%d ", etmp[ii]);
     * }
     * printf("\n"); */
#if PARALLEL_HASHING
    return check_insert_in_hash_table(etmp, 0, ht2);
#else
    return insert_in_hash_table(etmp, ht2);
#endif
}

static inline hm_t *poly_to_matrix_row(
    ht_t *sht,
    const ht_t *bht,
    const hm_t *poly
    )
{
  hm_t *row = (hm_t *)malloc((unsigned long)(poly[LENGTH]+OFFSET) * sizeof(hm_t));
  row[COEFFS]   = poly[COEFFS];
  row[PRELOOP]  = poly[PRELOOP];
  row[LENGTH]   = poly[LENGTH];
  /* hash table product insertions appear only here:
   * we check for hash table enlargements first and then do the insertions
   * without further elargment checks there */
  while (sht->eld+poly[LENGTH] >= sht->esz) {
    enlarge_hash_table(sht);
  }
  insert_poly_in_hash_table(row, poly, bht, sht);

  return row;
}
static inline hm_t *multiplied_poly_to_matrix_row(
    ht_t *sht,
    const ht_t *bht,
    const val_t hm,
    const exp_t * const em,
    const hm_t *poly
    )
{
  hm_t *row = (hm_t *)malloc((unsigned long)(poly[LENGTH]+OFFSET) * sizeof(hm_t));
  row[COEFFS]   = poly[COEFFS];
  row[PRELOOP]  = poly[PRELOOP];
  row[LENGTH]   = poly[LENGTH];
  /* hash table product insertions appear only here:
   * we check for hash table enlargements first and then do the insertions
   * without further elargment checks there */
  while (sht->eld+poly[LENGTH] >= sht->esz) {
    enlarge_hash_table(sht);
  }
  insert_multiplied_poly_in_hash_table(row, hm, em, poly, bht, sht);

  return row;
}
