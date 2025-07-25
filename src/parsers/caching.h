#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "parsers/types.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "matrix/matrix.h"
#include "matrix/tensor.h"

uint64_t compute_matrix_hash(const Matrix* m);

uint64_t compute_parameters_hash(const KernelParameters* params);

uint32_t hash_bytes(const void* key, size_t length);

uint32_t weather_entry_hash(const WeatherEntry* entry);

Cache* cache_create(size_t num_buckets);

CacheEntry* cache_lookup_entry(Cache* cache, uint64_t hash);

void cache_insert(Cache* cache, uint64_t hash, void* data, bool is_array, ssize_t array_size);

void cache_free(Cache* cache);

uint64_t hash_combine(uint64_t a, uint64_t b);

size_t hash_to_bucket(uint64_t hash);

const char* hash_cache_lookup_or_insert(HashCache* cache, Tensor* t, uint64_t hash, const char* new_path);

const char* hash_cache_lookup_or_insert2(HashCache* cache, uint64_t hash, const char* new_path);


size_t hash_mix(size_t hash, size_t value);

uint64_t hash_double(double x);

size_t tensor_hash(const Tensor* t);

HashCache* hash_cache_create();

void hash_cache_free(HashCache* cache);

#ifdef __cplusplus
}
#endif
