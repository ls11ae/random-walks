#pragma once

/**
 * @file
 * @brief Hashing and caching utilities for matrices, tensors, and parser-related structures.
 *
 * This header declares functions to compute stable hashes for various data structures,
 * as well as a lightweight cache API keyed by these hashes.
 * All APIs are C-compatible and usable from both C and C++.
 */

#ifdef __cplusplus
extern "C" {
#endif
#include "parsers/types.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "matrix/matrix.h"
#include "matrix/tensor.h"

/**
 * @brief Compute a 64-bit hash for a Matrix.
 * @param m Pointer to the matrix instance.
 * @return 64-bit hash value; implementation is stable across process runs.
 */
uint64_t compute_matrix_hash(const Matrix* m);

/**
 * @brief Compute a 64-bit hash for kernel parameters.
 * @param params Pointer to the parameters instance.
 * @return 64-bit hash value derived from the parameter fields.
 */
uint64_t compute_parameters_hash(const KernelParameters* params);

/**
 * @brief Hash an arbitrary sequence of bytes.
 * @param key Pointer to the byte buffer.
 * @param length Number of bytes to hash.
 * @return 32-bit hash of the provided data.
 */
uint32_t hash_bytes(const void* key, size_t length);

/**
 * @brief Compute a 32-bit hash for a WeatherEntry.
 * @param entry Pointer to the entry to hash.
 * @return 32-bit hash value.
 */
uint32_t weather_entry_hash(const WeatherEntry* entry);

/**
 * @brief Create a hash-based cache.
 * @param num_buckets Number of buckets for the internal hash table.
 * @return Newly allocated cache instance, or NULL on failure.
 */
Cache* cache_create(size_t num_buckets);

/**
 * @brief Look up an entry by its hash.
 * @param cache Cache to query.
 * @param hash Hash key to search for.
 * @return Pointer to the cache entry if found, or NULL otherwise.
 */
CacheEntry* cache_lookup_entry(Cache* cache, uint64_t hash);

/**
 * @brief Insert a new entry into the cache.
 * @param cache Target cache.
 * @param hash Key associated with the data.
 * @param data Pointer to user data to store.
 * @param is_array Whether data points to an array allocation.
 * @param array_size Number of elements in the array when @p is_array is true; otherwise ignored.
 */
void cache_insert(Cache* cache, uint64_t hash, void* data, bool is_array, ssize_t array_size);

/**
 * @brief Destroy a cache and release its resources.
 * @param cache Cache instance to free. It is safe to pass NULL.
 */
void cache_free(Cache* cache);

/**
 * @brief Combine two 64-bit hashes into one.
 * @param a First hash.
 * @param b Second hash.
 * @return Combined 64-bit hash value.
 */
uint64_t hash_combine(uint64_t a, uint64_t b);

/**
 * @brief Map a hash value to a bucket index.
 * @param hash Hash value.
 * @return Bucket index within the cache's range.
 */
size_t hash_to_bucket(uint64_t hash);

/**
 * @brief Look up a path by tensor and hash or insert a new one.
 * @param cache Hash cache to query.
 * @param t Tensor used for hashing or context.
 * @param hash Precomputed hash key.
 * @param new_path Path to associate if no entry exists.
 * @return Existing path if found, otherwise @p new_path after insertion.
 */
const char* hash_cache_lookup_or_insert(HashCache* cache, Tensor* t, uint64_t hash, const char* new_path);

/**
 * @brief Look up a path by hash or insert a new one.
 * @param cache Hash cache to query.
 * @param hash Precomputed hash key.
 * @param new_path Path to associate if absent.
 * @return Existing path if found, otherwise @p new_path after insertion.
 */
const char* hash_cache_lookup_or_insert2(HashCache* cache, uint64_t hash, const char* new_path);


/**
 * @brief Mix a value into an existing hash state.
 * @param hash Current hash accumulator.
 * @param value Value to incorporate.
 * @return Updated mixed hash.
 */
size_t hash_mix(size_t hash, size_t value);

/**
 * @brief Hash a double-precision floating point value.
 * @param x Value to hash.
 * @return 64-bit hash of the value's bit representation (with normalization as implemented).
 */
uint64_t hash_double(double x);

/**
 * @brief Compute a size_t hash for a Tensor.
 * @param t Tensor to hash.
 * @return Hash value representing the tensor's shape and/or contents.
 */
size_t tensor_hash(const Tensor* t);

/**
 * @brief Create a HashCache instance.
 * @return Newly allocated cache, or NULL on failure.
 */
HashCache* hash_cache_create();

/**
 * @brief Free a HashCache instance.
 * @param cache Instance to destroy. It is safe to pass NULL.
 */
void hash_cache_free(HashCache* cache);

#ifdef __cplusplus
}
#endif
