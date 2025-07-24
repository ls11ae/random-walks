#include "caching.h"

uint64_t compute_matrix_hash(const Matrix* m) {
    uint64_t h = 146527;
    for (size_t i = 0; i < m->len; i++) {
        uint64_t bits;
        memcpy(&bits, &m->data[i], sizeof(bits));
        h ^= bits + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

uint64_t compute_parameters_hash(const KernelParameters* params) {
    uint64_t h = 14695981039346656037ULL;
    h = (h ^ (params->is_brownian)) * 1099511628211ULL;
    h = (h ^ params->S) * 1099511628211ULL;
    h = (h ^ params->D) * 1099511628211ULL;
    // Hash float values by their bit pattern
    uint64_t bits;
    memcpy(&bits, &params->diffusity, sizeof(bits));
    h = (h ^ bits) * 1099511628211ULL;

    h = (h ^ params->bias_x) * 1099511628211ULL;
    h = (h ^ params->bias_y) * 1099511628211ULL;

    return h;
}

uint32_t hash_bytes(const void* key, size_t length) {
    const uint8_t* data = (const uint8_t*)key;
    uint32_t hash = 0;
    for (size_t i = 0; i < length; i++) {
        hash += data[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

uint32_t weather_entry_hash(const WeatherEntry* entry) {
    uint32_t hash = 0;
    // Hash each field individually and combine them
    hash = hash_bytes(&entry->temperature, sizeof(entry->temperature));
    hash ^= hash_bytes(&entry->humidity, sizeof(entry->humidity));
    hash ^= hash_bytes(&entry->precipitation, sizeof(entry->precipitation));
    hash ^= hash_bytes(&entry->wind_speed, sizeof(entry->wind_speed));
    hash ^= hash_bytes(&entry->wind_direction, sizeof(entry->wind_direction));
    hash ^= hash_bytes(&entry->snow_fall, sizeof(entry->snow_fall));
    hash ^= hash_bytes(&entry->weather_code, sizeof(entry->weather_code));
    hash ^= hash_bytes(&entry->cloud_cover, sizeof(entry->cloud_cover));

    // Final mixing
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}


Cache* cache_create(size_t num_buckets) {
    Cache* cache = (Cache*)malloc(sizeof(Cache));
    cache->num_buckets = num_buckets;
    cache->buckets = (CacheEntry**)calloc(num_buckets, sizeof(CacheEntry*));
    return cache;
}

CacheEntry* cache_lookup_entry(Cache* cache, uint64_t hash) {
    size_t bucket = hash % cache->num_buckets;

    CacheEntry* entry = cache->buckets[bucket];

    while (entry != NULL) {
        if (entry->hash == hash) {
            return entry;
        }

        entry = entry->next;
    }

    return NULL;
}

void cache_insert(Cache* cache, uint64_t hash, void* data, bool is_array, ssize_t array_size) {
    assert((is_array && data != NULL) || (!is_array && data != NULL));
    size_t bucket = hash % cache->num_buckets;
    CacheEntry* entry = malloc(sizeof(CacheEntry));
    entry->hash = hash;
    entry->is_array = is_array;
    entry->array_size = array_size;
    if (is_array) {
        entry->data.array = (Tensor*)data;
    }
    else {
        entry->data.single = (Matrix*)data;
    }
    entry->next = cache->buckets[bucket];
    cache->buckets[bucket] = entry;
}

void cache_free(Cache* cache) {
    for (size_t i = 0; i < cache->num_buckets; i++) {
        CacheEntry* entry = cache->buckets[i];
        while (entry != NULL) {
            CacheEntry* next = entry->next;
            if (entry->is_array) {
                tensor_free(entry->data.array);
            }
            else {
                matrix_free(entry->data.single);
            }
            free(entry);
            entry = next;
        }
    }
    free(cache->buckets);
    free(cache);
}

uint64_t hash_combine(uint64_t a, uint64_t b) {
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

size_t hash_to_bucket(uint64_t hash) {
    return hash % HASH_CACHE_BUCKETS;
}

// Gibt den Pfad zu einem bestehenden Tensor mit gleichem Inhalt zurück, oder NULL, wenn neuer Eintrag
const char* hash_cache_lookup_or_insert(HashCache* cache, Tensor* t, uint64_t hash, const char* new_path) {
    size_t bucket = hash_to_bucket(hash);
    HashEntry* entry = cache->buckets[bucket];

    // Lineare Suche in der verketteten Liste (für Hash-Kollisionen)
    while (entry) {
        if (entry->hash == hash) {
            return entry->path; // Tensor-Inhalt gleich → existierender Pfad
        }
        entry = entry->next;
    }

    // Kein passender Tensor gefunden → neuen Eintrag anlegen
    HashEntry* new_entry = malloc(sizeof(HashEntry));
    new_entry->hash = hash;
    new_entry->tensor = tensor_clone(t); // Duplizieren, damit Cache überlebt
    strncpy(new_entry->path, new_path, PATH_MAX);
    new_entry->path[PATH_MAX - 1] = '\0';
    new_entry->next = cache->buckets[bucket];
    cache->buckets[bucket] = new_entry;

    return NULL; // Neu eingefügt, kein vorhandener Pfad
}

const char* hash_cache_lookup_or_insert2(HashCache* cache, uint64_t hash, const char* new_path) {
    size_t bucket = hash % HASH_CACHE_BUCKETS;
    HashEntry* entry = cache->buckets[bucket];
    
    // Durchsuche Kollisionsliste
    while (entry) {
        if (entry->hash == hash) return entry->path; // Gefunden!
        entry = entry->next;
    }
    
    // Neuer Eintrag: Füge hinzu
    HashEntry* new_entry = malloc(sizeof(HashEntry));
    *new_entry = (HashEntry){
        .hash = hash,
        .path = "", //strndup(new_path, PATH_MAX),  // Now valid
        .next = cache->buckets[bucket]
    };
    cache->buckets[bucket] = new_entry;
    return NULL;  // Neu, kein existierender Pfad
}

size_t hash_mix(size_t hash, size_t value) {
    hash ^= value + 0x9e3779b97f4a7c15 + (hash << 6) + (hash >> 2); // gute Mischung
    return hash;
}

uint64_t hash_double(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof(double));
    return u;
}

size_t tensor_hash(const Tensor* t) {
    const size_t FNV_OFFSET = 14695981039346656037ULL;
    size_t hash = FNV_OFFSET;

    hash = hash_mix(hash, t->len);

    for (size_t i = 0; i < t->len; ++i) {
        Matrix* m = t->data[i];
        if (!m) continue;

        hash = hash_mix(hash, m->width);
        hash = hash_mix(hash, m->height);
        hash = hash_mix(hash, m->len);

        for (ssize_t j = 0; j < m->len; ++j) {
            hash = hash_mix(hash, hash_double(m->data[j]));
        }
    }

    return hash;
}

HashCache* hash_cache_create() {
    HashCache* cache = (HashCache*)malloc(sizeof(HashCache));
    if (!cache) {
        perror("Failed to allocate HashCache");
        exit(EXIT_FAILURE);
    }
    memset(cache->buckets, 0, sizeof(cache->buckets));
    return cache;
}
