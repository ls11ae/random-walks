#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "math/math_utils.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_mixed_walk(int T, int W, int H,
                             int start_x, int start_y,
                             int end_x, int end_y,
                             KernelsMap3D *kernels_map,
                             KernelParametersMapping *mapping,
                             TerrainMap *terrain_map,
                             bool serialize,
                             const char *serialization_path);

// static std::vector<std::vector<int2> >
// build_partitioned_reference(int D, int kernel_w) {
//     std::vector<std::vector<int2> > ref(D);
//     int S = kernel_w / 2;
//     float angle_step = 360.0f / (float) D;
//
//     for (int yy = -S; yy <= S; yy++) {
//         for (int xx = -S; xx <= S; xx++) {
//             float angle = compute_angle(xx, yy);
//             float closest = find_closest_angle(angle, angle_step);
//             size_t dir = ((closest == 360.0f) ? 0 : angle_to_direction(closest, angle_step)) % D;
//             ref[dir].push_back({xx, yy});
//         }
//     }
//     return ref;
// }

// === TEST-HARNESS ===
// static void test_build_kernel_pool(const KernelsMap3D *km,
//                                    const TerrainMap *terrain_map) {
//     //KernelPool gpu_out;
//     //build_kernel_pool_from_kernels_map(km, terrain_map, gpu_out);
//
//     // Wir nehmen einfach den ersten Kernel aus dem Pool
//     assert(!gpu_out.kernel_offsets.empty());
//     int first_offset = gpu_out.kernel_offsets[0];
//     int first_D = gpu_out.kernel_Ds[0];
//     int first_w = gpu_out.kernel_widths[0];
//     int first_size = gpu_out.offsets_size_per_kernel_dir[0]; // nur zur Kontrolle
//
//     printf("GPU-Pool: D=%d, w=%d, offsets-start=%d\n", first_D, first_w, first_offset);
//
//     auto ref_partitioned = build_partitioned_reference(first_D, first_w);
//
//     for (int di = 0; di < first_D; di++) {
//         int idx = gpu_out.offsets_index_per_kernel_dir[di];
//         int size = gpu_out.offsets_size_per_kernel_dir[di];
//
//         if (size != (int) ref_partitioned[di].size()) {
//             printf("Direction %d size mismatch: GPU=%d, REF=%zu\n",
//                    di, size, ref_partitioned[di].size());
//         }
//
//         for (int i = 0; i < std::min(size, (int) ref_partitioned[di].size()); i++) {
//             auto g = gpu_out.offsets_pool[idx + i];
//             auto r = ref_partitioned[di][i];
//             if (g.x != r.x || g.y != r.y) {
//                 printf("Dir %d mismatch at %d: GPU=(%d,%d), REF=(%d,%d)\n",
//                        di, i, g.x, g.y, r.x, r.y);
//             }
//         }
//     }
//
//     printf("Test done.\n");
// }

#ifdef __cplusplus
}
#endif



