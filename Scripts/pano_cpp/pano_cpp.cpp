#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For std::vector <-> Python list conversion
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

// Fast atan2 approximation - polynomial approximation, ~4x faster than std::atan2
inline float fast_atan2_approx(float y, float x) {
    if (x == 0.0f) x = 1e-5f;
    if (y == 0.0f) y = 1e-5f;
    float a = std::min(std::abs(x), std::abs(y)) / std::max(std::abs(x), std::abs(y));
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (std::abs(y) > std::abs(x)) r = 1.57079637f - r;
    if (x < 0) r = 3.14159274f - r;
    if (y < 0) r = -r;
    return r;
}

/**
 * Generate panorama from point cloud using z-buffer projection.
 *
 * Args:
 *     cloud: Point cloud array (N, 13) with columns [x, y, z, r, g, b, c1, c2, c3, c4, c5, c6, c7]
 *            RGB values should be in [0, 255], segmentation channels in [0, 1]
 *     pos: Current position (3,)
 *     rot_matrix: Rotation matrix (3, 3) for horizon-locked view
 *     view_dist: Maximum view distance for depth normalization (default 10.0)
 *
 * Returns:
 *     Panorama image (180, 360, 11) with channels [r, g, b, d, c1, c2, c3, c4, c5, c6, c7]
 *     RGB and D in [0, 255], segmentation channels in [0, 1] as float32
 */
py::array_t<float> generate_pano_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> cloud,
    py::array_t<float, py::array::c_style | py::array::forcecast> pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> rot_matrix,
    float view_dist = 10.0f
) {
    // Get input info
    auto cloud_buf = cloud.request();
    auto pos_buf = pos.request();
    auto rot_buf = rot_matrix.request();

    if (cloud_buf.ndim != 2 || cloud_buf.shape[1] < 13) {
        throw std::runtime_error("cloud must be (N, 13) array");
    }
    if (pos_buf.ndim != 1 || pos_buf.shape[0] != 3) {
        throw std::runtime_error("pos must be (3,) array");
    }
    if (rot_buf.ndim != 2 || rot_buf.shape[0] != 3 || rot_buf.shape[1] != 3) {
        throw std::runtime_error("rot_matrix must be (3, 3) array");
    }

    const int n_points = cloud_buf.shape[0];
    const float* cloud_ptr = static_cast<float*>(cloud_buf.ptr);
    const float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    const float* rot_ptr = static_cast<float*>(rot_buf.ptr);

    // Constants
    const int width = 360;
    const int height = 180;
    const int n_channels = 11;  // r, g, b, d, c1-c7
    const float ufac = 180.0f / M_PI;  // radians to degrees
    const float vfac = 180.0f / M_PI;

    // Allocate output
    auto result = py::array_t<float>({height, width, n_channels});
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    // Allocate z-buffer
    std::vector<float> z_buffer(height * width, view_dist);

    // Initialize result: d=255 (unknown depth), rest=0
    for (int i = 0; i < height * width; ++i) {
        for (int c = 0; c < n_channels; ++c) {
            result_ptr[i * n_channels + c] = 0.0f;
        }
        result_ptr[i * n_channels + 3] = 255.0f;  // d channel = 255 for unknown
    }

    // Extract position
    float px = pos_ptr[0];
    float py_ = pos_ptr[1];
    float pz = pos_ptr[2];

    // Project all points
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; ++i) {
        const float* pt = cloud_ptr + i * 13;

        // Transform: subtract position
        float x = pt[0] - px;
        float y = pt[1] - py_;
        float z = pt[2] - pz;

        // Apply rotation matrix (row-major): rotated = rot @ [x, y, z]
        float rx = rot_ptr[0] * x + rot_ptr[1] * y + rot_ptr[2] * z;
        float ry = rot_ptr[3] * x + rot_ptr[4] * y + rot_ptr[5] * z;
        float rz = rot_ptr[6] * x + rot_ptr[7] * y + rot_ptr[8] * z;

        // Skip zero points
        if (rx == 0 && ry == 0 && rz == 0) continue;

        // Calculate distance
        float dxy = std::sqrt(rx * rx + ry * ry);
        float dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        // Filter by distance
        if (dist > view_dist) continue;

        // Spherical projection (like lidar_to_surround_coords)
        float u_val = fast_atan2_approx(rx, ry) * ufac;
        float v_val = -fast_atan2_approx(rz, dxy) * vfac;

        // Convert to pixel indices (use floor to match panorama.py)
        float u_shifted = u_val + 90.0f + 360.0f;
        int u_idx = static_cast<int>(std::floor(u_shifted)) % 360;
        if (u_idx < 0) u_idx += 360;
        int v_idx = static_cast<int>(std::floor(v_val)) + 90;

        // Bounds check
        if (u_idx < 0 || u_idx >= width || v_idx < 0 || v_idx >= height) continue;

        // Z-buffer test and write (with critical section for thread safety)
        int pixel_idx = v_idx * width + u_idx;

        #pragma omp critical
        {
            if (dist < z_buffer[pixel_idx]) {
                z_buffer[pixel_idx] = dist;
                float* out = result_ptr + pixel_idx * n_channels;

                // RGB (input is 0-255)
                out[0] = pt[3];  // r
                out[1] = pt[4];  // g
                out[2] = pt[5];  // b

                // Depth normalized to 0-255
                out[3] = std::min(255.0f, dist / view_dist * 255.0f);

                // Segmentation channels c1-c7 (keep 0-1 range, matching panorama.py)
                for (int c = 0; c < 7; ++c) {
                    out[4 + c] = pt[6 + c];
                }
            }
        }
    }

    return result;
}

/**
 * Optimized version without critical section - uses relaxed consistency.
 * Slightly less accurate but much faster for dense point clouds.
 */
py::array_t<float> generate_pano_cpp_fast(
    py::array_t<float, py::array::c_style | py::array::forcecast> cloud,
    py::array_t<float, py::array::c_style | py::array::forcecast> pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> rot_matrix,
    float view_dist = 10.0f
) {
    auto cloud_buf = cloud.request();
    auto pos_buf = pos.request();
    auto rot_buf = rot_matrix.request();

    if (cloud_buf.ndim != 2 || cloud_buf.shape[1] < 13) {
        throw std::runtime_error("cloud must be (N, 13) array");
    }

    const int n_points = cloud_buf.shape[0];
    const float* cloud_ptr = static_cast<float*>(cloud_buf.ptr);
    const float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    const float* rot_ptr = static_cast<float*>(rot_buf.ptr);

    const int width = 360;
    const int height = 180;
    const int n_channels = 11;
    const float ufac = 180.0f / M_PI;
    const float vfac = 180.0f / M_PI;

    auto result = py::array_t<float>({height, width, n_channels});
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    std::vector<float> z_buffer(height * width, view_dist);

    // Initialize
    for (int i = 0; i < height * width; ++i) {
        for (int c = 0; c < n_channels; ++c) {
            result_ptr[i * n_channels + c] = 0.0f;
        }
        result_ptr[i * n_channels + 3] = 255.0f;
    }

    float px = pos_ptr[0];
    float py_ = pos_ptr[1];
    float pz = pos_ptr[2];

    // No critical section - relaxed consistency (may have minor artifacts)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; ++i) {
        const float* pt = cloud_ptr + i * 13;

        float x = pt[0] - px;
        float y = pt[1] - py_;
        float z = pt[2] - pz;

        float rx = rot_ptr[0] * x + rot_ptr[1] * y + rot_ptr[2] * z;
        float ry = rot_ptr[3] * x + rot_ptr[4] * y + rot_ptr[5] * z;
        float rz = rot_ptr[6] * x + rot_ptr[7] * y + rot_ptr[8] * z;

        if (rx == 0 && ry == 0 && rz == 0) continue;

        float dxy = std::sqrt(rx * rx + ry * ry);
        float dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        if (dist > view_dist) continue;

        float u_val = fast_atan2_approx(rx, ry) * ufac;
        float v_val = -fast_atan2_approx(rz, dxy) * vfac;

        // Convert to pixel indices (use floor to match panorama.py)
        float u_shifted = u_val + 90.0f + 360.0f;
        int u_idx = static_cast<int>(std::floor(u_shifted)) % 360;
        if (u_idx < 0) u_idx += 360;
        int v_idx = static_cast<int>(std::floor(v_val)) + 90;

        if (u_idx < 0 || u_idx >= width || v_idx < 0 || v_idx >= height) continue;

        int pixel_idx = v_idx * width + u_idx;

        // Relaxed write - possible race condition but acceptable for rendering
        if (dist < z_buffer[pixel_idx]) {
            z_buffer[pixel_idx] = dist;
            float* out = result_ptr + pixel_idx * n_channels;
            out[0] = pt[3];
            out[1] = pt[4];
            out[2] = pt[5];
            out[3] = std::min(255.0f, dist / view_dist * 255.0f);
            // Segmentation channels c1-c7 (keep 0-1 range, matching panorama.py)
            for (int c = 0; c < 7; ++c) {
                out[4 + c] = pt[6 + c];
            }
        }
    }

    return result;
}

/**
 * Generate panorama from multiple point cloud arrays (avoids Python concatenate).
 * Uses flat indexing with binary search to map global index to (cloud, local_idx).
 */
py::array_t<float> generate_pano_multi(
    std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> clouds,
    py::array_t<float, py::array::c_style | py::array::forcecast> pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> rot_matrix,
    float view_dist = 10.0f
) {
    auto pos_buf = pos.request();
    auto rot_buf = rot_matrix.request();

    const float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    const float* rot_ptr = static_cast<float*>(rot_buf.ptr);

    const int width = 360;
    const int height = 180;
    const int n_channels = 11;
    const float ufac = 180.0f / M_PI;
    const float vfac = 180.0f / M_PI;

    // Build index: store pointers and cumulative counts
    std::vector<const float*> cloud_ptrs;
    std::vector<size_t> cumsum;
    size_t total_points = 0;

    for (auto& cloud : clouds) {
        auto buf = cloud.request();
        if (buf.ndim == 2 && buf.shape[1] >= 13) {
            cloud_ptrs.push_back(static_cast<const float*>(buf.ptr));
            cumsum.push_back(total_points);
            total_points += buf.shape[0];
        }
    }
    const int n_clouds = static_cast<int>(cloud_ptrs.size());
    if (n_clouds == 0 || total_points == 0) {
        auto result = py::array_t<float>({height, width, n_channels});
        auto result_buf = result.request();
        float* result_ptr = static_cast<float*>(result_buf.ptr);
        for (int i = 0; i < height * width; ++i) {
            for (int c = 0; c < n_channels; ++c) {
                result_ptr[i * n_channels + c] = 0.0f;
            }
            result_ptr[i * n_channels + 3] = 255.0f;
        }
        return result;
    }

    auto result = py::array_t<float>({height, width, n_channels});
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    std::vector<float> z_buffer(height * width, view_dist);

    // Initialize
    for (int i = 0; i < height * width; ++i) {
        for (int c = 0; c < n_channels; ++c) {
            result_ptr[i * n_channels + c] = 0.0f;
        }
        result_ptr[i * n_channels + 3] = 255.0f;
    }

    float px = pos_ptr[0];
    float py_ = pos_ptr[1];
    float pz = pos_ptr[2];

    // Single parallel loop over all points using flat indexing
    const size_t n_total = total_points;
    const size_t* cumsum_ptr = cumsum.data();
    const float* const* ptrs = cloud_ptrs.data();

    #pragma omp parallel for schedule(static)
    for (size_t gi = 0; gi < n_total; ++gi) {
        // Binary search to find which cloud this point belongs to
        int lo = 0, hi = n_clouds;
        while (lo < hi - 1) {
            int mid = (lo + hi) / 2;
            if (cumsum_ptr[mid] <= gi) lo = mid;
            else hi = mid;
        }
        const int ci = lo;
        const size_t local_idx = gi - cumsum_ptr[ci];
        const float* pt = ptrs[ci] + local_idx * 13;

        float x = pt[0] - px;
        float y = pt[1] - py_;
        float z = pt[2] - pz;

        float rx = rot_ptr[0] * x + rot_ptr[1] * y + rot_ptr[2] * z;
        float ry = rot_ptr[3] * x + rot_ptr[4] * y + rot_ptr[5] * z;
        float rz = rot_ptr[6] * x + rot_ptr[7] * y + rot_ptr[8] * z;

        if (rx == 0 && ry == 0 && rz == 0) continue;

        float dxy = std::sqrt(rx * rx + ry * ry);
        float dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        if (dist > view_dist) continue;

        float u_val = fast_atan2_approx(rx, ry) * ufac;
        float v_val = -fast_atan2_approx(rz, dxy) * vfac;

        float u_shifted = u_val + 90.0f + 360.0f;
        int u_idx = static_cast<int>(std::floor(u_shifted)) % 360;
        if (u_idx < 0) u_idx += 360;
        int v_idx = static_cast<int>(std::floor(v_val)) + 90;

        if (u_idx < 0 || u_idx >= width || v_idx < 0 || v_idx >= height) continue;

        int pixel_idx = v_idx * width + u_idx;

        if (dist < z_buffer[pixel_idx]) {
            z_buffer[pixel_idx] = dist;
            float* out = result_ptr + pixel_idx * n_channels;
            out[0] = pt[3];
            out[1] = pt[4];
            out[2] = pt[5];
            out[3] = std::min(255.0f, dist / view_dist * 255.0f);
            for (int c = 0; c < 7; ++c) {
                out[4 + c] = pt[6 + c];
            }
        }
    }

    return result;
}

PYBIND11_MODULE(pano_cpp, m) {
    m.doc() = "Fast panorama generation using C++ with OpenMP";

    m.def("generate_pano", &generate_pano_cpp,
          "Generate panorama from point cloud (thread-safe version)",
          py::arg("cloud"),
          py::arg("pos"),
          py::arg("rot_matrix"),
          py::arg("view_dist") = 10.0f);

    m.def("generate_pano_fast", &generate_pano_cpp_fast,
          "Generate panorama from point cloud (fast version with relaxed consistency)",
          py::arg("cloud"),
          py::arg("pos"),
          py::arg("rot_matrix"),
          py::arg("view_dist") = 10.0f);

    m.def("generate_pano_multi", &generate_pano_multi,
          "Generate panorama from multiple point cloud arrays (avoids Python concatenate)",
          py::arg("clouds"),
          py::arg("pos"),
          py::arg("rot_matrix"),
          py::arg("view_dist") = 10.0f);
}
