// VAMP_Planner_Aligned.cpp
// ALIGNED IMPLEMENTATION with VAMP paper requirements
// - True 3D kinematics with SE(3) transforms
// - Hierarchical sphere culling (coarse → medium → fine)
// - FK/CC interleaving (check spheres as computed, early exit)
// - Unrolled branch-free FK for specific robots
// Compile: g++ -O3 -march=native -mavx2 -std=c++17 -flto -ffast-math VAMP_Planner_Aligned.cpp -o vamp_aligned

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <numeric>
#include <array>
#include <set>
#include <cstdint>
#include <cassert>
#ifdef _WIN32
  #include <malloc.h>  // for _aligned_malloc/_aligned_free
#else
  #include <stdlib.h>  // for posix_memalign
#endif
#ifdef USE_SLEEF
  // SLEEF provides vector trig like Sleef_sinf8_u35 etc.
  #include <sleef.h>
#endif

// ----------------------------- CONFIG -------------------------------------
constexpr size_t SIMD_WIDTH = 8;             // AVX2: 8 floats per __m256
constexpr float GRID_CELL_SIZE = 0.5f;
constexpr size_t RAKE_WIDTH = 8;              // Check 8 waypoints simultaneously

// Compile-time safety: ensure SIMD_WIDTH matches mask assumptions
static_assert(SIMD_WIDTH == 8, "This code assumes SIMD_WIDTH==8 lanes for uint8_t masks");

// Sentinel values for padding (safe for -ffast-math, avoids NaN issues)
constexpr float PAD_COORD = 1e8f;            // Sentinel coordinate for padded boxes
constexpr float PAD_SENTINEL = 1e8f;         // Threshold for detecting padded boxes
constexpr float SPHERE_PAD_RADIUS = -1.0f;   // Sentinel radius for padded spheres

// --------------------------------------------------------------------------

// ----------------------------- UTIL ---------------------------------------
// Helper function to produce human-friendly binary masks (useful during parity debugging)
static std::string bits8(uint32_t x) {
    std::string s;
    for (int i = 0; i < 8; ++i) {
        s += ((x >> i) & 1) ? '1' : '0';
    }
    return s;
}

inline double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}
inline float sqr(float x) { return x*x; }

// ----------------------- SIMD helpers -----------------------
static inline __m256 load8(const float *p) { return _mm256_loadu_ps(p); }
static inline void store8(float *p, __m256 v) { _mm256_storeu_ps(p, v); }

static inline __m256 set1_8(float v) { return _mm256_set1_ps(v); }
static inline int movemask8(__m256 cmp) { return _mm256_movemask_ps(cmp); }

// 3x3 rotation matrix
struct Mat3 {
    float m[9]; // row-major: m[row*3 + col]
    Mat3() { for(int i=0;i<9;++i) m[i]=0; m[0]=m[4]=m[8]=1; }
    
    static Mat3 rotZ(float angle) {
        Mat3 r;
        float c = std::cos(angle), s = std::sin(angle);
        r.m[0]=c; r.m[1]=-s; r.m[2]=0;
        r.m[3]=s; r.m[4]=c;  r.m[5]=0;
        r.m[6]=0; r.m[7]=0;  r.m[8]=1;
        return r;
    }
};

// SE(3) transform: rotation + translation
struct Transform3D {
    Mat3 rot;
    float tx, ty, tz;
    Transform3D() : tx(0), ty(0), tz(0) {}
    Transform3D(const Mat3& r, float x, float y, float z) : rot(r), tx(x), ty(y), tz(z) {}
    
    // Transform a point
    void transformPoint(float px, float py, float pz, float& ox, float& oy, float& oz) const {
        ox = rot.m[0]*px + rot.m[1]*py + rot.m[2]*pz + tx;
        oy = rot.m[3]*px + rot.m[4]*py + rot.m[5]*pz + ty;
        oz = rot.m[6]*px + rot.m[7]*py + rot.m[8]*pz + tz;
    }
    
    // Set to identity
    void setIdentity() {
        for(int i=0;i<9;++i) rot.m[i]=0;
        rot.m[0]=rot.m[4]=rot.m[8]=1;
        tx = ty = tz = 0;
    }
    
    // Compose transforms: this * other
    Transform3D compose(const Transform3D& other) const {
        Transform3D result;
        // Rotate other's translation
        transformPoint(other.tx, other.ty, other.tz, result.tx, result.ty, result.tz);
        // Multiply rotations: result.rot = this->rot * other.rot
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                result.rot.m[i*3+j] = 0;
                for(int k=0; k<3; ++k) {
                    result.rot.m[i*3+j] += rot.m[i*3+k] * other.rot.m[k*3+j];
                }
            }
        }
        return result;
    }
};
// --------------------------------------------------------------------------

// ----------------------------- STATES -------------------------------------
struct State3DArm {
    std::vector<float> joint_angles;
    State3DArm() {}
    State3DArm(int n) : joint_angles(static_cast<size_t>(n), 0.0f) {}
    State3DArm(const std::vector<float>& a) : joint_angles(a) {}
    float distance(const State3DArm& o) const {
        float s=0; 
        for (size_t i=0;i<joint_angles.size();++i){ 
            float d=joint_angles[i]-o.joint_angles[i]; 
            s+=d*d; 
        } 
        return std::sqrt(s);
    }
    float distanceSquared(const State3DArm& o) const {
        float s=0; 
        for (size_t i=0;i<joint_angles.size();++i){ 
            float d=joint_angles[i]-o.joint_angles[i]; 
            s+=d*d; 
        } 
        return s;
    }
    State3DArm interpolate(const State3DArm& o, float t) const {
        State3DArm out(static_cast<int>(joint_angles.size())); 
        for (size_t i=0;i<joint_angles.size();++i) 
            out.joint_angles[i] = joint_angles[i] + t*(o.joint_angles[i]-joint_angles[i]); 
        return out;
    }
};

// ========================= VECTORIZED CONFIGURATION BATCH =========================
// SOA representation of 8 configurations at once (ALIGNED WITH VAMP PAPER)
struct VectorizedConfig3DArm {
    int num_joints;
    std::vector<float*> joints_aligned; // pointers to 8-float aligned blocks
    
    VectorizedConfig3DArm(int n) : num_joints(n), joints_aligned(static_cast<size_t>(n), nullptr) {
        for (int j = 0; j < n; ++j) {
            void* p = nullptr;
#ifdef _WIN32
            // Windows: use _aligned_malloc
            p = _aligned_malloc(SIMD_WIDTH * sizeof(float), 32);
            if (!p) {
                std::cerr << "Error: Failed to allocate aligned memory\n";
                throw std::bad_alloc();
            }
#else
            // POSIX: use posix_memalign
            int result = posix_memalign(&p, 32, SIMD_WIDTH * sizeof(float));
            if (result != 0) {
                std::cerr << "Error: posix_memalign failed with code " << result << "\n";
                throw std::bad_alloc();
            }
#endif
            joints_aligned[static_cast<size_t>(j)] = reinterpret_cast<float*>(p);
            // Initialize to zero
            for (size_t k = 0; k < SIMD_WIDTH; ++k) {
                joints_aligned[static_cast<size_t>(j)][k] = 0.0f;
            }
        }
    }
    
    ~VectorizedConfig3DArm() {
        for (auto p : joints_aligned) {
            if (p) {
#ifdef _WIN32
                _aligned_free(p);
#else
                free(p);
#endif
            }
        }
    }
    
    // Load 8 configurations from AOS to SOA
    void loadFromAOS(const std::array<State3DArm, SIMD_WIDTH>& configs) {
        for (size_t cfg = 0; cfg < SIMD_WIDTH; ++cfg) {
            for (int j = 0; j < num_joints; ++j) {
                size_t j_idx = static_cast<size_t>(j);
                if (j < static_cast<int>(configs[cfg].joint_angles.size())) {
                    joints_aligned[j_idx][cfg] = configs[cfg].joint_angles[j_idx];
                } else {
                    joints_aligned[j_idx][cfg] = 0.0f;
                }
            }
        }
    }
    
    __m256 getJoint(int j) const {
        return _mm256_load_ps(joints_aligned[static_cast<size_t>(j)]); // aligned load (32-byte aligned)
    }
};
// ==================================================================================

// ============================= HIERARCHICAL SPHERE MODEL ==========================
// ALIGNED WITH VAMP: Three levels of detail for collision checking
struct Sphere3D {
    float x, y, z, r;
    int level;  // 0=coarse (entire link), 1=medium (2-3 per link), 2=fine (4+ per link)
    int link_index;
    
    Sphere3D(float X=0, float Y=0, float Z=0, float R=0, int lvl=0, int link=-1) 
        : x(X), y(Y), z(Z), r(R), level(lvl), link_index(link) {}
};

// DH (Denavit-Hartenberg) parameters for robot kinematics
struct DHParams {
    float a;          // Link length
    float alpha;      // Link twist
    float d;          // Link offset
    float theta_offset; // Joint angle offset
    
    DHParams(float a_=0, float alpha_=0, float d_=0, float offset_=0)
        : a(a_), alpha(alpha_), d(d_), theta_offset(offset_) {}
};

// Compute DH transform: T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
static inline Transform3D dhTransform(const DHParams& dh, float theta_var) {
    float theta = theta_var + dh.theta_offset;
    float cth = std::cos(theta), sth = std::sin(theta);
    float cal = std::cos(dh.alpha), sal = std::sin(dh.alpha);
    
    // Rotation matrix (Z rotation * X rotation)
    Mat3 rot;
    rot.m[0] = cth;
    rot.m[1] = -sth * cal;
    rot.m[2] = sth * sal;
    rot.m[3] = sth;
    rot.m[4] = cth * cal;
    rot.m[5] = -cth * sal;
    rot.m[6] = 0;
    rot.m[7] = sal;
    rot.m[8] = cal;
    
    // Translation: d along Z, then a along X (in rotated frame)
    float tx = dh.a * cth;
    float ty = dh.a * sth;
    float tz = dh.d;
    
    return Transform3D(rot, tx, ty, tz);
}

// Robot model with hierarchical spheres
struct RobotModel {
    struct Link {
        float length;
        float radius;
        std::vector<Sphere3D> spheres_level0; // Coarse: 1 sphere
        std::vector<Sphere3D> spheres_level1; // Medium: 2-3 spheres
        std::vector<Sphere3D> spheres_level2; // Fine: 4+ spheres
    };
    
    struct Kinematics {
        std::vector<DHParams> dh_params;  // DH parameters for each link
    };
    
    std::vector<Link> links;
    Kinematics kinematics;
    
    void addLink(float length, float radius) {
        Link link;
        link.length = length;
        link.radius = radius;
        
        // Level 0: Single coarse sphere covering entire link (reduced radius to avoid floor collision)
        link.spheres_level0.emplace_back(length*0.5f, 0, 0, length*0.5f + radius, 0, links.size());
        
        // Level 1: Two medium spheres
        link.spheres_level1.emplace_back(length*0.25f, 0, 0, length*0.3f + radius, 1, links.size());
        link.spheres_level1.emplace_back(length*0.75f, 0, 0, length*0.3f + radius, 1, links.size());
        
        // Level 2: Four fine spheres
        link.spheres_level2.emplace_back(length*0.125f, 0, 0, radius, 2, links.size());
        link.spheres_level2.emplace_back(length*0.375f, 0, 0, radius, 2, links.size());
        link.spheres_level2.emplace_back(length*0.625f, 0, 0, radius, 2, links.size());
        link.spheres_level2.emplace_back(length*0.875f, 0, 0, radius, 2, links.size());
        
        links.push_back(link);
    }
    
    // Create a simple 6-DOF arm with DH parameters (generic values for demonstration)
    void createSimple6DOFArm() {
        links.clear();
        kinematics.dh_params.clear();
        
        // Define DH parameters for 6-DOF arm (generic values)
        // For a real robot, use manufacturer's specifications
        std::vector<DHParams> dh = {
            DHParams(0.0f, 1.5708f, 0.333f, 0.0f),      // Link 1: a=0, alpha=90°, d=0.333
            DHParams(-0.316f, 0.0f, 0.0f, 0.0f),        // Link 2: a=-0.316, alpha=0, d=0
            DHParams(-0.0825f, -1.5708f, 0.384f, 0.0f),  // Link 3: a=-0.0825, alpha=-90°, d=0.384
            DHParams(0.0f, 1.5708f, 0.0f, 0.0f),        // Link 4: a=0, alpha=90°, d=0
            DHParams(0.088f, -1.5708f, 0.107f, 0.0f),   // Link 5: a=0.088, alpha=-90°, d=0.107
            DHParams(0.0f, 0.0f, 0.103f, 0.0f)          // Link 6: a=0, alpha=0, d=0.103
        };
        
        // Link lengths (approximate from DH parameters)
        std::vector<float> lengths = {0.333f, 0.316f, 0.384f, 0.0f, 0.107f, 0.103f};
        std::vector<float> radii = {0.1f, 0.1f, 0.08f, 0.08f, 0.06f, 0.06f};
        
        for (size_t i = 0; i < 6; ++i) {
            kinematics.dh_params.push_back(dh[i]);
            addLink(lengths[i], radii[i]);
        }
    }
    
    // Create a planar arm (2D, for backward compatibility)
    void createPlanarArm(const std::vector<float>& lengths, const std::vector<float>& radii) {
        links.clear();
        kinematics.dh_params.clear();
        
        for (size_t i = 0; i < lengths.size(); ++i) {
            // Planar arm: all joints rotate about Z, no offsets
            kinematics.dh_params.push_back(DHParams(lengths[i], 0.0f, 0.0f, 0.0f));
            addLink(lengths[i], radii[i]);
        }
    }
};
// ==================================================================================

// ----------------------------- ENVIRONMENT --------------------------------
struct Environment {
    alignas(32) std::vector<float> box_min_x, box_min_y, box_min_z;
    alignas(32) std::vector<float> box_max_x, box_max_y, box_max_z;
    alignas(32) std::vector<float> sphere_x, sphere_y, sphere_z, sphere_r;

    size_t num_boxes = 0;
    size_t num_spheres = 0;

    void addBox(float minx, float miny, float minz, float maxx, float maxy, float maxz) {
        box_min_x.push_back(minx); box_min_y.push_back(miny); box_min_z.push_back(minz);
        box_max_x.push_back(maxx); box_max_y.push_back(maxy); box_max_z.push_back(maxz);
        num_boxes = box_min_x.size();
    }

    void addSphere(float x, float y, float z, float r) {
        sphere_x.push_back(x); sphere_y.push_back(y); sphere_z.push_back(z); sphere_r.push_back(r);
        num_spheres = sphere_x.size();
    }

    void padToSIMDWidth() {
        // Use safe padding values to avoid NaN issues with -ffast-math
        while (box_min_x.size() % SIMD_WIDTH != 0) {
            box_min_x.push_back(PAD_COORD); box_min_y.push_back(PAD_COORD); box_min_z.push_back(PAD_COORD);
            box_max_x.push_back(PAD_COORD); box_max_y.push_back(PAD_COORD); box_max_z.push_back(PAD_COORD);
        }
        while (sphere_x.size() % SIMD_WIDTH != 0) {
            sphere_x.push_back(PAD_COORD); sphere_y.push_back(PAD_COORD); 
            sphere_z.push_back(PAD_COORD); sphere_r.push_back(SPHERE_PAD_RADIUS);
        }
        num_boxes = box_min_x.size();
        num_spheres = sphere_x.size();
    }

    size_t actualNumBoxes() const {
        // Check for padded sentinel values
        for (size_t i = 0; i < box_min_x.size(); ++i) {
            if (box_min_x[i] >= PAD_SENTINEL) return i;
        }
        return box_min_x.size();
    }
    
    size_t actualNumSpheres() const {
        // Check for padded sentinel (negative radius)
        for (size_t i = 0; i < sphere_r.size(); ++i) {
            if (sphere_r[i] < 0.0f) return i;
        }
        return sphere_r.size();
    }
};

// ==================== VECTORIZED COLLISION CHECKING ====================
// CORRECTED: Proper obstacle-broadcasting pattern (VAMP Section III.B)
// Inputs:
//   cx, cy, cz, radius2: 8 robot sphere positions/radii (one per config)
//   box_min/max_*: Block of up to 8 obstacle boxes
//   lane_count: Number of valid obstacles in this block
// Returns: Bitmask where bit i = 1 if config i collides with ANY obstacle in block
static inline int vectorizedSphereBoxBatchTest(
        __m256 cx, __m256 cy, __m256 cz,
        __m256 radius2,
        const float *box_min_x,
        const float *box_min_y,
        const float *box_min_z,
        const float *box_max_x,
        const float *box_max_y,
        const float *box_max_z,
        int lane_count = 8)
{
    uint32_t collision_mask = 0u;
    __m256 zero = set1_8(0.0f);

    // KEY VAMP PATTERN: Broadcast each obstacle and test against all 8 configs
    for (int obs_idx = 0; obs_idx < lane_count; ++obs_idx) {
        // Broadcast obstacle bounds to all lanes
        __m256 bxmin = set1_8(box_min_x[obs_idx]);
        __m256 bymin = set1_8(box_min_y[obs_idx]);
        __m256 bzmin = set1_8(box_min_z[obs_idx]);
        __m256 bxmax = set1_8(box_max_x[obs_idx]);
        __m256 bymax = set1_8(box_max_y[obs_idx]);
        __m256 bzmax = set1_8(box_max_z[obs_idx]);
        
        // Compute distance from each config sphere to obstacle box
        // dx = max(bxmin - cx, 0, cx - bxmax)
        __m256 dx_left = _mm256_sub_ps(bxmin, cx);
        __m256 dx_right = _mm256_sub_ps(cx, bxmax);
        __m256 dx = _mm256_max_ps(_mm256_max_ps(dx_left, dx_right), zero);
        
        __m256 dy_left = _mm256_sub_ps(bymin, cy);
        __m256 dy_right = _mm256_sub_ps(cy, bymax);
        __m256 dy = _mm256_max_ps(_mm256_max_ps(dy_left, dy_right), zero);
        
        __m256 dz_left = _mm256_sub_ps(bzmin, cz);
        __m256 dz_right = _mm256_sub_ps(cz, bzmax);
        __m256 dz = _mm256_max_ps(_mm256_max_ps(dz_left, dz_right), zero);
        
        // dist_sq = dx² + dy² + dz²
        __m256 dist_sq = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
            _mm256_mul_ps(dz, dz)
        );
        
        // Compare: dist_sq <= radius²
        __m256 collision = _mm256_cmp_ps(dist_sq, radius2, _CMP_LE_OQ);
        int mask = movemask8(collision);
        
        // Accumulate collisions
        collision_mask |= static_cast<uint32_t>(mask);
        
        // Early exit: if all configs collided, no need to check more obstacles
        if (collision_mask == 0xFFu) break;
    }
    
    return static_cast<int>(collision_mask & 0xFFu);
}

// ==================== VECTORIZED SPHERE-SPHERE COLLISION ====================
// CORRECTED: Sphere-sphere collision with proper broadcasting
static inline int vectorizedSphereSphereBatchTest(
        __m256 cx, __m256 cy, __m256 cz, __m256 cr,
        const float *ox, const float *oy, const float *oz, const float *orad,
        int lane_count = 8)
{
    uint32_t collision_mask = 0u;
    
    // KEY VAMP PATTERN: Broadcast each obstacle and test against all 8 configs
    for (int obs_idx = 0; obs_idx < lane_count; ++obs_idx) {
        // Broadcast obstacle sphere to all lanes
        __m256 ocx = set1_8(ox[obs_idx]);
        __m256 ocy = set1_8(oy[obs_idx]);
        __m256 ocz = set1_8(oz[obs_idx]);
        __m256 orr = set1_8(orad[obs_idx]);
        
        // Compute distance between config spheres and obstacle sphere
        __m256 dx = _mm256_sub_ps(cx, ocx);
        __m256 dy = _mm256_sub_ps(cy, ocy);
        __m256 dz = _mm256_sub_ps(cz, ocz);

        __m256 dist_sq = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
            _mm256_mul_ps(dz, dz)
        );
        
        // Collision if dist² <= (r1 + r2)²
        __m256 r_sum = _mm256_add_ps(cr, orr);
        __m256 r_sum_sq = _mm256_mul_ps(r_sum, r_sum);
        
        __m256 collision = _mm256_cmp_ps(dist_sq, r_sum_sq, _CMP_LE_OQ);
        int mask = movemask8(collision);
        
        // Accumulate collisions
        collision_mask |= static_cast<uint32_t>(mask);
        
        // Early exit
        if (collision_mask == 0xFFu) break;
    }
    
    return static_cast<int>(collision_mask & 0xFFu);
}

// ==================== VECTORIZED FORWARD KINEMATICS (HELPER - UNUSED) ====================
// NOTE: This function is an EXAMPLE/HELPER and is NOT used by the interleaved FK+CC path.
// The interleaved path (computeFK_Interleaved_6DOF) computes per-link centers itself.
// This function is kept for reference only - it returns only end-effector positions, not per-link.
// If you need per-link centers, use computeFK_Interleaved_6DOF instead.
namespace detail {
struct LinkCenters8 {
    __m256 cx, cy, cz;
};

// inputs: joints[ndof][8] as contiguous SOA arrays (each joint index points to 8 floats)
// link_lengths: array of floats (ndof)
// Returns: end-effector center positions only (not per-link, unlike interleaved version)
[[maybe_unused]] static inline LinkCenters8 vectorizedFK_endEffector_only_for_test(
        const float *joint0_8, const float *joint1_8, const float *joint2_8,
        const float *joint3_8, const float *joint4_8, const float *joint5_8,
        const float *link_lengths) 
{
    // load joint angles per-lane
    __m256 j0 = load8(joint0_8);
    __m256 j1 = load8(joint1_8);
    __m256 j2 = load8(joint2_8);
    __m256 j3 = load8(joint3_8);
    __m256 j4 = load8(joint4_8);
    __m256 j5 = load8(joint5_8);

    // compute sines and cosines for first joint only (others use cumulative angles)
    // Prefer vector trig (SLEEF) if available:
#ifdef USE_SLEEF
    // SLEEF for AVX2: split 8-float vector into two 4-float vectors
    __m128 j0_low = _mm256_extractf128_ps(j0, 0);
    __m128 j0_high = _mm256_extractf128_ps(j0, 1);
    __m128 s0_low = Sleef_sinf4_u35(j0_low);
    __m128 s0_high = Sleef_sinf4_u35(j0_high);
    __m128 c0_low = Sleef_cosf4_u35(j0_low);
    __m128 c0_high = Sleef_cosf4_u35(j0_high);
    __m256 s0 = _mm256_set_m128(s0_high, s0_low);
    __m256 c0 = _mm256_set_m128(c0_high, c0_low);
#else
    // fallback: compute sin/cos scalar per-lane and pack into vectors
    alignas(32) float a0[8];
    _mm256_storeu_ps(a0, j0);
    alignas(32) float s0s[8], c0s[8];
    for (int i=0;i<8;++i) {
        s0s[i] = sinf(a0[i]);
        c0s[i] = cosf(a0[i]);
    }
    // Use aligned loads for aligned temporary arrays (micro-optimization)
    __m256 s0 = _mm256_load_ps(s0s);
    __m256 c0 = _mm256_load_ps(c0s);
#endif

    // Example forward kinematics (2D planar chain stacked into z=0). Replace with your robot FK.
    // link_lengths[0..5]
    __m256 L0 = set1_8(link_lengths[0]);
    __m256 L1 = set1_8(link_lengths[1]);
    __m256 L2 = set1_8(link_lengths[2]);
    __m256 L3 = set1_8(link_lengths[3]);
    __m256 L4 = set1_8(link_lengths[4]);
    __m256 L5 = set1_8(link_lengths[5]);

    // compute cumulative angles and positions vectorized
    // For planar arm: cumulative angle = sum of previous joints
    __m256 th0 = j0;
    __m256 x0 = _mm256_mul_ps(L0, c0);
    __m256 y0 = _mm256_mul_ps(L0, s0);

    __m256 th1 = _mm256_add_ps(th0, j1);
    // Compute cos/sin of cumulative angle th1
#ifdef USE_SLEEF
    __m128 th1_low = _mm256_extractf128_ps(th1, 0);
    __m128 th1_high = _mm256_extractf128_ps(th1, 1);
    __m128 cth1_low = Sleef_cosf4_u35(th1_low);
    __m128 cth1_high = Sleef_cosf4_u35(th1_high);
    __m128 sth1_low = Sleef_sinf4_u35(th1_low);
    __m128 sth1_high = Sleef_sinf4_u35(th1_high);
    __m256 cth1 = _mm256_set_m128(cth1_high, cth1_low);
    __m256 sth1 = _mm256_set_m128(sth1_high, sth1_low);
#else
    alignas(32) float th1_arr[8];
    _mm256_storeu_ps(th1_arr, th1);
    alignas(32) float cth1_arr[8], sth1_arr[8];
    for (int i=0; i<8; ++i) {
        cth1_arr[i] = cosf(th1_arr[i]);
        sth1_arr[i] = sinf(th1_arr[i]);
    }
    __m256 cth1 = _mm256_load_ps(cth1_arr);
    __m256 sth1 = _mm256_load_ps(sth1_arr);
#endif
    __m256 x1 = _mm256_add_ps(x0, _mm256_mul_ps(L1, cth1));
    __m256 y1 = _mm256_add_ps(y0, _mm256_mul_ps(L1, sth1));

    __m256 th2 = _mm256_add_ps(th1, j2);
#ifdef USE_SLEEF
    __m128 th2_low = _mm256_extractf128_ps(th2, 0);
    __m128 th2_high = _mm256_extractf128_ps(th2, 1);
    __m128 cth2_low = Sleef_cosf4_u35(th2_low);
    __m128 cth2_high = Sleef_cosf4_u35(th2_high);
    __m128 sth2_low = Sleef_sinf4_u35(th2_low);
    __m128 sth2_high = Sleef_sinf4_u35(th2_high);
    __m256 cth2 = _mm256_set_m128(cth2_high, cth2_low);
    __m256 sth2 = _mm256_set_m128(sth2_high, sth2_low);
#else
    alignas(32) float th2_arr[8];
    _mm256_storeu_ps(th2_arr, th2);
    alignas(32) float cth2_arr[8], sth2_arr[8];
    for (int i=0; i<8; ++i) {
        cth2_arr[i] = cosf(th2_arr[i]);
        sth2_arr[i] = sinf(th2_arr[i]);
    }
    __m256 cth2 = _mm256_load_ps(cth2_arr);
    __m256 sth2 = _mm256_load_ps(sth2_arr);
#endif
    __m256 x2 = _mm256_add_ps(x1, _mm256_mul_ps(L2, cth2));
    __m256 y2 = _mm256_add_ps(y1, _mm256_mul_ps(L2, sth2));

    __m256 th3 = _mm256_add_ps(th2, j3);
#ifdef USE_SLEEF
    __m128 th3_low = _mm256_extractf128_ps(th3, 0);
    __m128 th3_high = _mm256_extractf128_ps(th3, 1);
    __m128 cth3_low = Sleef_cosf4_u35(th3_low);
    __m128 cth3_high = Sleef_cosf4_u35(th3_high);
    __m128 sth3_low = Sleef_sinf4_u35(th3_low);
    __m128 sth3_high = Sleef_sinf4_u35(th3_high);
    __m256 cth3 = _mm256_set_m128(cth3_high, cth3_low);
    __m256 sth3 = _mm256_set_m128(sth3_high, sth3_low);
#else
    alignas(32) float th3_arr[8];
    _mm256_storeu_ps(th3_arr, th3);
    alignas(32) float cth3_arr[8], sth3_arr[8];
    for (int i=0; i<8; ++i) {
        cth3_arr[i] = cosf(th3_arr[i]);
        sth3_arr[i] = sinf(th3_arr[i]);
    }
    __m256 cth3 = _mm256_load_ps(cth3_arr);
    __m256 sth3 = _mm256_load_ps(sth3_arr);
#endif
    __m256 x3 = _mm256_add_ps(x2, _mm256_mul_ps(L3, cth3));
    __m256 y3 = _mm256_add_ps(y2, _mm256_mul_ps(L3, sth3));

    __m256 th4 = _mm256_add_ps(th3, j4);
#ifdef USE_SLEEF
    __m128 th4_low = _mm256_extractf128_ps(th4, 0);
    __m128 th4_high = _mm256_extractf128_ps(th4, 1);
    __m128 cth4_low = Sleef_cosf4_u35(th4_low);
    __m128 cth4_high = Sleef_cosf4_u35(th4_high);
    __m128 sth4_low = Sleef_sinf4_u35(th4_low);
    __m128 sth4_high = Sleef_sinf4_u35(th4_high);
    __m256 cth4 = _mm256_set_m128(cth4_high, cth4_low);
    __m256 sth4 = _mm256_set_m128(sth4_high, sth4_low);
#else
    alignas(32) float th4_arr[8];
    _mm256_storeu_ps(th4_arr, th4);
    alignas(32) float cth4_arr[8], sth4_arr[8];
    for (int i=0; i<8; ++i) {
        cth4_arr[i] = cosf(th4_arr[i]);
        sth4_arr[i] = sinf(th4_arr[i]);
    }
    __m256 cth4 = _mm256_load_ps(cth4_arr);
    __m256 sth4 = _mm256_load_ps(sth4_arr);
#endif
    __m256 x4 = _mm256_add_ps(x3, _mm256_mul_ps(L4, cth4));
    __m256 y4 = _mm256_add_ps(y3, _mm256_mul_ps(L4, sth4));

    __m256 th5 = _mm256_add_ps(th4, j5);
#ifdef USE_SLEEF
    __m128 th5_low = _mm256_extractf128_ps(th5, 0);
    __m128 th5_high = _mm256_extractf128_ps(th5, 1);
    __m128 cth5_low = Sleef_cosf4_u35(th5_low);
    __m128 cth5_high = Sleef_cosf4_u35(th5_high);
    __m128 sth5_low = Sleef_sinf4_u35(th5_low);
    __m128 sth5_high = Sleef_sinf4_u35(th5_high);
    __m256 cth5 = _mm256_set_m128(cth5_high, cth5_low);
    __m256 sth5 = _mm256_set_m128(sth5_high, sth5_low);
#else
    alignas(32) float th5_arr[8];
    _mm256_storeu_ps(th5_arr, th5);
    alignas(32) float cth5_arr[8], sth5_arr[8];
    for (int i=0; i<8; ++i) {
        cth5_arr[i] = cosf(th5_arr[i]);
        sth5_arr[i] = sinf(th5_arr[i]);
    }
    __m256 cth5 = _mm256_load_ps(cth5_arr);
    __m256 sth5 = _mm256_load_ps(sth5_arr);
#endif
    __m256 x5 = _mm256_add_ps(x4, _mm256_mul_ps(L5, cth5));
    __m256 y5 = _mm256_add_ps(y4, _mm256_mul_ps(L5, sth5));

    // choose which link center to return — here we return end-effector center
    LinkCenters8 out;
    out.cx = x5;
    out.cy = y5;
    out.cz = set1_8(0.0f);
    return out;
}
} // namespace detail
// ==================================================================================

// ============================ VECTORIZED FK + INTERLEAVED CC ============================
// KEY VAMP CONTRIBUTION: FK and collision checking are interleaved
// Check each sphere immediately after computing its position, enabling early exit
class VectorizedFK_InterleavedCC {
public:
    const RobotModel& robot_model;
    const Environment& env;
    
    VectorizedFK_InterleavedCC(const RobotModel& model, const Environment& e) 
        : robot_model(model), env(e) {}
    
    // VECTORIZED FK FOR 6-DOF ARM - Per-link interleaved FK+CC (True VAMP implementation)
    // Returns collision mask: bit i set if config i collides
    // Key: Computes FK link-by-link and checks hierarchical spheres immediately per-link
    // CORRECTED: True per-sphere FK/CC interleaving with proper hierarchical logic
    uint8_t computeFK_Interleaved_6DOF(const VectorizedConfig3DArm& configs) const {
        if (configs.num_joints != 6) {
            std::cerr << "Error: 6DOF FK called with " << configs.num_joints << " joints\n";
            return 0xFF;
        }
        
        if (robot_model.links.size() < 6) {
            std::cerr << "Error: Robot model has " << robot_model.links.size() << " links, expected 6\n";
            return 0xFF;
        }
        
        // Load joint angles as vectors
        __m256 j[6];
        for (int i = 0; i < 6; ++i) {
            j[i] = configs.getJoint(i);
        }
        
        uint32_t collision_mask = 0u;
        uint32_t active_lanes = 0xFFu;  // All 8 lanes start active
        
        size_t actual_boxes = env.actualNumBoxes();
        size_t actual_spheres = env.actualNumSpheres();
        
        // Store SE(3) transforms per lane
        alignas(32) Transform3D T_world[SIMD_WIDTH];
        for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
            T_world[lane].setIdentity();
        }
        
        bool use_3d = (robot_model.kinematics.dh_params.size() >= 6);
        
        // ===== KEY VAMP INSIGHT: PER-SPHERE FK/CC INTERLEAVING =====
        // Process each link, checking spheres IMMEDIATELY after computing their positions
        for (int link_idx = 0; link_idx < 6 && active_lanes != 0; ++link_idx) {
            const auto& link = robot_model.links[static_cast<size_t>(link_idx)];
            
            // === Step 1: Update transforms for this link ===
            if (use_3d && link_idx < static_cast<int>(robot_model.kinematics.dh_params.size())) {
                const DHParams& dh = robot_model.kinematics.dh_params[static_cast<size_t>(link_idx)];
                alignas(32) float theta[SIMD_WIDTH];
                _mm256_store_ps(theta, j[link_idx]);
                
                for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
                    Transform3D T_local = dhTransform(dh, theta[lane]);
                    T_world[lane] = T_world[lane].compose(T_local);
                }
            } else {
                // Planar fallback
                __m256 th = set1_8(0.0f);
                for (int k = 0; k <= link_idx; ++k) {
                    th = _mm256_add_ps(th, j[k]);
                }
                
                alignas(32) float th_arr[SIMD_WIDTH];
                _mm256_storeu_ps(th_arr, th);
                alignas(32) float cth_arr[SIMD_WIDTH], sth_arr[SIMD_WIDTH];
                for (size_t i = 0; i < SIMD_WIDTH; ++i) {
                    cth_arr[i] = cosf(th_arr[i]);
                    sth_arr[i] = sinf(th_arr[i]);
                }
                
                for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
                    float L = link.length;
                    T_world[lane].tx += L * cth_arr[lane];
                    T_world[lane].ty += L * sth_arr[lane];
                }
            }
            
            // === Step 2: HIERARCHICAL SPHERE CHECKING ===
            // CORRECTED LOGIC (per 41.md): Coarse/Medium are conservative bounds, only Fine is definite
            // VAMP's actual logic:
            //   - Coarse hit → must check medium
            //   - Medium hit → must check fine
            //   - Only fine hits are definitive collisions
            //   - No coarse hit → definitely safe (can skip)
            
            // --- Level 0: Coarse Check (Conservative Bounding Sphere) ---
            uint32_t coarse_hits = 0u;
            
            if (!link.spheres_level0.empty()) {
                const Sphere3D& coarse = link.spheres_level0[0];
                
                // Transform coarse sphere center to world frame
                alignas(32) float world_x[SIMD_WIDTH], world_y[SIMD_WIDTH], world_z[SIMD_WIDTH];
                for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
                    T_world[lane].transformPoint(coarse.x, coarse.y, coarse.z,
                                                world_x[lane], world_y[lane], world_z[lane]);
                }
                __m256 cx = _mm256_load_ps(world_x);
                __m256 cy = _mm256_load_ps(world_y);
                __m256 cz = _mm256_load_ps(world_z);
                __m256 cr2 = set1_8(coarse.r * coarse.r);
                
                // Check coarse sphere against all obstacles
                // Check boxes
                for (size_t b = 0; b < actual_boxes; b += 8) {
                    size_t block_size = std::min(8UL, actual_boxes - b);
                    int hit_mask = vectorizedSphereBoxBatchTest(
                        cx, cy, cz, cr2,
                            &env.box_min_x[b], &env.box_min_y[b], &env.box_min_z[b],
                            &env.box_max_x[b], &env.box_max_y[b], &env.box_max_z[b],
                        static_cast<int>(block_size));
                    coarse_hits |= (static_cast<uint32_t>(hit_mask) & active_lanes);
                }
                
                // Check spheres
                if (actual_spheres > 0) {
                    __m256 cr_vec = set1_8(coarse.r);
                    for (size_t b = 0; b < actual_spheres; b += 8) {
                        size_t block_size = std::min(8UL, actual_spheres - b);
                        int hit_mask = vectorizedSphereSphereBatchTest(
                            cx, cy, cz, cr_vec,
                                &env.sphere_x[b], &env.sphere_y[b], &env.sphere_z[b], &env.sphere_r[b],
                            static_cast<int>(block_size));
                        coarse_hits |= (static_cast<uint32_t>(hit_mask) & active_lanes);
                    }
                }
            }
            
            // VAMP KEY INSIGHT: Lanes with NO coarse hit are DEFINITELY SAFE for this link
            // Skip medium/fine checks for those lanes
            if (coarse_hits == 0) {
                continue;  // All active lanes passed coarse check - skip to next link
            }
            
            // --- Level 1: Medium Check (Tighter Bound) ---
            // Only check medium spheres for lanes that had coarse hits
            uint32_t medium_hits = 0u;
            
            if (!link.spheres_level1.empty()) {
                // Check ALL medium spheres, accumulate hits
                for (const auto& medium_sphere : link.spheres_level1) {
                    // Transform sphere to world frame
                    alignas(32) float world_x[SIMD_WIDTH], world_y[SIMD_WIDTH], world_z[SIMD_WIDTH];
                    for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
                        T_world[lane].transformPoint(medium_sphere.x, medium_sphere.y, medium_sphere.z,
                                                    world_x[lane], world_y[lane], world_z[lane]);
                    }
                    __m256 mx = _mm256_load_ps(world_x);
                    __m256 my = _mm256_load_ps(world_y);
                    __m256 mz = _mm256_load_ps(world_z);
                    __m256 mr2 = set1_8(medium_sphere.r * medium_sphere.r);
                    
                    // Check against boxes
                    for (size_t b = 0; b < actual_boxes; b += 8) {
                        size_t block_size = std::min(8UL, actual_boxes - b);
                        int hit_mask = vectorizedSphereBoxBatchTest(
                            mx, my, mz, mr2,
                                &env.box_min_x[b], &env.box_min_y[b], &env.box_min_z[b],
                                &env.box_max_x[b], &env.box_max_y[b], &env.box_max_z[b],
                            static_cast<int>(block_size));
                        medium_hits |= (static_cast<uint32_t>(hit_mask) & coarse_hits);
                    }
                    
                    // Check against spheres
                    if (actual_spheres > 0) {
                        __m256 mr_vec = set1_8(medium_sphere.r);
                        for (size_t b = 0; b < actual_spheres; b += 8) {
                            size_t block_size = std::min(8UL, actual_spheres - b);
                            int hit_mask = vectorizedSphereSphereBatchTest(
                                mx, my, mz, mr_vec,
                                    &env.sphere_x[b], &env.sphere_y[b], &env.sphere_z[b], &env.sphere_r[b],
                                static_cast<int>(block_size));
                            medium_hits |= (static_cast<uint32_t>(hit_mask) & coarse_hits);
                        }
                    }
                }
            }
            
            // ----------------- Hierarchical-sphere invariant -----------------
            // We use a nested sphere hierarchy: coarse ⊇ medium ⊇ fine.
            // - If coarse has NO hit (coarse_hits == 0) then the lane is definitely safe.
            // - If coarse has a hit, we check medium (a tighter bound inside coarse).
            // - If medium has NO hit, then fine-level spheres (which lie inside medium) cannot hit.
            //   Therefore it is correct and safe to skip fine checks for lanes that passed medium.
            // - Only fine_hits are used to mark definitive collisions.
            //
            // This is the intended optimization pattern described in VAMP and is *correct*
            // provided your medium spheres are geometrically inside the coarse spheres.
            if (medium_hits == 0) {
                // No medium hits → no fine hits are possible for those lanes → skip fine.
                continue;
            }
            
            // --- Level 2: Fine Check (Exact Geometry) ---
            // Only check fine spheres for lanes that had medium hits
            // CORRECTED: Only fine-level collisions are DEFINITE collisions
            uint32_t fine_hits = 0u;
                    
                    for (const auto& fine_sphere : link.spheres_level2) {
                if (medium_hits == 0) break;  // Early exit if all lanes cleared
                    
                // Transform sphere to world frame
                    alignas(32) float world_x[SIMD_WIDTH], world_y[SIMD_WIDTH], world_z[SIMD_WIDTH];
                    for (size_t lane = 0; lane < SIMD_WIDTH; ++lane) {
                        T_world[lane].transformPoint(fine_sphere.x, fine_sphere.y, fine_sphere.z,
                                                    world_x[lane], world_y[lane], world_z[lane]);
                    }
                __m256 fx = _mm256_load_ps(world_x);
                __m256 fy = _mm256_load_ps(world_y);
                __m256 fz = _mm256_load_ps(world_z);
                __m256 fr2 = set1_8(fine_sphere.r * fine_sphere.r);
                
                // Check against boxes
                    for (size_t b = 0; b < actual_boxes; b += 8) {
                        size_t block_size = std::min(8UL, actual_boxes - b);
                    int hit_mask = vectorizedSphereBoxBatchTest(
                        fx, fy, fz, fr2,
                                &env.box_min_x[b], &env.box_min_y[b], &env.box_min_z[b],
                                &env.box_max_x[b], &env.box_max_y[b], &env.box_max_z[b],
                        static_cast<int>(block_size));
                    fine_hits |= (static_cast<uint32_t>(hit_mask) & medium_hits);
                }
                
                // Check against spheres
                    if (actual_spheres > 0) {
                    __m256 fr_vec = set1_8(fine_sphere.r);
                        for (size_t b = 0; b < actual_spheres; b += 8) {
                            size_t block_size = std::min(8UL, actual_spheres - b);
                        int hit_mask = vectorizedSphereSphereBatchTest(
                            fx, fy, fz, fr_vec,
                                    &env.sphere_x[b], &env.sphere_y[b], &env.sphere_z[b], &env.sphere_r[b],
                            static_cast<int>(block_size));
                        fine_hits |= (static_cast<uint32_t>(hit_mask) & medium_hits);
                    }
                }
            }
            
            // CORRECTED: Fine hits are DEFINITE collisions - mark and deactivate
            if (fine_hits != 0) {
                collision_mask |= fine_hits;
                active_lanes &= ~fine_hits;
                
                // Early exit if all lanes collided
                if (collision_mask == 0xFFu) break;
            }
            
            // Lanes with medium hits but NOT fine hits are CLEAR (passed all checks)
        }
        
        return static_cast<uint8_t>(collision_mask & 0xFFu);
    }
    
    // Generic FK for any DOF (less optimized but general)
    uint8_t computeFK_Interleaved_Generic(const VectorizedConfig3DArm& configs) const {
        uint8_t collision_mask = 0;
        size_t actual_boxes = env.actualNumBoxes();
        size_t actual_spheres = env.actualNumSpheres();
        
        // Check if we should use 3D DH transforms (fix from 39.md)
        bool use_3d = (robot_model.kinematics.dh_params.size() >= static_cast<size_t>(configs.num_joints));
        
        // Store transforms per lane (for 3D mode)
        std::vector<Transform3D> T_world(SIMD_WIDTH);
        for (auto& T : T_world) T.setIdentity();
        
        // For planar mode fallback
        alignas(32) float link_end_x[SIMD_WIDTH] = {0};
        alignas(32) float link_end_y[SIMD_WIDTH] = {0};
        alignas(32) float link_end_z[SIMD_WIDTH] = {0};
        alignas(32) float cumulative_angle[SIMD_WIDTH] = {0};
        
        for (size_t link_idx = 0; link_idx < std::min((size_t)configs.num_joints, robot_model.links.size()); ++link_idx) {
            const auto& link = robot_model.links[link_idx];
            
            alignas(32) float joint_angles[SIMD_WIDTH];
            _mm256_store_ps(joint_angles, configs.getJoint(static_cast<int>(link_idx)));
            
            alignas(32) float new_x[SIMD_WIDTH], new_y[SIMD_WIDTH], new_z[SIMD_WIDTH];
            
            if (use_3d && link_idx < robot_model.kinematics.dh_params.size()) {
                // 3D mode: Use DH transforms (fix from 39.md)
                const DHParams& dh = robot_model.kinematics.dh_params[link_idx];
                
                for (size_t cfg = 0; cfg < SIMD_WIDTH; ++cfg) {
                    if (collision_mask & (1 << cfg)) continue;
                    
                    Transform3D T_local = dhTransform(dh, joint_angles[cfg]);
                    T_world[cfg] = T_world[cfg].compose(T_local);
                    
                    // Store endpoint for this link
                    new_x[cfg] = T_world[cfg].tx;
                    new_y[cfg] = T_world[cfg].ty;
                    new_z[cfg] = T_world[cfg].tz;
                }
            } else {
                // Planar mode fallback
                for (size_t cfg = 0; cfg < SIMD_WIDTH; ++cfg) {
                    if (collision_mask & (1 << cfg)) continue;
                    
                    cumulative_angle[cfg] += joint_angles[cfg];
                    float angle = cumulative_angle[cfg];
                    
                float c = std::cos(angle);
                float s = std::sin(angle);
                
                float local_x = link.length;
                float rotated_x = c * local_x;
                float rotated_y = s * local_x;
                
                new_x[cfg] = link_end_x[cfg] + rotated_x;
                new_y[cfg] = link_end_y[cfg] + rotated_y;
                new_z[cfg] = link_end_z[cfg];
                }
            }
            
            for (size_t cfg = 0; cfg < SIMD_WIDTH; ++cfg) {
                if (collision_mask & (1 << cfg)) continue;
                
                // Get coarse sphere center in world frame (fix from 39.md)
                float cx, cy, cz;
                const auto& coarse = link.spheres_level0[0];
                
                if (use_3d && link_idx < robot_model.kinematics.dh_params.size()) {
                    // Use transform to get sphere center
                    T_world[cfg].transformPoint(coarse.x, coarse.y, coarse.z, cx, cy, cz);
                } else {
                    // Planar mode: interpolate between link endpoints
                    cx = (link_end_x[cfg] + new_x[cfg]) * 0.5f;
                    cy = (link_end_y[cfg] + new_y[cfg]) * 0.5f;
                    cz = (link_end_z[cfg] + new_z[cfg]) * 0.5f;
                }
                
                float cr = coarse.r;
                
                bool coarse_hit = false;
                for (size_t bi = 0; bi < actual_boxes && !coarse_hit; ++bi) {
                    float dx = std::max({env.box_min_x[bi] - cx, 0.0f, cx - env.box_max_x[bi]});
                    float dy = std::max({env.box_min_y[bi] - cy, 0.0f, cy - env.box_max_y[bi]});
                    float dz = std::max({env.box_min_z[bi] - cz, 0.0f, cz - env.box_max_z[bi]});
                    if (dx*dx + dy*dy + dz*dz <= cr*cr) coarse_hit = true;
                }
                for (size_t si = 0; si < actual_spheres && !coarse_hit; ++si) {
                    float dx = env.sphere_x[si] - cx;
                    float dy = env.sphere_y[si] - cy;
                    float dz = env.sphere_z[si] - cz;
                    float r_sum = cr + env.sphere_r[si];
                    if (dx*dx + dy*dy + dz*dz <= r_sum*r_sum) coarse_hit = true;
                }
                
                if (!coarse_hit) continue;
                
                bool medium_hit = false;
                if (!link.spheres_level1.empty()) {
                    for (const auto& medium : link.spheres_level1) {
                        float mx, my, mz;
                        
                        if (use_3d && link_idx < robot_model.kinematics.dh_params.size()) {
                            // Use transform to get medium sphere center (fix from 39.md)
                            T_world[cfg].transformPoint(medium.x, medium.y, medium.z, mx, my, mz);
                        } else {
                            // Planar mode: interpolate
                            float t = medium.x / link.length;
                            mx = link_end_x[cfg] + t * (new_x[cfg] - link_end_x[cfg]);
                            my = link_end_y[cfg] + t * (new_y[cfg] - link_end_y[cfg]);
                            mz = link_end_z[cfg] + t * (new_z[cfg] - link_end_z[cfg]);
                        }
                        
                        float mr = medium.r;
                        
                        for (size_t bi = 0; bi < actual_boxes && !medium_hit; ++bi) {
                            float dx = std::max({env.box_min_x[bi] - mx, 0.0f, mx - env.box_max_x[bi]});
                            float dy = std::max({env.box_min_y[bi] - my, 0.0f, my - env.box_max_y[bi]});
                            float dz = std::max({env.box_min_z[bi] - mz, 0.0f, mz - env.box_max_z[bi]});
                            if (dx*dx + dy*dy + dz*dz <= mr*mr) medium_hit = true;
                        }
                        for (size_t si = 0; si < actual_spheres && !medium_hit; ++si) {
                            float dx = env.sphere_x[si] - mx;
                            float dy = env.sphere_y[si] - my;
                            float dz = env.sphere_z[si] - mz;
                            float r_sum = mr + env.sphere_r[si];
                            if (dx*dx + dy*dy + dz*dz <= r_sum*r_sum) medium_hit = true;
                        }
                        if (medium_hit) break;
                    }
                }
                
                if (!medium_hit) continue;
                
                bool fine_hit = false;
                for (const auto& fine : link.spheres_level2) {
                    float sx, sy, sz;
                    
                    if (use_3d && link_idx < robot_model.kinematics.dh_params.size()) {
                        // Use transform to get fine sphere center (fix from 39.md)
                        T_world[cfg].transformPoint(fine.x, fine.y, fine.z, sx, sy, sz);
                    } else {
                        // Planar mode: interpolate
                        float t = fine.x / link.length;
                        sx = link_end_x[cfg] + t * (new_x[cfg] - link_end_x[cfg]);
                        sy = link_end_y[cfg] + t * (new_y[cfg] - link_end_y[cfg]);
                        sz = link_end_z[cfg] + t * (new_z[cfg] - link_end_z[cfg]);
                    }
                    
                    float sr = fine.r;
                    
                    for (size_t bi = 0; bi < actual_boxes && !fine_hit; ++bi) {
                        float dx = std::max({env.box_min_x[bi] - sx, 0.0f, sx - env.box_max_x[bi]});
                        float dy = std::max({env.box_min_y[bi] - sy, 0.0f, sy - env.box_max_y[bi]});
                        float dz = std::max({env.box_min_z[bi] - sz, 0.0f, sz - env.box_max_z[bi]});
                        if (dx*dx + dy*dy + dz*dz <= sr*sr) fine_hit = true;
                    }
                    for (size_t si = 0; si < actual_spheres && !fine_hit; ++si) {
                        float dx = env.sphere_x[si] - sx;
                        float dy = env.sphere_y[si] - sy;
                        float dz = env.sphere_z[si] - sz;
                        float r_sum = sr + env.sphere_r[si];
                        if (dx*dx + dy*dy + dz*dz <= r_sum*r_sum) fine_hit = true;
                    }
                    if (fine_hit) break;
                }
                
                if (fine_hit) {
                        collision_mask |= (1 << cfg);
                }
            }
            
            // Update link endpoints for next iteration (planar mode only) (fix from 39.md)
            if (!use_3d) {
                for (size_t cfg = 0; cfg < SIMD_WIDTH; ++cfg) {
                    link_end_x[cfg] = new_x[cfg];
                    link_end_y[cfg] = new_y[cfg];
                    link_end_z[cfg] = new_z[cfg];
                }
            }
            
            if (collision_mask == 0xFF) return collision_mask;
        }
        
        return collision_mask;
    }
};  // END of VectorizedFK_InterleavedCC class
// ==================================================================================

// ============================ RAKED MOTION VALIDATOR ==============================
class RakedMotionValidator {
public:
    const VectorizedFK_InterleavedCC& fk_cc;
    
    RakedMotionValidator(const VectorizedFK_InterleavedCC& fk) : fk_cc(fk) {}
    
    bool validateMotion(const State3DArm& start, const State3DArm& goal, 
                       float resolution = 0.1f) const {
        float dist = start.distance(goal);
        if (dist < 1e-8f) return true;
        
        int n_samples = std::max(2, static_cast<int>(std::ceil(dist / resolution)));
        
        // RAKE: Check spatially distributed waypoints [0, n/8, 2n/8, ..., 7n/8]
        std::array<State3DArm, RAKE_WIDTH> rake_configs;
        for (size_t i = 0; i < RAKE_WIDTH; ++i) {
            float t = static_cast<float>(i) / (RAKE_WIDTH - 1);
            rake_configs[i] = start.interpolate(goal, t);
        }
        
        VectorizedConfig3DArm vec_configs(static_cast<int>(start.joint_angles.size()));
        vec_configs.loadFromAOS(rake_configs);
        
        // Check all 8 rake waypoints simultaneously
        uint8_t rake_collision = (start.joint_angles.size() == 6) ?
            fk_cc.computeFK_Interleaved_6DOF(vec_configs) :
            fk_cc.computeFK_Interleaved_Generic(vec_configs);
        
        // Mask out dummy lanes - only check valid samples
        uint8_t valid_mask = (1u << RAKE_WIDTH) - 1u;  // All 8 lanes are valid for rake
        if (rake_collision & valid_mask) return false;
        
        // Comb through gaps
        int samples_per_gap = std::max(1, n_samples / (static_cast<int>(RAKE_WIDTH) - 1));
        if (samples_per_gap <= 1) return true;
        
        std::array<State3DArm, RAKE_WIDTH> batch;
        int batch_count = 0;
        
        for (size_t gap = 0; gap < RAKE_WIDTH - 1; ++gap) {
            const State3DArm& gap_start = rake_configs[gap];
            const State3DArm& gap_end = rake_configs[gap + 1];
            
            for (int s = 1; s < samples_per_gap; ++s) {
                float t = static_cast<float>(s) / static_cast<float>(samples_per_gap);
                batch[static_cast<size_t>(batch_count++)] = gap_start.interpolate(gap_end, t);
                
                if (batch_count == static_cast<int>(RAKE_WIDTH)) {
                    VectorizedConfig3DArm batch_vec(static_cast<int>(start.joint_angles.size()));
                    batch_vec.loadFromAOS(batch);
                    uint8_t collision = (start.joint_angles.size() == 6) ?
                        fk_cc.computeFK_Interleaved_6DOF(batch_vec) :
                        fk_cc.computeFK_Interleaved_Generic(batch_vec);
                    // All lanes are valid in full batch
                    uint8_t valid_mask = (1u << RAKE_WIDTH) - 1u;
                    if (collision & valid_mask) return false;
                    batch_count = 0;
                }
            }
        }
        
        // Check remaining partial batch
        if (batch_count > 0) {
            State3DArm dummy(static_cast<int>(start.joint_angles.size()));
            for (auto& a : dummy.joint_angles) a = 0.0f; // Safe zero position
            for (int i = batch_count; i < static_cast<int>(RAKE_WIDTH); ++i) {
                batch[static_cast<size_t>(i)] = dummy;
            }
            
            VectorizedConfig3DArm batch_vec(static_cast<int>(start.joint_angles.size()));
            batch_vec.loadFromAOS(batch);
            uint8_t collision = (start.joint_angles.size() == 6) ?
                fk_cc.computeFK_Interleaved_6DOF(batch_vec) :
                fk_cc.computeFK_Interleaved_Generic(batch_vec);
            
            for (int i = 0; i < batch_count; ++i) {
                if (collision & (1 << i)) return false;
            }
        }
        
        return true;
    }
};
// ==================================================================================

// ----------------------------- ROBOT CLASS ---------------------------------
class Robot3DArm_VAMP {
public:
    RobotModel model;
    VectorizedFK_InterleavedCC fk_cc;
    RakedMotionValidator motion_validator;
    
    // Constructor for backward compatibility (2D planar)
    Robot3DArm_VAMP(const std::vector<float>& lengths, 
                    const std::vector<float>& radii,
                    const Environment& env)
        : fk_cc(model, env)
        , motion_validator(fk_cc)
    {
        model.createPlanarArm(lengths, radii);
    }
    
    // Constructor with 3D support
    Robot3DArm_VAMP(const Environment& env, bool use_3d = true)
        : fk_cc(model, env)
        , motion_validator(fk_cc)
    {
        if (use_3d) {
            model.createSimple6DOFArm();  // 3D robot with DH parameters
        } else {
            // Default planar arm
            std::vector<float> lengths = {1.0f, 1.0f, 0.8f, 0.8f, 0.6f, 0.6f};
            std::vector<float> radii = {0.1f, 0.1f, 0.08f, 0.08f, 0.06f, 0.06f};
            model.createPlanarArm(lengths, radii);
        }
    }
    
    bool isValid(const State3DArm& state) const {
        std::array<State3DArm, RAKE_WIDTH> batch;
        batch[0] = state;
        
        // Use far-away config for safe state to ensure it never collides
        // (prevents parity issues if zero config happens to collide)
        State3DArm safe(static_cast<int>(state.joint_angles.size()));
        for (auto& a : safe.joint_angles) a = 100.0f;  // Far away from obstacles
        for (size_t i = 1; i < RAKE_WIDTH; ++i) {
            batch[i] = safe;
        }
        
        VectorizedConfig3DArm vec(static_cast<int>(state.joint_angles.size()));
        vec.loadFromAOS(batch);
        
        uint8_t collision;
        if (state.joint_angles.size() == 6)
            collision = fk_cc.computeFK_Interleaved_6DOF(vec);
        else
            collision = fk_cc.computeFK_Interleaved_Generic(vec);
        
        return (collision & 1) == 0;
        }
        
        bool checkMotion(const State3DArm& start, const State3DArm& goal, 
                        float resolution = 0.1f) const {
            return motion_validator.validateMotion(start, goal, resolution);
        }

    };

// --------------------------------------------------------------------------

// ============================= RRT-CONNECT PLANNER =============================
class RRTConnect {
public:
    struct Node {
        State3DArm state;
        int parent_idx;
        Node(const State3DArm& s, int p = -1) : state(s), parent_idx(p) {}
    };
    
    std::vector<Node> tree_start;
    std::vector<Node> tree_goal;
    const Robot3DArm_VAMP& robot;
    std::mt19937 rng;
    std::uniform_real_distribution<float> angle_dist;
    
    float step_size;
    int max_iterations;
    
    RRTConnect(const Robot3DArm_VAMP& r, float step = 0.5f, int max_iter = 10000)
        : robot(r)
        , angle_dist(-3.14159265358979323846f, 3.14159265358979323846f)
        , step_size(step)
        , max_iterations(max_iter)
    {
        rng.seed(std::random_device{}());
    }
    
    int nearestNode(const std::vector<Node>& tree, const State3DArm& state) const {
        int best_idx = 0;
        float best_dist = tree[0].state.distanceSquared(state);
        
        for (size_t i = 1; i < tree.size(); ++i) {
            float d = tree[i].state.distanceSquared(state);
            if (d < best_dist) {
                best_dist = d;
                best_idx = static_cast<int>(i);
            }
        }
        return best_idx;
    }
    
    State3DArm randomState(int dof) {
        State3DArm state(dof);
        for (int i = 0; i < dof; ++i) {
            state.joint_angles[static_cast<size_t>(i)] = angle_dist(rng);
        }
        return state;
    }
    
    State3DArm steer(const State3DArm& from, const State3DArm& to) const {
        float dist = from.distance(to);
        if (dist <= step_size) return to;
        
        float t = step_size / dist;
        return from.interpolate(to, t);
    }
    
    int extend(std::vector<Node>& tree, const State3DArm& target) {
        int nearest_idx = nearestNode(tree, target);
        State3DArm new_state = steer(tree[static_cast<size_t>(nearest_idx)].state, target);
        
        if (!robot.isValid(new_state)) return -1;
        if (!robot.checkMotion(tree[static_cast<size_t>(nearest_idx)].state, new_state)) return -1;
        
        tree.emplace_back(new_state, nearest_idx);
        return static_cast<int>(tree.size() - 1);
    }
    
    int connect(std::vector<Node>& tree, const State3DArm& target) {
        int idx = -1;
        while (true) {
            int new_idx = extend(tree, target);
            if (new_idx == -1) return idx;
            
            idx = new_idx;
            if (tree[static_cast<size_t>(idx)].state.distance(target) < 1e-3f) return idx;
        }
    }
    
    std::vector<State3DArm> extractPath(int idx_start, int idx_goal) const {
        std::vector<State3DArm> path_start, path_goal;
        
        // Extract from start tree
        int idx = idx_start;
        while (idx != -1) {
            path_start.push_back(tree_start[static_cast<size_t>(idx)].state);
            idx = tree_start[static_cast<size_t>(idx)].parent_idx;
        }
        std::reverse(path_start.begin(), path_start.end());
        
        // Extract from goal tree
        idx = idx_goal;
        while (idx != -1) {
            path_goal.push_back(tree_goal[static_cast<size_t>(idx)].state);
            idx = tree_goal[static_cast<size_t>(idx)].parent_idx;
        }
        
        // Combine
        path_start.insert(path_start.end(), path_goal.begin(), path_goal.end());
        return path_start;
    }
    
    std::vector<State3DArm> plan(const State3DArm& start, const State3DArm& goal) {
        tree_start.clear();
        tree_goal.clear();
        
        if (!robot.isValid(start) || !robot.isValid(goal)) {
            std::cerr << "Start or goal invalid!\n";
            return {};
        }
        
        tree_start.emplace_back(start);
        tree_goal.emplace_back(goal);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Extend start tree toward random
            State3DArm rand = randomState(static_cast<int>(start.joint_angles.size()));
            int idx_start = extend(tree_start, rand);
            
            if (idx_start != -1) {
                // Try to connect goal tree to new node
                int idx_goal = connect(tree_goal, tree_start[static_cast<size_t>(idx_start)].state);
                
                if (idx_goal != -1 && 
                    tree_start[static_cast<size_t>(idx_start)].state.distance(tree_goal[static_cast<size_t>(idx_goal)].state) < 1e-3f) {
                    std::cout << "Solution found in " << iter << " iterations!\n";
                    return extractPath(idx_start, idx_goal);
                }
            }
            
            // Swap trees
            std::swap(tree_start, tree_goal);
        }
        
        std::cerr << "No solution found\n";
        return {};
    }
};
// ==================================================================================

// Create dense environment with many obstacles (fix from 38.md)
void createDenseEnvironment(Environment& env) {
    // Floor
    env.addBox(-10.0f, -10.0f, -1.0f, 10.0f, 10.0f, 0.0f);
    
    // Create a cluttered workspace
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> pos(-3.0f, 3.0f);
    std::uniform_real_distribution<float> size(0.3f, 0.8f);
    
    // Add 50 random obstacles
    for (int i = 0; i < 50; ++i) {
        float x = pos(rng);
        float y = pos(rng);
        float z = pos(rng) + 1.0f; // Above floor
        float s = size(rng);
        env.addBox(x, y, z, x+s, y+s, z+s);
    }
    
    // Add sphere obstacles
    for (int i = 0; i < 20; ++i) {
        env.addSphere(pos(rng), pos(rng), pos(rng)+1.5f, size(rng)*0.5f);
    }
    
    env.padToSIMDWidth();
}

// Create moderate environment for planning tests (fix from 39.md)
void createModerateEnvironment(Environment& env) {
    // Floor
    env.addBox(-10.0f, -10.0f, -1.0f, 10.0f, 10.0f, 0.0f);
    
    // Create a moderate workspace (fewer obstacles than dense)
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> pos(-2.0f, 2.0f);
    std::uniform_real_distribution<float> size(0.4f, 0.7f);
    
    // Add only 10 random obstacles (much less than dense)
    for (int i = 0; i < 10; ++i) {
        float x = pos(rng);
        float y = pos(rng);
        float z = pos(rng) + 1.0f; // Above floor
        float s = size(rng);
        env.addBox(x, y, z, x+s, y+s, z+s);
    }
    
    // Add only 5 sphere obstacles
    for (int i = 0; i < 5; ++i) {
        env.addSphere(pos(rng), pos(rng), pos(rng)+1.5f, size(rng)*0.5f);
    }
    
    env.padToSIMDWidth();
}

// Helper function to sample a valid state (improved from 38.md)
State3DArm sampleValidState(const Robot3DArm_VAMP& robot, std::mt19937& rng, 
                           int dof, int max_attempts = 10000) {
    // Use smaller angle range for better sampling (fix from 38.md)
    std::uniform_real_distribution<float> angle_dist(-1.57f, 1.57f); // ±90°
    State3DArm state(dof);
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        for (int i = 0; i < dof; ++i) {
            state.joint_angles[static_cast<size_t>(i)] = angle_dist(rng);
        }
        if (robot.isValid(state)) {
            return state;
        }
    }
    
    // If we can't find a valid state, throw error (fix from 38.md)
    throw std::runtime_error("Cannot find valid state");
}

// Sample valid start/goal states that are far apart (fix from 38.md)
State3DArm sampleValidStartGoal(const Robot3DArm_VAMP& robot, 
                                std::mt19937& rng, int dof) {
    // Sample from workspace regions that are likely valid but non-trivial
    std::uniform_real_distribution<float> dist_small(-1.57f, 1.57f); // ±90°
    State3DArm state(dof);
    
    int attempts = 0;
    while (attempts++ < 10000) {
        for (int i = 0; i < dof; ++i) {
            state.joint_angles[static_cast<size_t>(i)] = dist_small(rng);
        }
        if (robot.isValid(state)) return state;
    }
    throw std::runtime_error("Cannot find valid state");
}

// ============================ SEQUENTIAL COLLISION CHECKER ============================
// Scalar baseline for comparison (fix from 38.md)
class SequentialCollisionChecker {
    const RobotModel& model;
    const Environment& env;
    
public:
    SequentialCollisionChecker(const RobotModel& m, const Environment& e) 
        : model(m), env(e) {}
    
    bool checkCollision(const State3DArm& state) const {
        // Scalar FK + collision checking
        std::vector<Transform3D> transforms(model.links.size());
        transforms[0].setIdentity();
        
        // Forward kinematics
        for (size_t i = 0; i < model.links.size(); ++i) {
            if (i < model.kinematics.dh_params.size()) {
                Transform3D local = dhTransform(
                    model.kinematics.dh_params[i], 
                    state.joint_angles[i]
                );
                transforms[i] = (i > 0) ? 
                    transforms[i-1].compose(local) : local;
            }
            
            // Check hierarchical spheres
            const auto& link = model.links[i];
            
            // Level 0 (coarse)
            if (checkSphereCollision(link.spheres_level0[0], transforms[i])) {
                // Level 1 (medium)
                bool medium_hit = false;
                for (const auto& sph : link.spheres_level1) {
                    if (checkSphereCollision(sph, transforms[i])) {
                        medium_hit = true;
                        break;
                    }
                }
                if (!medium_hit) continue;
                
                // Level 2 (fine)
                for (const auto& sph : link.spheres_level2) {
                    if (checkSphereCollision(sph, transforms[i])) {
                        return true; // Collision!
                    }
                }
            }
        }
        return false;
    }
    
private:
    bool checkSphereCollision(const Sphere3D& sphere, 
                             const Transform3D& T) const {
        float wx, wy, wz;
        T.transformPoint(sphere.x, sphere.y, sphere.z, wx, wy, wz);
        
        // Check against boxes
        for (size_t i = 0; i < env.actualNumBoxes(); ++i) {
            float dx = std::max({env.box_min_x[i] - wx, 0.0f, 
                                wx - env.box_max_x[i]});
            float dy = std::max({env.box_min_y[i] - wy, 0.0f, 
                                wy - env.box_max_y[i]});
            float dz = std::max({env.box_min_z[i] - wz, 0.0f, 
                                wz - env.box_max_z[i]});
            if (dx*dx + dy*dy + dz*dz <= sphere.r * sphere.r) 
                return true;
        }
        
        // Check against spheres
        for (size_t i = 0; i < env.actualNumSpheres(); ++i) {
            float dx = env.sphere_x[i] - wx;
            float dy = env.sphere_y[i] - wy;
            float dz = env.sphere_z[i] - wz;
            float r_sum = sphere.r + env.sphere_r[i];
            if (dx*dx + dy*dy + dz*dz <= r_sum * r_sum) 
                return true;
        }
        
        return false;
    }
};

// ================================ BENCHMARK ======================================
void benchmark() {
    std::cout << "\n=== VAMP-Aligned Implementation Benchmark ===\n\n";
    
    // Create dense environment with many obstacles (fix from 38.md)
    Environment env;
    createDenseEnvironment(env);
    
    std::cout << "Environment: " << env.actualNumBoxes() << " boxes, " 
              << env.actualNumSpheres() << " spheres\n";
    
    // Create 6-DOF robot
    // Create robot with 3D SE(3) transforms (use true for 3D, false for 2D planar)
    Robot3DArm_VAMP robot(env, true);  // true = use 3D with DH parameters
    
    std::cout << "Robot: 6-DOF arm with hierarchical spheres\n";
    std::cout << "  Links: " << robot.model.links.size() << "\n";
    std::cout << "  Spheres per link: L0=" << robot.model.links[0].spheres_level0.size()
              << ", L1=" << robot.model.links[0].spheres_level1.size()
              << ", L2=" << robot.model.links[0].spheres_level2.size() << "\n\n";
    
#ifndef NDEBUG
    // Parity test: compare vectorized vs scalar implementations
    std::cout << "--- Parity Test (Vectorized vs Scalar) ---\n";
    std::mt19937 rng_parity(12345);
    std::uniform_real_distribution<float> dist_parity(-3.14159265358979323846f, 3.14159265358979323846f);
    
    std::array<State3DArm, SIMD_WIDTH> sample;
    for (size_t i = 0; i < SIMD_WIDTH; ++i) {
        sample[i] = State3DArm(6);
        for (auto& a : sample[i].joint_angles) {
            a = dist_parity(rng_parity);
        }
    }
    
    VectorizedConfig3DArm vcfg(6);
    vcfg.loadFromAOS(sample);
    
    uint8_t vx = robot.fk_cc.computeFK_Interleaved_6DOF(vcfg);
    uint8_t gx = robot.fk_cc.computeFK_Interleaved_Generic(vcfg);
    
    if (vx != gx) {
        std::cerr << "ERROR: Parity mismatch! Vectorized=" << int(vx) 
                  << " Generic=" << int(gx) << "\n";
        std::cerr << "  Binary masks: Vectorized=" << bits8(vx) 
                  << " Generic=" << bits8(gx) << "\n";
        // Print mismatching lanes when parity fails
        for (int lane = 0; lane < 8; ++lane) {
            bool v = (vx >> lane) & 1;
            bool g = (gx >> lane) & 1;
            if (v != g) {
                std::cerr << "  lane " << lane << " differs: vec=" << v 
                          << " gen=" << g << "\n";
            }
        }
        std::cerr << "  This indicates a correctness bug in the vectorized implementation.\n";
        
        // === DIAGNOSTIC: dump per-lane link centers and coarse masks ===
        std::cout << "Detailed diagnostic (per-link, per-lane):\n";
        for (int link_idx = 0; link_idx < 6; ++link_idx) {
            // --- Generic per-lane centers for this link ---
            std::cout << "Link " << link_idx << " (Generic centers):\n";
            alignas(32) float joint_angles[8];
            alignas(32) float px[8] = {0}, py[8] = {0};
            alignas(32) float cumulative_angle[8] = {0};
            for (int l = 0; l <= link_idx; ++l) {
                _mm256_store_ps(joint_angles, vcfg.getJoint(l));
                for (int lane = 0; lane < 8; ++lane) {
                    cumulative_angle[lane] += joint_angles[lane];
                    float c = cosf(cumulative_angle[lane]);
                    float s = sinf(cumulative_angle[lane]);
                    float newx = px[lane] + c * robot.model.links[static_cast<size_t>(l)].length;
                    float newy = py[lane] + s * robot.model.links[static_cast<size_t>(l)].length;
                    px[lane] = newx; py[lane] = newy;
                }
            }
            // Compute previous link endpoints for coarse sphere center
            alignas(32) float prev_px[8] = {0}, prev_py[8] = {0};
            if (link_idx > 0) {
                alignas(32) float cum2[8] = {0};
                for (int l = 0; l < link_idx; ++l) {
                    _mm256_store_ps(joint_angles, vcfg.getJoint(l));
                    for (int lane = 0; lane < 8; ++lane) {
                        cum2[lane] += joint_angles[lane];
                        float c = cosf(cum2[lane]), s = sinf(cum2[lane]);
                        prev_px[lane] += c * robot.model.links[static_cast<size_t>(l)].length;
                        prev_py[lane] += s * robot.model.links[static_cast<size_t>(l)].length;
                    }
                }
            }
            for (int lane = 0; lane < 8; ++lane) {
                float cx = (prev_px[lane] + px[lane]) * 0.5f;
                float cy = (prev_py[lane] + py[lane]) * 0.5f;
                float cz = 0.0f;
                std::cout << "  lane " << lane << ": (" << cx << "," << cy << "," << cz << ")\n";
            }
        }
        std::cout << "End diagnostic\n";
    } else {
        std::cout << "  Parity check PASSED: Vectorized and Generic implementations match.\n";
    }
    std::cout << "\n";
#endif
    
    // Benchmark single state validation
    std::cout << "--- Single State Validation Benchmark ---\n";
    const int N_STATES = 10000;
    std::vector<State3DArm> test_states;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.14159265358979323846f, 3.14159265358979323846f);
    
    for (int i = 0; i < N_STATES; ++i) {
        State3DArm state(6);
        for (auto& a : state.joint_angles) a = dist(rng);
        test_states.push_back(state);
    }
    
    double t_start = nowMs();
    int valid_count = 0;
    for (const auto& state : test_states) {
        if (robot.isValid(state)) valid_count++;
    }
    double t_end = nowMs();
    
    double elapsed = t_end - t_start;
    double per_check = elapsed / N_STATES;
    double throughput = N_STATES / (elapsed / 1000.0);
    
    std::cout << "Validated " << N_STATES << " states in " << elapsed << " ms\n";
    std::cout << "  Time per check: " << per_check << " ms (" << (per_check * 1000.0) << " µs)\n";
    std::cout << "  Throughput: " << throughput << " checks/sec\n";
    std::cout << "  Valid states: " << valid_count << " (" 
              << (100.0 * valid_count / N_STATES) << "%)\n\n";
    
    // Benchmark motion validation
    std::cout << "--- Motion Validation Benchmark ---\n";
    const int N_MOTIONS = 1000;
    std::vector<std::pair<State3DArm, State3DArm>> test_motions;
    
    for (int i = 0; i < N_MOTIONS; ++i) {
        State3DArm s1(6), s2(6);
        for (int j = 0; j < 6; ++j) {
            s1.joint_angles[static_cast<size_t>(j)] = dist(rng);
            s2.joint_angles[static_cast<size_t>(j)] = dist(rng);
        }
        test_motions.emplace_back(s1, s2);
    }
    
    t_start = nowMs();
    int valid_motions = 0;
    for (const auto& [s1, s2] : test_motions) {
        if (robot.checkMotion(s1, s2, 0.1f)) valid_motions++;
    }
    t_end = nowMs();
    
    elapsed = t_end - t_start;
    per_check = elapsed / N_MOTIONS;
    throughput = N_MOTIONS / (elapsed / 1000.0);
    
    std::cout << "Validated " << N_MOTIONS << " motions in " << elapsed << " ms\n";
    std::cout << "  Time per motion: " << per_check << " ms\n";
    std::cout << "  Throughput: " << throughput << " motions/sec\n";
    std::cout << "  Valid motions: " << valid_motions << " (" 
              << (100.0 * valid_motions / N_MOTIONS) << "%)\n\n";
    
    // Planning benchmark - use MODERATE environment (fix from 39.md)
    std::cout << "--- RRT-Connect Planning Benchmark ---\n";
    std::cout << "Creating moderate environment for planning...\n";
    
    Environment env_planning;
    createModerateEnvironment(env_planning);
    
    Robot3DArm_VAMP robot_planning(env_planning, true);
    
    std::cout << "Planning environment: " << env_planning.actualNumBoxes() << " boxes, " 
              << env_planning.actualNumSpheres() << " spheres\n";
    
    // Sample valid start and goal states that are far apart (fix from 38.md)
    std::mt19937 start_rng(12345);   // RNG for start state sampling
    std::mt19937 goal_rng(54321);    // Different RNG for goal state sampling
    std::cout << "Sampling valid start/goal states...\n";
    
    State3DArm start = sampleValidStartGoal(robot_planning, start_rng, 6);
    State3DArm goal = sampleValidStartGoal(robot_planning, goal_rng, 6);
    
    // Ensure they're far apart (fix from 38.md)
    while (start.distance(goal) < 2.0f) {
        goal = sampleValidStartGoal(robot_planning, goal_rng, 6);
    }
    
    // Verify start/goal are valid before planning
    std::cout << "Start state: " << (robot_planning.isValid(start) ? "VALID" : "INVALID") << "\n";
    std::cout << "Goal state: " << (robot_planning.isValid(goal) ? "VALID" : "INVALID") << "\n";
    std::cout << "Distance between start and goal: " << start.distance(goal) << "\n";
    
    if (!robot_planning.isValid(start) || !robot_planning.isValid(goal)) {
        std::cerr << "ERROR: Could not find valid start/goal states after sampling!\n";
        std::cerr << "  This suggests the environment is too dense or robot model is incorrect.\n";
        return;
    }
    
    RRTConnect planner(robot_planning, 0.3f, 50000);
    
    const int N_PLANS = 10;
    std::vector<double> plan_times;
    std::vector<int> path_lengths;
    
    for (int i = 0; i < N_PLANS; ++i) {
        t_start = nowMs();
        auto path = planner.plan(start, goal);
        t_end = nowMs();
        
        if (!path.empty()) {
            plan_times.push_back(t_end - t_start);
            path_lengths.push_back(static_cast<int>(path.size()));
        }
    }
    
    if (!plan_times.empty()) {
        double mean_time = std::accumulate(plan_times.begin(), plan_times.end(), 0.0) / static_cast<double>(plan_times.size());
        double mean_length = std::accumulate(path_lengths.begin(), path_lengths.end(), 0.0) / static_cast<double>(path_lengths.size());
        
        std::sort(plan_times.begin(), plan_times.end());
        double median_time = plan_times[plan_times.size() / 2];
        
        std::cout << "Successful plans: " << plan_times.size() << "/" << N_PLANS << "\n";
        std::cout << "  Mean planning time: " << mean_time << " ms\n";
        std::cout << "  Median planning time: " << median_time << " ms\n";
        std::cout << "  Mean path length: " << mean_length << " waypoints\n";
    } else {
        std::cout << "No successful plans found\n";
    }
    
    std::cout << "\n=== Benchmark Complete ===\n";
}

// ============================ COMPARATIVE BENCHMARK ============================
// Compare sequential vs SIMD performance (fix from 38.md)
void comparativeBenchmark() {
    std::cout << "\n=== Sequential vs SIMD Comparison ===\n\n";
    
    Environment env;
    createDenseEnvironment(env);
    
    Robot3DArm_VAMP robot_simd(env, true);
    SequentialCollisionChecker checker_seq(robot_simd.model, env);
    VectorizedFK_InterleavedCC& fk_cc = robot_simd.fk_cc;
    
    const int N = 100000;
    std::vector<State3DArm> states;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.14159f, 3.14159f);
    
    for (int i = 0; i < N; ++i) {
        State3DArm s(6);
        for (auto& a : s.joint_angles) a = dist(rng);
        states.push_back(s);
    }
    
    // Sequential
    auto t0 = nowMs();
    int seq_collisions = 0;
    for (const auto& s : states) {
        if (checker_seq.checkCollision(s)) seq_collisions++;
    }
    auto t1 = nowMs();
    double seq_time = t1 - t0;
    
    // SIMD - PROPERLY BATCHED (check 8 states at once)
    auto t2 = nowMs();
    int simd_collisions = 0;
    for (size_t i = 0; i < states.size(); i += RAKE_WIDTH) {
        // Batch 8 states together
        std::array<State3DArm, RAKE_WIDTH> batch;
        for (size_t j = 0; j < RAKE_WIDTH && (i + j) < states.size(); ++j) {
            batch[j] = states[i + j];
        }
        // Pad last batch if needed
        for (size_t j = std::min(RAKE_WIDTH, states.size() - i); j < RAKE_WIDTH; ++j) {
            batch[j] = states[i];  // Duplicate last state
        }
        
        VectorizedConfig3DArm vec(6);
        vec.loadFromAOS(batch);
        
        uint8_t collision_mask = fk_cc.computeFK_Interleaved_6DOF(vec);
        
        // Count collisions from mask
        for (size_t j = 0; j < RAKE_WIDTH && (i + j) < states.size(); ++j) {
            if (collision_mask & (1 << j)) {
                simd_collisions++;
            }
        }
    }
    auto t3 = nowMs();
    double simd_time = t3 - t2;
    
    std::cout << "States checked: " << N << "\n";
    std::cout << "Sequential:\n";
    std::cout << "  Time: " << seq_time << " ms\n";
    std::cout << "  Throughput: " << (N / (seq_time/1000.0)) << " checks/sec\n";
    std::cout << "  Collisions: " << seq_collisions << "\n";
    std::cout << "\nSIMD:\n";
    std::cout << "  Time: " << simd_time << " ms\n";
    std::cout << "  Throughput: " << (N / (simd_time/1000.0)) << " checks/sec\n";
    std::cout << "  Collisions: " << simd_collisions << "\n";
    std::cout << "\nSpeedup: " << (seq_time / simd_time) << "x\n";
    
    // Verify correctness
    if (seq_collisions != simd_collisions) {
        std::cerr << "WARNING: Mismatch between sequential and SIMD!\n";
        std::cerr << "  Sequential found " << seq_collisions << " collisions\n";
        std::cerr << "  SIMD found " << simd_collisions << " collisions\n";
    } else {
        std::cout << "Correctness check: PASSED (both methods agree)\n";
    }
}

// ================================ MAIN ==========================================
int main() {
    std::cout << "VAMP-Aligned Motion Planner\n";
    std::cout << "============================\n";
    std::cout << "SIMD Width: " << SIMD_WIDTH << " (" << (SIMD_WIDTH * 4) << " bytes)\n";
    std::cout << "Rake Width: " << RAKE_WIDTH << " configurations\n";
#ifdef USE_SLEEF
    std::cout << "Using SLEEF vector trig\n";
#else
    std::cout << "Using scalar trig fallback\n";
#endif
    std::cout << "\n";
    
    benchmark();
    comparativeBenchmark();
    
    return 0;
}