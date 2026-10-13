#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>

// ===========================================================================
// GEOMETRY STRUCTURES
// ===========================================================================
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Vec2 {
    float u = 0.0f;
    float v = 0.0f;
};

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 texcoord;
};

struct Triangle {
    Vertex vertices[3];
};

// ===========================================================================
// OBJ PARSER LOGIC
// ===========================================================================
struct IndexTriplet {
    int v = 0;
    int vt = 0;
    int vn = 0;
};

static IndexTriplet parse_face_token(const std::string& token) {
    IndexTriplet triplet{};
    if (token.find("//") != std::string::npos) {
        std::sscanf(token.c_str(), "%d//%d", &triplet.v, &triplet.vn);
    } else {
        std::sscanf(token.c_str(), "%d/%d/%d", &triplet.v, &triplet.vt, &triplet.vn);
    }
    return triplet;
}

bool load_obj(const std::string& filepath, std::vector<Triangle>& out_triangles) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: Could not open file: %s\n", filepath.c_str());
        return false;
    }

    std::vector<Vec3> temp_positions;
    std::vector<Vec3> temp_normals;
    std::vector<Vec2> temp_texcoords;

    struct RawFace {
        std::vector<IndexTriplet> vertices;
    };
    std::vector<RawFace> raw_faces;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Vec3 pos;
            iss >> pos.x >> pos.y >> pos.z;
            temp_positions.push_back(pos);
        } else if (prefix == "vt") {
            Vec2 uv;
            iss >> uv.u >> uv.v;
            temp_texcoords.push_back(uv);
        } else if (prefix == "vn") {
            Vec3 norm;
            iss >> norm.x >> norm.y >> norm.z;
            temp_normals.push_back(norm);
        } else if (prefix == "f") {
            RawFace face;
            std::string token;
            while (iss >> token) {
                face.vertices.push_back(parse_face_token(token));
            }
            if (face.vertices.size() >= 3) {
                raw_faces.push_back(face);
            }
        }
    }

    out_triangles.clear();
    for (const auto& face : raw_faces) {
        const size_t vertex_count = face.vertices.size();
        for (size_t i = 1; i + 1 < vertex_count; ++i) {
            IndexTriplet t_indices[3] = {
                face.vertices[0],
                face.vertices[i],
                face.vertices[i + 1]
            };

            Triangle tri{};
            for (int v_idx = 0; v_idx < 3; ++v_idx) {
                const auto& idx = t_indices[v_idx];
                if (idx.v > 0 && (size_t)(idx.v - 1) < temp_positions.size()) {
                    tri.vertices[v_idx].position = temp_positions[idx.v - 1];
                }
                if (idx.vt > 0 && (size_t)(idx.vt - 1) < temp_texcoords.size()) {
                    tri.vertices[v_idx].texcoord = temp_texcoords[idx.vt - 1];
                }
                if (idx.vn > 0 && (size_t)(idx.vn - 1) < temp_normals.size()) {
                    tri.vertices[v_idx].normal = temp_normals[idx.vn - 1];
                }
            }
            out_triangles.push_back(tri);
        }
    }

    std::printf("Loaded %s -> %zu triangles successfully.\n", filepath.c_str(), out_triangles.size());
    return true;
}

// ===========================================================================
// MAIN TEST DRIVER
// ===========================================================================
int main() {
    std::vector<Triangle> triangles;
    // Adjust path if necessary depending on where the executable runs from
    const std::string asset_path = "c:/dev/cycles-vulkan/assets/cube.obj";

    if (load_obj(asset_path, triangles)) {
        std::cout << "SUCCESS: Parsed mesh with " << triangles.size() << " triangles.\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << "  Vertex " << i << ": (" 
                      << triangles[0].vertices[i].position.x << ", " 
                      << triangles[0].vertices[i].position.y << ", " 
                      << triangles[0].vertices[i].position.z << ")\n";
        }
    } else {
        std::cerr << "FAIL: Could not load object file.\n";
    }

    return 0;
}