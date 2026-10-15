/*
cycles_bvh_dump.cpp  --  M2 harness: let CYCLES build the BVH, then dump it.

WHAT THIS PROGRAM DOES (and why)
Reads a Cycles standalone XML file (exported from Blender by export_scene_xml.py).
Uses Cycles' built-in XML parser to build a real Cycles `Scene`.
Asks Cycles to "update" the scene on the CPU device. During that update
Cycles' OWN code builds the BVH2 and packs it into flat arrays.
Writes those arrays (raw bytes) plus the camera/BVH parts of KernelData
to an output folder. Later, a Vulkan program reads these files into
GPU buffers and a Slang shader walks them.

WHY XML?
Standalone Cycles ships with a robust XML loader (src/app/cycles_xml.cpp).
By using the exact same XML format that Cycles uses for debugging and
standalone rendering, we reuse Cycles' own parser instead of writing and
maintaining custom OBJ/JSON parsers. Less code of ours = less to get wrong.

USAGE
cycles_bvh_dump <scene.xml> <out_dir>

NOTE ON BUILDING:
Ensure that `src/app/cycles_xml.cpp` is included in your CMake/build system 
when linking against Cycles, as it contains the `xml_read_file` implementation.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Cycles headers
#include "device/device.h"
#include "scene/camera.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "session/session.h"
#include "kernel/types.h" /* KernelData, KernelCamera, KernelBVH */
#include "util/array.h"
#include "util/path.h"
#include "util/progress.h"
#include "util/transform.h"
#include "util/vector.h"

// Cycles built-in XML loader
#include "app/cycles_xml.h"

CCL_NAMESPACE_BEGIN

/* Write one DeviceScene array to <dir>/<name>.bin and log a line to meta. */
template<typename V>
static void dump_array(const std::string &dir, const char *name, V &vec, FILE *meta)
{
    const size_t bytes = vec.memory_size();
    const size_t elem_size = sizeof(*vec.data());
    const size_t count = elem_size ? bytes / elem_size : 0;
    std::printf("  %-16s %10zu elements x %2zu bytes = %10zu bytes\n", name, count, elem_size, bytes);
    std::fprintf(meta, "%s %zu %zu %zu\n", name, bytes, elem_size, count);
    if (bytes == 0) {
        return;
    }
    const std::string path = dir + "/" + name + ".bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    std::fwrite(vec.data(), 1, bytes, f);
    std::fclose(f);
}

static void dump_struct(const std::string &dir, const char *name, const void *p, size_t bytes, FILE *meta)
{
    std::printf("  %-16s struct of %zu bytes\n", name, bytes);
    std::fprintf(meta, "%s %zu %zu 1\n", name, bytes, bytes);
    const std::string path = dir + "/" + name + ".bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    std::fwrite(p, 1, bytes, f);
    std::fclose(f);
}

/* The arrays we dump. This X-macro list is the one place to edit if a name
does not exist in your DeviceScene (see kernel/data_arrays.h for the real
list). */
#define DUMP_LIST(X) \
    X(bvh_nodes) \
    X(bvh_leaf_nodes) \
    X(prim_type) \
    X(prim_visibility) \
    X(prim_index) \
    X(prim_object) \
    X(object_node) \
    X(object_prim_offset) \
    X(objects) \
    X(tri_verts) \
    X(tri_vindex)

static int run(const std::string &xml_path, const std::string &out_dir)
{
    /* ---- 1. Pick the CPU device ----------------------------------------- */
    vector<DeviceInfo> devices = Device::available_devices(DEVICE_MASK_CPU);
    if (devices.empty()) {
        std::fprintf(stderr, "No CPU device found (that should not happen).\n");
        return 1;
    }

    SessionParams session_params;
    session_params.device = devices.front();
    session_params.background = true; /* no viewport / display driver */

    SceneParams scene_params;
    scene_params.bvh_layout = BVH_LAYOUT_BVH2; /* ask for BVH2 explicitly */

    /* A Session owns the Device and the Scene. We never call session->start():
    we do not want an actual render, only the scene upload. */
    Session *session = new Session(session_params, scene_params);
    Scene *scene = &*session->scene;

    /* ---- 2. Load the XML file using Cycles' built-in parser ------------- */
    std::printf("Loading Cycles XML: %s\n", xml_path.c_str());
    
    // xml_read_file parses the XML and populates the Scene with meshes, 
    // objects, camera, lights, and shaders exactly as Cycles expects.
    // It automatically handles the camera scale(1,1,-1) fix baked into the XML.
    ccl::xml_read_file(scene, xml_path.c_str());

    /* ---- 3. Force the scene update (builds the BVH) --------------------- */
    Progress progress;
    /* This is the step that matters: Scene::update() runs every manager's
    device_update(). The GeometryManager builds the BVH (BVH2 on CPU) and
    fills scene->dscene.bvh_nodes etc. */
    scene->update(progress);

    /* ---- 4. Inspect and dump -------------------------------------------- */
    DeviceScene &ds = scene->dscene;
    std::printf("BVH root index: %d, bvh_layout: %d (BVH2 is %d)\n",
                int(ds.data.bvh.root),
                int(ds.data.bvh.bvh_layout),
                int(BVH_LAYOUT_BVH2));

    const std::string meta_path = out_dir + "/meta.txt";
    FILE *meta = std::fopen(meta_path.c_str(), "w");
    if (!meta) {
        std::fprintf(stderr, "cannot create %s (does the folder exist?)\n", meta_path.c_str());
        delete session;
        return 1;
    }

    std::fprintf(meta, "# name total_bytes element_bytes element_count\n");
    std::printf("Writing to %s:\n", out_dir.c_str());

#define X(name) dump_array(out_dir, #name, ds.name, meta);
    DUMP_LIST(X)
#undef X

    /* Small fixed-size structs the kernel reads through kernel_data.*. */
    dump_struct(out_dir, "kernel_bvh", &ds.data.bvh, sizeof(ds.data.bvh), meta);
    dump_struct(out_dir, "kernel_cam", &ds.data.cam, sizeof(ds.data.cam), meta);

    std::fclose(meta);
    delete session;

    std::printf("Done.\n");
    return 0;
}

CCL_NAMESPACE_END

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <scene.xml> <out_dir>\n", argv[0]);
        return 2;
    }

    try {
        /* Tells Cycles where its own data files are (kernels, shaders). */
        ccl::path_init();
        return ccl::run(argv[1], argv[2]);
    }
    catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}