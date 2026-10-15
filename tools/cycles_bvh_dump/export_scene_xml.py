"""
export_scene_xml.py  --  Blender scene  ->  Cycles standalone XML
===================================================================

WHAT THIS DOES
    Writes the current Blender scene (mesh objects, the active camera, point
    lights) into the XML format that the standalone Cycles app (cycles.exe)
    reads. That file is the single input for BOTH:
        1. the CPU baseline render (cycles.exe scene.xml), and
        2. the BVH dump harness (see bvh_dump.h), which asks Cycles' own CPU
           code to build the BVH and then copies the result out for Vulkan.

WHY XML AND NOT OBJ + JSON
    Standalone Cycles has no OBJ importer, but it ships an XML loader
    (src/app/cycles_xml.cpp). Writing XML means we reuse Cycles' own loader
    instead of writing a second one in C++. Less code of ours = less to get wrong.

HOW TO RUN (pick one)
    From a terminal (adjust the path to your blender.exe):
        blender.exe -b my_scene.blend -P export_scene_xml.py -- out.xml
    Everything after the lone "--" is ours; Blender ignores it. If you leave
    out the output path, the XML is written next to the .blend file.

    Or open this file in Blender's Text Editor and press "Run Script".

WHAT IT DELIBERATELY DOES NOT DO
    * Materials: every mesh gets one plain grey diffuse shader. Real material
      export is a much later milestone (Principled BSDF is M5-M6).
    * Light units: Blender watts are NOT converted to Cycles units. The image
      brightness will not match Blender. Irrelevant for a geometry/BVH test.
    * Only POINT lights and PERSPECTIVE cameras. Others are skipped with a warning.
    * Camera lens shift is ignored.

COORDINATES
    Blender and Cycles are both Z-up and right-handed, so world-space positions
    need no axis conversion. We bake each object's world matrix into its
    vertices, so the XML meshes need no <transform> wrapper.

Verified against: the scene_caustics.xml example in the Cycles v4.2.0 tag
(camera/transform/light/shader/mesh syntax). I have NOT been able to check it
against the 5.2 loader source, so if cycles.exe complains about a tag, the
error message tells you which one.
"""

import math
import os
import sys

import bpy


def fmt(value):
    """Format a float compactly but without losing precision (9 significant digits
    is enough to round-trip a 32-bit float)."""
    return f"{value:.9g}"


def main():
    # ------------------------------------------------------------------
    # 1. Work out where to write the file.
    # ------------------------------------------------------------------
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if args:
        out_path = args[0]
    else:
        base = bpy.data.filepath or os.path.join(os.getcwd(), "scene.blend")
        out_path = os.path.splitext(base)[0] + ".xml"

    scene = bpy.context.scene

    # ------------------------------------------------------------------
    # 2. Camera.
    #    Cycles' camera looks down its local +Z while Blender's looks down -Z,
    #    which is why the official exporter (and the Cycles examples) append
    #    scale="1 1 -1". We do the same.
    #    Rotation is written as: angle-in-DEGREES, then the axis x y z.
    # ------------------------------------------------------------------
    cam_obj = scene.camera
    if cam_obj is None or cam_obj.type != 'CAMERA':
        raise SystemExit("export_scene_xml: the scene has no active camera.")

    cam = cam_obj.data
    if cam.type != 'PERSP':
        print("WARNING: only perspective cameras are supported; exporting as perspective.")
    if cam.shift_x or cam.shift_y:
        print("WARNING: camera lens shift is ignored.")

    render = scene.render
    pct = render.resolution_percentage / 100.0
    width = max(1, int(render.resolution_x * pct))
    height = max(1, int(render.resolution_y * pct))

    # Cycles' <camera fov=...> is measured across the LONGER image side, in
    # radians. angle_x / angle_y are Blender's horizontal / vertical angles
    # regardless of the sensor-fit setting, so pick the one for the longer side.
    fov = cam.angle_x if width >= height else cam.angle_y

    loc, rot, _scale = cam_obj.matrix_world.decompose()
    axis, angle = rot.to_axis_angle()   # angle is in radians

    lines = ["<cycles>", ""]
    lines += [
        "<!-- Camera -->",
        f'<transform rotate="{fmt(math.degrees(angle))} {fmt(axis.x)} {fmt(axis.y)} {fmt(axis.z)}"',
        f'           translate="{fmt(loc.x)} {fmt(loc.y)} {fmt(loc.z)}"',
        '           scale="1 1 -1">',
        f'  <camera type="perspective" fov="{fmt(fov)}" full_width="{width}" full_height="{height}"/>',
        "</transform>",
        "",
        "<!-- One shared grey diffuse material for every mesh -->",
        '<shader name="grey">',
        '  <diffuse_bsdf name="grey_bsdf" color="0.8 0.8 0.8"/>',
        '  <connect from="grey_bsdf bsdf" to="output surface"/>',
        "</shader>",
        "",
    ]

    # ------------------------------------------------------------------
    # 3. Meshes and lights.
    #    depsgraph.object_instances gives every object that actually appears in
    #    the scene, with modifiers applied and collection instances expanded,
    #    each with its own world matrix.
    # ------------------------------------------------------------------
    depsgraph = bpy.context.evaluated_depsgraph_get()
    n_mesh = n_tri = n_light = 0

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.hide_render:
            continue
        mw = inst.matrix_world

        if obj.type == 'MESH':
            me = obj.to_mesh()              # evaluated mesh (modifiers applied)
            me.calc_loop_triangles()        # Blender's own triangulation
            tris = me.loop_triangles
            if len(tris) == 0:
                obj.to_mesh_clear()
                continue

            # Bake the world matrix into the vertex positions ("P" in the XML).
            P = " ".join(fmt(c) for v in me.vertices for c in (mw @ v.co))

            # A mirrored object (negative determinant) flips triangle winding;
            # swap two indices to keep the faces facing outward.
            flip = mw.determinant() < 0
            idx = []
            for t in tris:
                a, b, c = t.vertices
                if flip:
                    b, c = c, b
                idx += (a, b, c)

            name = f"obj{n_mesh}"
            lines += [
                f"<!-- mesh: {obj.name} ({len(tris)} triangles) -->",
                f'<object name="{name}"/>',
                f'<state interpolation="flat" shader="grey" object="{name}">',
                f'  <mesh P="{P}"',
                # nverts = "how many vertices each polygon has"; all 3 for triangles.
                f'        nverts="{" ".join(["3"] * len(tris))}"',
                f'        verts="{" ".join(map(str, idx))}"/>',
                "</state>",
                "",
            ]
            n_mesh += 1
            n_tri += len(tris)
            obj.to_mesh_clear()

        elif obj.type == 'LIGHT':
            if obj.data.type != 'POINT':
                print(f"WARNING: light '{obj.name}' is {obj.data.type}; only POINT is exported.")
                continue
            # tfm = the top 3 rows of the 4x4 world matrix, row by row, with the
            # translation in the 4th column (this matches the Cycles example files).
            tfm = " ".join(fmt(mw[r][c]) for r in range(3) for c in range(4))
            sname = f"light{n_light}_shader"
            lines += [
                f"<!-- point light: {obj.name} -->",
                f'<shader name="{sname}">',
                f'  <emission name="light{n_light}_emit" color="1.0 1.0 1.0" strength="{fmt(obj.data.energy)}"/>',
                f'  <connect from="light{n_light}_emit emission" to="output surface"/>',
                "</shader>",
                f'<state shader="{sname}">',
                f'  <light name="light{n_light}" light_type="point" strength="1 1 1" tfm="{tfm}"/>',
                "</state>",
                "",
            ]
            n_light += 1

    lines.append("</cycles>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"export_scene_xml: wrote {out_path}")
    print(f"  {n_mesh} mesh object(s), {n_tri} triangle(s), {n_light} point light(s), {width}x{height}")


main()
