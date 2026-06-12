"""
Export posed SMPL meshes from SURREAL motion data as OBJ frames.

This is a mesh-only path. It intentionally skips the original SURREAL render
pipeline: no backgrounds, textures, EXR passes, ffmpeg, Python 2, or .mat files.

Run with Blender: blender -b -P export_obj_frames.py -- ...
or execute export_obj_frames.sh
"""

import argparse
import json
import os
import sys
import traceback
from os.path import abspath, exists, join

import bpy
import numpy as np
from mathutils import Matrix, Vector


AXIS_CHOICES = ("x", "-x", "y", "-y", "z", "-z")
AXIS_DIMS = {"x": 0, "y": 1, "z": 2}
AXIS_OPTIONS = ("--forward-axis", "--left-axis", "--up-axis")


PART_MATCH = {
    "root": "root",
    "bone_00": "Pelvis",
    "bone_01": "L_Hip",
    "bone_02": "R_Hip",
    "bone_03": "Spine1",
    "bone_04": "L_Knee",
    "bone_05": "R_Knee",
    "bone_06": "Spine2",
    "bone_07": "L_Ankle",
    "bone_08": "R_Ankle",
    "bone_09": "Spine3",
    "bone_10": "L_Foot",
    "bone_11": "R_Foot",
    "bone_12": "Neck",
    "bone_13": "L_Collar",
    "bone_14": "R_Collar",
    "bone_15": "Head",
    "bone_16": "L_Shoulder",
    "bone_17": "R_Shoulder",
    "bone_18": "L_Elbow",
    "bone_19": "R_Elbow",
    "bone_20": "L_Wrist",
    "bone_21": "R_Wrist",
    "bone_22": "L_Hand",
    "bone_23": "R_Hand",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    argv = normalize_axis_args(argv)
    parser = argparse.ArgumentParser(description="Export SURREAL SMPL OBJ frames.")
    parser.add_argument("--idx", type=str, default="02_02", help="SURREAL/CMU sequence index.")
    parser.add_argument("--gender", choices=("female", "male"), default="female")
    parser.add_argument("--start", type=int, default=0, help="Original motion start frame.")
    parser.add_argument("--frames", type=int, default=30, help="Number of OBJ frames to export.")
    parser.add_argument("--stepsize", type=int, default=4, help="Source motion frame stride.")
    parser.add_argument("--out", default="../outputs/obj_frames", help="Output root directory.")
    parser.add_argument("--smpl-data-folder", default="smpl_data")
    parser.add_argument("--smpl-data-filename", default="smpl_data.npz")
    parser.add_argument("--shape", choices=("average",), default="average")
    parser.add_argument(
        "--zrot",
        type=float,
        default=0.0,
        help="Post-canonical yaw around target +Z in radians.",
    )
    parser.add_argument(
        "--forward-axis",
        choices=AXIS_CHOICES,
        default="x",
        help="Raw evaluated mesh axis to use as canonical +X forward.",
    )
    parser.add_argument(
        "--left-axis",
        choices=AXIS_CHOICES,
        default="z",
        help="Raw evaluated mesh axis to use as canonical +Y left.",
    )
    parser.add_argument(
        "--up-axis",
        choices=AXIS_CHOICES,
        default="-y",
        help="Raw evaluated mesh axis to use as canonical +Z up.",
    )
    return parser.parse_args(argv)


def normalize_axis_args(argv):
    normalized = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if (
            arg in AXIS_OPTIONS
            and index + 1 < len(argv)
            and argv[index + 1] in AXIS_CHOICES
        ):
            normalized.append("%s=%s" % (arg, argv[index + 1]))
            index += 2
            continue
        normalized.append(arg)
        index += 1
    return normalized


def axis_dim(axis):
    return AXIS_DIMS[axis[-1]]


def axis_vector(axis):
    sign = -1.0 if axis.startswith("-") else 1.0
    vec = np.zeros(3)
    vec[axis_dim(axis)] = sign
    return vec


def build_axis_transform(forward_axis, left_axis, up_axis):
    axes = (forward_axis, left_axis, up_axis)
    dims = [axis_dim(axis) for axis in axes]
    if len(set(dims)) != 3:
        raise ValueError(
            "Coordinate axes must use distinct raw dimensions; got "
            "--forward-axis %s, --left-axis %s, --up-axis %s."
            % (forward_axis, left_axis, up_axis)
        )

    matrix = np.vstack([axis_vector(axis) for axis in axes])
    det = np.linalg.det(matrix)
    if det <= 0:
        raise ValueError(
            "Coordinate axes must form a right-handed basis; got "
            "--forward-axis %s, --left-axis %s, --up-axis %s."
            % (forward_axis, left_axis, up_axis)
        )
    return matrix


def yaw_matrix(zrot):
    cos_z = np.cos(zrot)
    sin_z = np.sin(zrot)
    return np.asarray(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]]
    )


def rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray(
        [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]
    )
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


def rodrigues_to_blendshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [rodrigues(rod_rot) for rod_rot in rod_rots]
    blendshapes = np.concatenate(
        [(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]]
    )
    return mat_rots, blendshapes


def set_active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def find_imported_objects(obname):
    mesh_ob = bpy.data.objects.get(obname)
    arm_ob = bpy.data.objects.get("Armature")

    if mesh_ob is None:
        candidates = [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and obj.data.shape_keys is not None
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not identify imported SMPL mesh object '%s'." % obname
            )
        mesh_ob = candidates[0]

    if arm_ob is None:
        candidates = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        if len(candidates) != 1:
            raise RuntimeError("Could not identify imported SMPL armature.")
        arm_ob = candidates[0]

    return mesh_ob, arm_ob


def init_scene(smpl_data_folder, gender):
    clear_scene()
    gender_prefix = gender[0]
    obname = "%s_avg" % gender_prefix
    fbx_path = join(
        smpl_data_folder, "basicModel_%s_lbs_10_207_0_v1.0.2.fbx" % gender_prefix
    )
    if not exists(fbx_path):
        raise FileNotFoundError("Missing SMPL FBX: %s" % fbx_path)

    bpy.ops.import_scene.fbx(
        filepath=fbx_path, axis_forward="Y", axis_up="Z", global_scale=100
    )
    ob, arm_ob = find_imported_objects(obname)
    ob.data.use_auto_smooth = False

    if ob.data.shape_keys is None:
        raise RuntimeError("Imported SMPL mesh has no shape keys.")

    ob.data.shape_keys.animation_data_clear()
    arm_ob.animation_data_clear()

    for key in ob.data.shape_keys.key_blocks:
        key.slider_min = -10
        key.slider_max = 10

    return ob, obname, arm_ob, abspath(fbx_path)


def load_body_data(smpl_data, gender, idx):
    sequence_names = sorted(
        key.replace("pose_", "") for key in smpl_data.files if key.startswith("pose_")
    )
    if not sequence_names:
        raise RuntimeError("No pose_* sequences found in SMPL data.")

    sequence_name = idx.strip()
    poses = smpl_data["pose_" + sequence_name]
    trans = smpl_data["trans_" + sequence_name]
    return sequence_names, sequence_name, poses, trans


def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname):
    mat_rots, pose_blendshapes = rodrigues_to_blendshapes(pose)

    arm_ob.pose.bones[obname + "_Pelvis"].location = trans

    for ibone, mat_rot in enumerate(mat_rots):
        bone_name = obname + "_" + PART_MATCH["bone_%02d" % ibone]
        arm_ob.pose.bones[bone_name].rotation_quaternion = Matrix(mat_rot).to_quaternion()

    for ibshape, value in enumerate(pose_blendshapes):
        ob.data.shape_keys.key_blocks["Pose%03d" % ibshape].value = value

    for ibshape, value in enumerate(shape):
        ob.data.shape_keys.key_blocks["Shape%03d" % ibshape].value = value


def evaluated_mesh_for_object(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.context.view_layer.update()
    evaluated = ob.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=False, depsgraph=depsgraph
    )
    return mesh, evaluated.matrix_world.copy()


def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, reg_ivs, joint_reg):
    apply_trans_pose_shape(Vector(orig_trans), np.zeros(72), shape, ob, arm_ob, obname)
    mesh, _ = evaluated_mesh_for_object(ob)

    reg_vs = np.empty((len(reg_ivs), 3))
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = mesh.vertices[int(iv)].co
    bpy.data.meshes.remove(mesh)

    joint_xyz = joint_reg.dot(reg_vs)

    set_active(arm_ob)
    arm_ob.hide_set(False)
    bpy.ops.object.mode_set(mode="EDIT")
    for ibone in range(24):
        bone_name = obname + "_" + PART_MATCH["bone_%02d" % ibone]
        edit_bone = arm_ob.data.edit_bones[bone_name]
        offset = edit_bone.tail - edit_bone.head
        edit_bone.head = joint_xyz[ibone]
        edit_bone.tail = edit_bone.head + offset
    bpy.ops.object.mode_set(mode="OBJECT")


def vector_to_array(vec):
    return np.asarray((vec.x, vec.y, vec.z), dtype=np.float64)


def pelvis_world_position(arm_ob, obname):
    pelvis = arm_ob.pose.bones[obname + "_Pelvis"].head.copy()
    return vector_to_array(arm_ob.matrix_world @ pelvis)


def write_obj(filepath, vertices, normals, mesh):
    with open(filepath, "w") as f:
        f.write("# Canonical pelvis-XY-centered, feet-grounded SMPL OBJ\n")
        f.write("o smpl_canonical\n")
        for vertex in vertices:
            f.write("v %.9f %.9f %.9f\n" % tuple(vertex))
        for normal in normals:
            f.write("vn %.9f %.9f %.9f\n" % tuple(normal))
        for polygon in mesh.polygons:
            face = []
            for vertex_index, loop_index in zip(polygon.vertices, polygon.loop_indices):
                face.append("%d//%d" % (vertex_index + 1, loop_index + 1))
            f.write("f %s\n" % " ".join(face))


def export_evaluated_obj(ob, filepath, pelvis_world, source_to_target):
    mesh, matrix_world = evaluated_mesh_for_object(ob)
    try:
        normal_matrix = matrix_world.to_3x3().inverted().transposed()
        mesh.calc_normals_split()

        pelvis_target = source_to_target.dot(pelvis_world)
        vertices = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        for index, vertex in enumerate(mesh.vertices):
            world = vector_to_array(matrix_world @ vertex.co)
            vertices[index] = source_to_target.dot(world)

        vertices[:, 0] -= pelvis_target[0]
        vertices[:, 1] -= pelvis_target[1]
        ground_height = float(vertices[:, 2].min())
        vertices[:, 2] -= ground_height

        normals = np.empty((len(mesh.loops), 3), dtype=np.float64)
        for index, loop in enumerate(mesh.loops):
            world_normal = vector_to_array(normal_matrix @ loop.normal)
            normal = source_to_target.dot(world_normal)
            length = np.linalg.norm(normal)
            normals[index] = normal / length if length > 0.0 else normal

        write_obj(filepath, vertices, normals, mesh)

        vertex_count = len(mesh.vertices)
        face_count = len(mesh.polygons)
        return vertex_count, face_count, ground_height
    finally:
        bpy.data.meshes.remove(mesh)


def validate_args(args, poses):
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.frames <= 0:
        raise ValueError("--frames must be positive.")
    if args.stepsize <= 0:
        raise ValueError("--stepsize must be positive.")

    last_frame = args.start + (args.frames - 1) * args.stepsize
    if last_frame >= len(poses):
        raise ValueError(
            "Requested frames exceed sequence length: last source frame %d, "
            "sequence has %d frames." % (last_frame, len(poses))
        )
    build_axis_transform(args.forward_axis, args.left_axis, args.up_axis)


def main():
    args = parse_args()
    smpl_data_folder = abspath(args.smpl_data_folder)
    smpl_data_path = join(smpl_data_folder, args.smpl_data_filename)
    if not exists(smpl_data_path):
        raise FileNotFoundError("Missing SMPL data file: %s" % smpl_data_path)

    print("Loading SMPL data: %s" % smpl_data_path)
    smpl_data = np.load(smpl_data_path)
    sequence_names, sequence_name, poses, trans = load_body_data(smpl_data, args.gender, args.idx)
    validate_args(args, poses)
    axis_transform = build_axis_transform(
        args.forward_axis, args.left_axis, args.up_axis
    )
    source_to_target = yaw_matrix(args.zrot).dot(axis_transform)

    shape = np.zeros(10)
    print("Initializing SMPL %s model" % args.gender)
    ob, obname, arm_ob, fbx_path = init_scene(smpl_data_folder, args.gender)

    orig_trans = np.asarray(arm_ob.pose.bones[obname + "_Pelvis"].location).copy()
    reset_joint_positions(
        orig_trans,
        shape,
        ob,
        arm_ob,
        obname,
        smpl_data["regression_verts"],
        smpl_data["joint_regressor"],
    )

    sequence_dir = "%s_%s_avg_pelvis_centered_axes" % (
        sequence_name.replace(" ", ""),
        args.gender,
    )
    output_dir = abspath(join(args.out, sequence_dir))
    os.makedirs(output_dir, exist_ok=True)

    source_frame_indices = [
        args.start + export_frame * args.stepsize for export_frame in range(args.frames)
    ]
    exported_files = []
    pelvis_positions = []
    ground_heights = []
    vertex_count = None
    face_count = None

    print("Exporting %d OBJ frames to %s" % (args.frames, output_dir))
    for export_frame, source_frame in enumerate(source_frame_indices):
        apply_trans_pose_shape(
            Vector(trans[source_frame]), poses[source_frame], shape, ob, arm_ob, obname
        )
        bpy.context.view_layer.update()

        filename = "frame_%06d.obj" % export_frame
        filepath = join(output_dir, filename)
        pelvis_world = pelvis_world_position(arm_ob, obname)
        curr_vertex_count, curr_face_count, ground_height = export_evaluated_obj(
            ob, filepath, pelvis_world, source_to_target
        )
        if vertex_count is None:
            vertex_count = curr_vertex_count
            face_count = curr_face_count
        elif vertex_count != curr_vertex_count or face_count != curr_face_count:
            raise RuntimeError("Vertex/face count changed while exporting %s." % filename)
        exported_files.append(filename)
        pelvis_positions.append(pelvis_world.tolist())
        ground_heights.append(ground_height)
        print("Exported %s from source frame %d" % (filename, source_frame))

    metadata = {
        "sequence_name": sequence_name,
        "gender": args.gender,
        "shape_mode": args.shape,
        "shape_coefficients": shape.tolist(),
        "source_frame_indices": source_frame_indices,
        "start": args.start,
        "frames": args.frames,
        "stepsize": args.stepsize,
        "coordinate_mode": "pelvis_centered_axes",
        "origin_mode": "pelvis_xy_mesh_floor_z_per_frame",
        "ground_source": "mesh_min_z",
        "ground_heights_before_grounding": ground_heights,
        "units": "meters",
        "axes": {
            "forward": args.forward_axis,
            "left": args.left_axis,
            "up": args.up_axis,
            "target_forward": "+x",
            "target_left": "+y",
            "target_up": "+z",
        },
        "source_to_target_matrix": source_to_target.tolist(),
        "raw_axis_transform_matrix": axis_transform.tolist(),
        "raw_pelvis_positions": pelvis_positions,
        "zrot": args.zrot,
        "blender_version": bpy.app.version_string,
        "smpl_fbx_path": fbx_path,
        "smpl_data_path": abspath(smpl_data_path),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "exported_files": exported_files,
    }

    metadata_path = join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Wrote metadata: %s" % metadata_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        sys.stdout.flush()
        sys.stderr.flush()
        code = exc.code if isinstance(exc.code, int) else 1
        os._exit(code)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
