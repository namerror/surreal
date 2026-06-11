This is a forked repo of SURREAL.

We aim to modify the original codebase to fit our use case:

We are conducting research on training neural networks to follow human walking. We need to augment a synthetic human mesh into the original scene. Therefore we need to generate large amount of synthetic human meshes with walking animation, a large enough sample that we can put in the scene. The single output we need from this forked repo, is the output 3D mesh, with each walking frame as a separate mesh. We will then use these meshes to augment the original scene.

## Proposed steps to achieve this:

This is the simplest and likely best first step.

1. Acquire required assets:

- SURREAL repo.
- SMPL model access/license from the SMPL site.
- SURREAL smpl_data.npz or equivalent SMPL motion data.
- Blender, preferably a modernized script path rather than fighting Blender 2.78 forever.

2. Choose walking MoCap:

- Start with a few CMU/SURREAL walking sequences.
- Generate 30-100 posed meshes per loop.
- Pick 3-10 body shapes and genders for diversity.

3. Export posed meshes:

- For each animation frame, evaluate SMPL pose + shape.
- Export OBJ meshes with shared topology.
- Canonicalize them: pelvis/root at origin, feet near ground, forward axis fixed, consistent units in meters.
- Store metadata: fps, stride_length_m, root_joint_offset, vertical_axis, forward_axis, body_shape_id.

## Notes for future agents

The useful code path is `datageneration/main_part1.py`, not the Torch training code. It already:

- Imports SMPL FBX models in `init_scene`.
- Applies SMPL `trans`, `pose`, and `shape` in `apply_trans_pose_shape`.
- Iterates over animation frames before rendering.
- Uses `ob.to_mesh(scene, True, 'PREVIEW')` elsewhere, which is the likely mechanism for evaluated posed mesh export.

There are two project paths under consideration:

1. Primary verification path: animated mesh export.

- Goal: output one posed OBJ mesh per walking frame.
- This should be the first proof because it avoids RGB compositing, camera calibration, lighting, and Blender scene setup.
- Keep topology stable across frames.
- Export metadata with pose/shape/frame index/root transform/units/axis conventions.
- Required assets: SMPL FBX files and SURREAL `smpl_data.npz` or equivalent SMPL motion data.

2. Adaptation path: RGB-D human rendering.

- Goal: render synthetic human RGB, depth, and mask from a project-specified camera.
- Original SURREAL renders a human over 2D backgrounds; it does not directly know our target scene.
- To use it for RGB-D, inject camera intrinsics/extrinsics, human root position, human orientation, SMPL pose, and shape.
- Disable SURREAL random camera tracking and random root `zrot` when matching our scene.
- Main risk is coordinate/calibration mismatch between SMPL, Blender world, project world, camera coordinates, and depth units.
- Validate with a minimal calibration test: place the body at a known position, render depth/mask, and compare pelvis/feet depth against expected camera-space distance.

Recommended next step: acquire/place the licensed SMPL and SURREAL motion assets, then prove the primary path by exporting 10-30 OBJ frames from one walking sequence before attempting RGB-D rendering.
