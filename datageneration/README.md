# Mesh Export Guide

This directory contains the mesh-only export path for this SURREAL fork. Use it
to turn SMPL walking motion sequences into one OBJ mesh per animation frame.

The exporter skips the original SURREAL RGB/depth pipeline. It does not require
background images, ffmpeg, OpenEXR, Python 2 post-processing, Torch training, or
rendered video output.

## Required Assets

Place the licensed SMPL and SURREAL motion assets here:

```text
datageneration/smpl_data/
  basicModel_f_lbs_10_207_0_v1.0.2.fbx
  basicModel_m_lbs_10_207_0_v1.0.2.fbx
  smpl_data.npz
```

For textured OBJ output, also include the SURREAL clothing textures:

```text
datageneration/smpl_data/textures/
```

If textures are not needed, pass `--no-texture` when exporting.

## Blender Setup

Install Blender and point `BLENDER_PATH` at the Blender executable if the
default path in `export_obj_frames.sh` is not correct for your machine.

```bash
export BLENDER_PATH=/path/to/blender
```

The script runs Blender in background mode and writes logs under
`datageneration/logs/` by default.

## Basic Export

Run from this directory:

```bash
cd datageneration
./export_obj_frames.sh --idx 02_02 --gender female --frames 30 --stepsize 4 --out ../outputs/obj_frames
```

This creates an output directory like:

```text
../outputs/obj_frames/02_02_female_avg_pelvis_centered_axes/
  frame_000000.obj
  frame_000001.obj
  ...
  metadata.json
  material.mtl              # when texture export is enabled
  textures/                 # when texture export is enabled
```

## Choosing a Walking Sequence

Use `misc/walking_sequences.txt` when deciding which walking sequence to export.
The currently checked walking sequences are:

```text
02_01
02_02
```

Pass the sequence name directly to `--idx`:

```bash
./export_obj_frames.sh --idx 02_01 --gender female --frames 30 --out ../outputs/obj_frames
```

For a broader list of sequence names present in the SURREAL motion data, see
`misc/sequence_idx_map`.

## Common Options

`--idx`
: SURREAL/CMU sequence name, such as `02_01` or `02_02`.

`--gender`
: SMPL model gender. Use `female` or `male`.

`--start`
: Source motion frame to start from. Defaults to `0`.

`--frames`
: Number of OBJ frames to export. Defaults to `30`.

`--stepsize`
: Source motion frame stride between exported frames. Defaults to `4`.

`--out`
: Output root directory. Defaults to `../outputs/obj_frames`.

`--no-texture`
: Export geometry and UVs without writing `material.mtl` or copying texture
  files.

`--texture-index`
: Deterministic index into the filtered texture list.

`--texture-seed`
: Seed used for deterministic texture selection when `--texture-index` is not
  set. Defaults to `0`.

`--clothing-option`
: Texture filter. Use `all`, `grey`, or `nongrey`.

`--forward-axis`, `--left-axis`, `--up-axis`
: Raw evaluated mesh axes mapped into the target coordinate system. Defaults are
  `--forward-axis x`, `--left-axis z`, and `--up-axis -y`, producing target axes
  `+X` forward, `+Y` left, and `+Z` up.

`--zrot`
: Additional yaw rotation around target `+Z`, in radians. Defaults to `0.0`.

## Output Conventions

Each exported OBJ is:

- pelvis-centered in XY
- grounded per frame using the mesh minimum Z
- written in meters
- exported with stable topology across frames
- oriented with target `+X` forward, `+Y` left, and `+Z` up

`metadata.json` records the source sequence, source frame indices, axes, units,
Blender version, SMPL asset paths, topology counts, texture choices, and exported
file list.

## Geometry-Only Example

For downstream robotics pipelines that only need mesh geometry:

```bash
cd datageneration
./export_obj_frames.sh --idx 02_02 --gender female --frames 30 --stepsize 4 --no-texture --out ../outputs/obj_frames
```

## Troubleshooting

If Blender cannot be found, set `BLENDER_PATH`.

If `smpl_data.npz` or the SMPL FBX files are missing, verify the files are under
`datageneration/smpl_data/`.

If texture export fails, either install the texture folder under
`datageneration/smpl_data/textures/` or use `--no-texture`.

If an export requests frames beyond the sequence length, reduce `--frames`, use a
smaller `--start`, or reduce `--stepsize`.
