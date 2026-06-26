# SURREAL Mesh Export Fork

This repository is a focused fork of
[SURREAL](http://www.di.ens.fr/willow/research/surreal/) for exporting posed
human walking meshes. The intended use is robotics data augmentation: generate
one 3D human mesh per walking animation frame, then place those meshes into a
separate scene or simulator.

The primary workflow is mesh-only. It does not require the original RGB/depth
rendering pipeline, LSUN backgrounds, ffmpeg, OpenEXR, Torch training code, or
pretrained SURREAL models.

## What This Fork Produces

The export path in `datageneration/export_obj_frames.py` loads SURREAL SMPL
motion data and writes:

- one OBJ file per animation frame, for example `frame_000000.obj`
- `metadata.json` describing the source frames, axes, units, topology, and
  texture choices
- optional `material.mtl` and copied clothing texture files

Meshes are exported in meters, pelvis-centered in XY, grounded per frame, with
target axes `+X` forward, `+Y` left, and `+Z` up.

## Essential Setup

Only the assets required for mesh export need to be installed.

### 1. SMPL FBX Models

Download the licensed SMPL for MAYA models from the SMPL website after accepting
their license terms. Place these files under `datageneration/smpl_data/`:

```text
datageneration/smpl_data/
  basicModel_f_lbs_10_207_0_v1.0.2.fbx
  basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

### 2. SURREAL SMPL Motion Data

Download the SURREAL SMPL data after accepting the SURREAL license terms. The
mesh export script needs:

```text
datageneration/smpl_data/
  smpl_data.npz
```

This file contains the CMU MoCap-derived SMPL pose and translation sequences
used by the exporter.

### 3. Optional Clothing Textures

If textured OBJ output is desired, also place the SURREAL texture data under:

```text
datageneration/smpl_data/textures/
```

Use `--no-texture` when exporting geometry only.

### 4. Blender

Install Blender. The helper script uses `BLENDER_PATH` if it is set; otherwise
it falls back to the local path currently encoded in
`datageneration/export_obj_frames.sh`.

Example:

```bash
export BLENDER_PATH=/path/to/blender
```

## Export Meshes

See [datageneration/README.md](datageneration/README.md) for the full export
guide.

Minimal example:

```bash
cd datageneration
./export_obj_frames.sh --idx 02_02 --gender female --frames 30 --stepsize 4 --out ../outputs/obj_frames
```

When choosing a walking sequence, start with
`datageneration/misc/walking_sequences.txt`. It lists the walking sequences that
have been checked for this fork. `datageneration/misc/sequence_idx_map` lists
broader sequence names available in the SURREAL motion data.

## Original SURREAL Project

This fork is based on:

Gul Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black,
Ivan Laptev, and Cordelia Schmid, *Learning from Synthetic Humans*, CVPR 2017.

Original resources:

- [Project page](http://www.di.ens.fr/willow/research/surreal/)
- [Paper](https://arxiv.org/abs/1701.01370)
- [Original repository](https://github.com/gulvarol/surreal)

## Citation

If you use this code or the SURREAL assets, please cite the original work:

```bibtex
@INPROCEEDINGS{varol17_surreal,
  title     = {Learning from Synthetic Humans},
  author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2017}
}
```

## License

Check the repository license in [LICENSE.md](LICENSE.md) and the SURREAL/SMPL
asset license terms before downloading, using, or redistributing code, models,
or data.
