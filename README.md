# Introduction 
This codebase can be used to format evaluate different 3D methods (All the baselines from Efficient-SAM2).
We will assume that you have downloaded the Scannet Dataset into a folder called ${ROOT} as scannet/scans/scene\*\*\*\*_\*\*

we will call ${SCANNET} = ${ROOT}/scannet/

## SAM3D

## SAM3D Pro

## SAI3D
first you need to format your scannet data (this overlaps with SAM3D but they have a slightly different folder structure and we didn't uniform them)

TODO: put uv
python helpers/format_scannet.py --scannet_path ${SCANNET}/scans  --output_path ${SCANNET}/posed_images