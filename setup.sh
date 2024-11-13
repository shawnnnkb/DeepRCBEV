pip install -v -e .
cd packages/DFA3D
bash setup.sh 0
cd ../..
python packages/Voxelization/setup.py develop
python packages/Voxelization/setup_v2.py develop