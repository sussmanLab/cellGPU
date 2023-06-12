Instructions: copy the installAll.sh files to your home directory (or wherever you'd like), then run it.

This is intended for, e.g., installing the dependencies of cellGPU on a clean environment (e.g., a cluster). On your environment you may already have some of the dependencies installed, so edit the file as needed.

If you used this installer, you may need to compile cellGPU via:
 cmake -DGMP_LIBRARIES=$HOME/.local/lib -DGMP_INCLUDE_DIR=$HOME/.local/include -DMPFR_LIBRARIES=$HOME/.local/lib -DMPFR_INCLUDE_DIR=$HOME/.local/include -DEIGEN3_INCLUDE_DIR=$HOME/.local/include/eigen3 .. 
