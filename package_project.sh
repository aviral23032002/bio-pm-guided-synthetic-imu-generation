#!/bin/bash
# package_project.sh — Zip up all results, synthetic tokens, and visualizations.

OUT_ZIP="car_imu_results_package.zip"

echo "===================================================="
echo "CAR-IMU: Packaging Results"
echo "===================================================="

# 1. Collect all evaluation results (.npy)
echo "Collecting .npy results..."
find . -name "*.npy" > files_to_zip.txt

# 2. Collect all plots (.png)
echo "Collecting plots..."
find . -name "*.png" >> files_to_zip.txt

# 3. Collect specific synthetic token stores
# (We exclude real data HDF5s to keep the size manageable if needed,
# but include synthetic ones as requested)
echo "Collecting synthetic HDF5 stores..."
find . -name "synthetic_tokens.hdf5" >> files_to_zip.txt
find . -name "synthetic_waveforms.hdf5" >> files_to_zip.txt

# 4. Include documentation
echo "Adding documentation..."
echo "README.md" >> files_to_zip.txt
echo "EXPERIMENTS.md" >> files_to_zip.txt
echo "colab_evaluation.ipynb" >> files_to_zip.txt

# 5. Create the zip
echo "Creating zip: $OUT_ZIP ..."
zip -@ $OUT_ZIP < files_to_zip.txt

rm files_to_zip.txt
echo "===================================================="
echo "Done! Zip created: $OUT_ZIP"
echo "Total size: $(du -h $OUT_ZIP | cut -f1)"
echo "===================================================="
