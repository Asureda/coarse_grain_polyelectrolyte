#!/bin/bash
#!/usr/bin/env python
#Getting the date and time to create the
tmp_folder=$(date +'%Y_%m_%d_%H_%M_%S')
#echo $(date)
#echo $tmp_folder

cores=3
# Creating the temporal file in the PROGRAM dir.

mkdir OUTPUT/

# Copy the INPUT filles to tmp. folder

cp INPUT/* .


#Copy the program scripts in the tmp. folder

#cp *.py $tmp_folder/
#cp *.bin $tmp_folder/
#cp r_* $tmp_folder/



# Running the programs in the tmp folder

#cd $tmp_folder/

./pypresso PROGRAM/run_test.py
#mpirun -n $cores ./pypresso prova_polymer.py
#nuitka  --nofollow-imports binning.py
#./binning.bin





# After the program ends, create tar, delete folder and move to OUTPUT
pwd
mkdir OUTPUT/$tmp_folder/
mkdir OUTPUT/$tmp_folder/figures/
mkdir OUTPUT/$tmp_folder/data/
mv  *.png OUTPUT/$tmp_folder/figures/
mv *.vtf OUTPUT/$tmp_folder/data/
mv *.xyz OUTPUT/$tmp_folder/data/
mv input.dat OUTPUT/$tmp_folder/

cp PROGRAM/* OUTPUT/$tmp_folder/


#rm -r PROGRAM/$tmp_folder/

echo "Folder for this simulation ${tmp_folder}"
