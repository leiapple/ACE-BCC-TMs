#!/bin/bash

# copy the restart file
cp ../0_relax_initial/restart.screw .
cp ../0_relax_final/restart.screw  restart.screw.moved 

lmp  -in input_restart2data_start >log.initial
echo 'Initial replica generated!'

lmp  -in input_restart2data >log.final
echo 'Final replica generated!'

python data2neb_input.py final.data neb.final

cp initial.data ../2_NEB_calc
cp neb.final ../2_NEB_calc

