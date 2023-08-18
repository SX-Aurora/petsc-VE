#!/bin/sh

rm -rf Aurora_MPI_static
rm -f *.err *.res *.log
for fn in `find -name __pycache__`;do rm -rf $fn;done
