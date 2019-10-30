#!/usr/bin/env python3
import os
from tlegenerator.iod import is_iod_observation, decode_iod_observation

if __name__ == "__main__":
    # Open file
    fp = open("temp.dat", errors="replace")
    newlines = fp.readlines()
    fp.close()

    # Check if IOD
    for newline in newlines:
        # Check if this is an IOD observation
        if is_iod_observation(newline):
            # Clean line
            newline = newline.replace("\xa0", " ")
            
            # Decode IOD observation
            o = decode_iod_observation(newline)

            # Generate directory
            if not os.path.exists("observations"):
                os.makedirs("observations")
            
            # Data file name
            fname = os.path.join("observations", "%05d.dat" % o.satno)

            # Read existing observations
            oldlines = []
            if os.path.exists(fname):
                with open(fname, "r") as fp:
                    oldlines = fp.readlines()

            # Append if no duplicate
            if not newline in oldlines:
                oldlines.append(newline)
                
            # Lines to write
            fp = open(fname, "w")
            for line in oldlines:
                fp.write("%s" % line)
            fp.close()
                    
