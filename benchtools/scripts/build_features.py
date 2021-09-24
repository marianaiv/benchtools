# Importing the libraries      
import pandas as pd                                                        # Managing DataFrames
from benchtools.substructure import deltaR, tau21                          # Calculation of substructure variables
from benchtools.clustering import build_features                           # Functions for the clustering
from benchtools.datatools import generator, ascii_column, save_df          # Tools for data management
from tqdm import tqdm                                                      # Progress bar
from optparse import OptionParser                                          # Managing the options to run the script

# Options for running the script
parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--RD", action="store_true", default=False, help="Use the RD dataset")
parser.add_option("--BB1", action="store_true", default=False, help="Use the BB1 datase")
parser.add_option("--dir", type="string", default= None, help="Path of the dataset")
parser.add_option("--label", type="string", default= None, help="Path of the dataset labels")
parser.add_option("--out", type="string", default="../data/", help=" Path to save the files generated")
parser.add_option("--outname", type="string", default="None", help="Principal name of the generated files")
parser.add_option("--nbatch", type="int", default=3, help="Number of files to generate")

(flags, args) = parser.parse_args()

RD = flags.RD 
BB1 = flags.BB1

# Selecting the data to use according to the option choosed when running the script
if RD:
    path_data = "../../events_anomalydetection.h5"
    path_label = None

elif BB1:
    path_data = "../../events_LHCO2020_BlackBox1.h5"
    path_label = "../../events_LHCO2020_BlackBox1.masterkey"

else: 
    path_data = flags.dir
    path_label = flags.label

# Creating the object to generate the dataframe
fb = generator(path_data,chunksize=512*100)

nbatch = flags.nbatch
outname = "".join((flags.outname,'_{}')).format(batch_idx)
outdir = '../data'

build_features(path_data, nbatch, outname, path_label, outdir, chunksize=512*100)