# Importing the libraries      
import pandas as pd                                                        # Managing DataFrames
from benchtools.substructure import deltaR, tau21                          # Calculation of substructure variables
from benchtools.clustering import jets, features                           # Functions for the clustering
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
    path_datos = "../../events_anomalydetection.h5"

elif BB1:
    path_datos = "../../events_LHCO2020_BlackBox1.h5"
    path_label = "../../events_LHCO2020_BlackBox1.masterkey"

else: 
    path_datos = flags.dir
    path_label = flags.label

# Creating the object to generate the dataframe
fb = generator(path_datos,chunksize=512*100)

# Initiating the counter
batch_idx = 0

for batch in fb:
    # A DataFrame to store the features 
    df = pd.DataFrame(columns=['pT_j1', 'm_j1', 'eta_j1', 'phi_j1', 'E_j1', 'tau_21_j1', 'nhadrons_j1',
                            'pT_j2', 'm_j2', 'eta_j2', 'phi_j2', 'E_j2', 'tau_21_j2', 'nhadrons_j2',
                            'm_jj', 'deltaR_j12','n_hadrons', 'label'])
    
    # Checking the number of batch
    if batch_idx < flags.nbatch:
        print('Part {}/{}'.format(batch_idx+1, flags.nbatch))
    elif batch_idx == flags.nbatch:
        print('Done')
        break
        
    data = batch

    # Getting the label column if we are not using the RD DataFrame
    if RD == False:
        label = ascii_column(path_label)
        data = pd.concat([data, label.iloc[data.index]], axis=1)

    n_events = data.shape[0]

    for event in tqdm(range(n_events)):

        # Getting the data for one event
        data_event = data.iloc[event,:]
        
        # Clustering the jets
        jets_event = jets(data_event, R = 1.0, p = -1, minpt=20)
        
        # Obtaining the features on a DataFrame
        entry = features(data_event, jets_event)
        
        # Adding to the principal DataFrame
        df = df.append(entry, sort=False)
    
    # Saving the DataFrame as csv for every batch  
    outname = "".join((flags.outname,'_{}')).format(batch_idx)
    outdir = '../data'
    save_df(outname, outdir, df)
    
    batch_idx += 1 