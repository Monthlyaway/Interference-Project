3-	Time domain Comparative Datasets (Overlap Datasets )
Contain magnitude values of time domain received signals. Here,  four distinct Overlap (OVRLP%) sets are generated, each based on the varying percentage of overlap bandwidth (OVRLP100, OVRLP75, OVRLP50, and OVRLP25),  each with subsets organized as follows,
•	Full_Test_Data_Q: GSO ModCod: “QPSK”,  contains of  4800 x 801 interference and interference-free data (data points x Magnitude + label).
•	Full_Test_Data_8: GSO ModCod: “8PSK”,  contains of  4800 x 801 interference and interference-free data (data points x Magnitude + label).
•	Full_Test_Data_16: GSO ModCod: “16APSK”,  contains of  4800 x 801 interference and interference-free data (data points x Magnitude + label).



To concatenate the subsets:
# Directory containing the datasets
data_dir = '......\Overlap100Data' # Add your data directory

# List of filenames
filenames = ['Full_Test_Set_Q.mat', 'Full_Test_Set_8.mat', 'Full_Test_Set_16.mat']  

# Initialize an empty list to store the datasets
all_data = []

# Loop over the filenames and load each dataset
for filename in filenames:
    file_path = os.path.join(data_dir, filename)
    data = sio.loadmat(file_path)
    all_data.append(data['Full_Test_Data'])

# Combine the datasets
combined_data = np.concatenate(all_data, axis=0)

# Extract the signals and labels
signals = combined_data[:, :-1]  # first 800 columns are signals
signals = (signals - min_val) / (max_val - min_val)
signals = tf.cast(signals, tf.float32)

labels = combined_data[:, -1]    # last column is labels
labels =labels.astype(bool) # 