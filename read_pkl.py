import pickle
# Open and read the PKL file
with open('train/models/llc_stage1.vecnormalize.pkl', 'rb') as file:
    data = pickle.load(file)
# Print the loaded data
print(data)