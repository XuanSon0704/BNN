 # Export weights -> packed u64
 
import torch
import numpy as np
import json
from bnn import BNN_MLP, BinarizedLinear

MODEL_PATH = '../results/model.pt'
OUT_PATH = '../results/bnn_weights.npz'
MEAN_PATH = '../results/feature_means.json'

N_FEATURES = 10
N_HIDDEN = 64

model = BNN_MLP(N_FEATURES, N_HIDDEN)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

def pack_binary_matrix(W):
    Wb = np.sign(W)
    rows, cols = Wb.shape
    packed_cols = (cols + 63) // 64
    packed = np.zeros((rows, packed_cols), dtype=np.uint64)

    for i in range(rows):
        for j in range(cols):
            if Wb[i, j] >= 0:
                packed[i, j // 64] |= (np.uint64(1) << np.uint64(j % 64))
    return packed

def compute_bn_threshold(bn_layer):
    mean = bn_layer.running_mean.detach().numpy()
    var = bn_layer.running_var.detach().numpy()
    gamma = bn_layer.weight.detach().numpy() # gamma
    beta = bn_layer.bias.detach().numpy()    # beta
    eps = bn_layer.eps
    
    # Tính std = sqrt(var + eps)
    std = np.sqrt(var + eps)
    
    thresh = mean - (beta * std) / (gamma + 1e-6) # tránh chia cho 0
    
    return thresh.astype(np.float32)

w1 = model.hidden1[0].weight.detach().numpy()
t1 = compute_bn_threshold(model.hidden1[1])

w2 = model.hidden2[0].weight.detach().numpy()
t2 = compute_bn_threshold(model.hidden2[1])

wout = model.output_layer[0].weight.detach().numpy()

with open(MEAN_PATH, 'r') as f:
    input_means = json.load(f)
input_means = np.array(input_means, dtype=np.float32)

np.savez(
    OUT_PATH,
    w1=pack_binary_matrix(w1),
    t1 =t1,
    w2=pack_binary_matrix(w2),
    t2 =t2,
    wout=pack_binary_matrix(wout),
    input_means=input_means
)

print("Exported weights to", OUT_PATH)
