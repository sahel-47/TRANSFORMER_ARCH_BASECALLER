import torch 
import os 


DIM = 16
SEQ_LEN = 10
THETA  = 10000.0


class StandardRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, interleaved=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved

    def forward(self, qkv):
       
        seq_len = qkv.shape[1]
        device = qkv.device

        # Create theta
        # MODIFIED: Removed device="cuda" and changed .half() to .float() for golden verification
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Create cos and sin embeddings 
        cos_emb = emb.cos()[None, :, None, None, :]
        sin_emb = emb.sin()[None, :, None, None, :]

        # Separate q, k, v
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # Apply rotary embeddings to q and k
        q_rot = (q * cos_emb) + (self._rotate_half(q) * sin_emb)
        k_rot = (k * cos_emb) + (self._rotate_half(k) * sin_emb)

        # Reassemble and return
        return torch.cat([q_rot, k_rot, v], dim=2)

    def _rotate_half(self, x):
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 : self.dim]
        return torch.cat((-x2, x1), dim=-1)


torch.manual_seed(77)

N,H = 1, 1

qkv_input = torch.randn(N,SEQ_LEN,3,H,DIM)

model = StandardRotaryEmbedding(DIM,base = THETA)

qkv_output = model(qkv_input)

q_out, k_out, v_out = torch.chunk(qkv_output,3, dim=2)

q_in, k_in, v_in = torch.chunk(qkv_input, 3, dim=2)

print("PYTORCH RESULTS")
print(f"Input Q shape: {q_in.squeeze().shape}") # Should be (SEQ_LEN, DIM)


def save_tensor(tensor, filename):
    # Flatten and save one number per line
    flat = tensor.flatten().tolist()
    with open(filename, 'w') as f:
        for val in flat:
            f.write(f"{val:.8f}\n")
    print(f"Saved {filename}")

# Save the squeezed versions (removing batch/head dims for simple C++ test)
save_tensor(q_in.squeeze(), "q_input.txt")
save_tensor(q_out.squeeze(), "q_result_golden.txt")

# import torch
# import os

# # --- 1. Define Constraints ---
# DIM = 16
# SEQ_LEN = 10
# THETA = 10000.0

# # --- 2. Define the "Neighbor-Pair" RoPE (HLS Friendly) ---
# class NeighborRoPE(torch.nn.Module):
#     def __init__(self, dim, base=10000):
#         super().__init__()
#         self.dim = dim
#         self.base = base

#     def forward(self, x):
#         # x shape: (N, T, 3, H, D) or just (T, D) for testing
#         # We assume input x is just the Q or K tensor of shape (..., Dim)
        
#         seq_len = x.shape[-2] # Time dimension
#         dim = x.shape[-1]     # Head Dimension
#         device = x.device

#         # 1. Create Theta frequencies (Same formula)
#         # theta_i = 1.0 / (base ^ (2i / dim))
#         # Shape: (Dim / 2)
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        
#         # 2. Create Position Angles
#         # Shape: (Seq_Len, Dim / 2)
#         t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
        
#         # 3. Expand Frequencies for Neighbor Pairs
#         # We want [theta_0, theta_0, theta_1, theta_1, ...]
#         # Current 'freqs' is [theta_0, theta_1, ...]
#         # We repeat it interleave-style
#         emb = torch.repeat_interleave(freqs, 2, dim=-1) # Shape: (Seq_Len, Dim)

#         # 4. Calculate Cos/Sin
#         cos = emb.cos()
#         sin = emb.sin()

#         # 5. Apply Neighbor-Pair Rotation
#         # Pairs are (0,1), (2,3), etc.
#         # x_evens = [x0, x2, x4...]
#         # x_odds  = [x1, x3, x5...]
#         x_evens = x[..., 0::2]
#         x_odds  = x[..., 1::2]
        
#         # We need to slice cos/sin to match the pairs
#         # (They are currently duplicated, so we just take the evens)
#         cos_half = cos[..., 0::2]
#         sin_half = sin[..., 0::2]

#         # Formula:
#         # x_new_even = x_even * cos - x_odd * sin
#         # x_new_odd  = x_even * sin + x_odd * cos
#         x_evens_rot = (x_evens * cos_half) - (x_odds * sin_half)
#         x_odds_rot  = (x_evens * sin_half) + (x_odds * cos_half)

#         # 6. Interleave back together
#         # Stack them [even, odd] along the last dimension and flatten
#         x_out = torch.stack((x_evens_rot, x_odds_rot), dim=-1).flatten(-2)
        
#         return x_out

# # --- 3. Generate Data ---
# torch.manual_seed(77)

# # Create random input (1 Batch, 10 Time, 16 Dim)
# # We create separate Q and K to keep it simple
# q_input = torch.randn(1, SEQ_LEN, DIM)

# # Run Model
# model = NeighborRoPE(DIM, base=THETA)
# q_output = model(q_input)

# print("PYTORCH RESULTS (Neighbor-Pair)")
# print(f"Input Q shape: {q_input.squeeze().shape}") 

# # --- 4. Save to Files ---
# def save_tensor(tensor, filename):
#     flat = tensor.flatten().tolist()
#     with open(filename, 'w') as f:
#         for val in flat:
#             f.write(f"{val:.8f}\n")
#     print(f"Saved {filename}")

# save_tensor(q_input.squeeze(), "q_input.txt")
# save_tensor(q_output.squeeze(), "q_result_golden.txt")

# # Verify Time Step 0 (Should be identical)
# print("\nVerifying Time Step 0 (Should be almost identical):")
# print("In: ", q_input.squeeze()[0, :4].tolist())
# print("Out:", q_output.squeeze()[0, :4].tolist())

# # Verify Time Step 1 (Should be rotated)
# print("\nVerifying Time Step 1 (Should be rotated):")
# print("In: ", q_input.squeeze()[1, :4].tolist())
# print("Out:", q_output.squeeze()[1, :4].tolist())