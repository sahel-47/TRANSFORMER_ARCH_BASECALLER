from build import MATRIX_INNER_ENGINE
import numpy as np

# --- 1. Setup Data ---
np.random.seed(11)
# Input: 10 images, 16 pixels each
input_matrix = np.random.randint(low=0, high=100, size=(10, 16)).astype(np.int32)
# Weights: 16 output features, 16 input pixels
weight_matrix = np.random.randint(low=0, high=100, size=(16, 16)).astype(np.int32)

# --- 2. Calculate Reference (Golden Truth) ---
# We use .T because HLS treats Rows as filters
output_matrix_ref = np.matmul(input_matrix, weight_matrix.T)
print(f"Reference Output Shape: {output_matrix_ref.shape}") # Should be (10, 16)

# --- 3. Helper Functions ---
def flatten_image(image: np.ndarray):
    return image.flatten(order='C')

def pack_weights_cyclic(weight_matrix, NES, TILES, SIMD):
    # ... (Keep your existing function code here) ...
    # (Pasted for completeness of the logic)
    rows, cols = weight_matrix.shape
    reshaped = weight_matrix.reshape(rows // NES, NES, cols)
    transposed = reshaped.transpose(1, 0, 2)
    return transposed.flatten().astype(np.int32)

# --- 4. Prepare Hardware Data ---
# USE A NEW VARIABLE NAME so we don't destroy 'weight_matrix'
packed_weights = pack_weights_cyclic(weight_matrix, 2, 32, 4)

# --- 5. Initialize Hardware ---
# FIX: Pass Depth (500) to constructors
input_fifo_buffer = MATRIX_INNER_ENGINE.stream_fifo4(500)
output_fifo_buffer = MATRIX_INNER_ENGINE.stream_fifo2(500)
Weights_cpp = MATRIX_INNER_ENGINE.PACKED_WEIGHTS()

# Load Data
Weights_cpp.load_weights(packed_weights)
input_fifo_buffer.fill_stream(flatten_image(input_matrix))

# --- 6. Run Engine ---
print("Running Hardware Simulation...")
TUPAC_SHAKUR = MATRIX_INNER_ENGINE.MatrixInnerEngine(input_fifo_buffer, output_fifo_buffer, Weights_cpp, 10)
TUPAC_SHAKUR.Multipy()

# --- 7. Verify Output ---
print("Reading Results...")

# FIX: Calculate exact number of packets to read
# 10 Rows * (16 Features / 2 FeaturesPerPacket) = 80 Packets
TOTAL_PACKETS = 10 * (16 // 2)

hw_results_flat = []

for i in range(TOTAL_PACKETS):
    # Read packet (contains 2 ints)
    packet = output_fifo_buffer.read().view()
    hw_results_flat.append(packet[0])
    hw_results_flat.append(packet[1])

# Convert list to numpy array for comparison
hw_output_matrix = np.array(hw_results_flat).reshape(10, 16)

# Compare
print("\n--- Comparison (First Row) ---")
print(f"Hardware:  {hw_output_matrix[0, :5]} ...")
print(f"Reference: {output_matrix_ref[0, :5]} ...")

if np.array_equal(hw_output_matrix, output_matrix_ref):
    print("\nSUCCESS: Hardware output matches NumPy reference perfectly!")
else:
    print("\nMISMATCH: Check your packing logic or accumulator reset.")