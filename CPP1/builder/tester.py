from build import MATRIX_INNER_ENGINE
import numpy as np 


np.random.seed(11)

input_matrix = np.random.randint(low=0, high=100, size=(10, 16)).astype(np.int32)
weight_matrix = np.random.randint(low=0, high=100, size=(16, 16)).astype(np.int32)

output_matrix_ref = np.matmul(input_matrix, weight_matrix.T)
print(f"Reference Output Shape: {output_matrix_ref.shape}") # Should be (10, 16)

def flatten_image(image: np.ndarray):
    return image.flatten(order='C')


def pack_weights_cyclic(weight_matrix, NES, TILES, SIMD):
    # ... (Keep your existing function code here) ...
    # (Pasted for completeness of the logic)
    rows, cols = weight_matrix.shape
    reshaped = weight_matrix.reshape(rows // NES, NES, cols)
    transposed = reshaped.transpose(1, 0, 2)
    return transposed.flatten().astype(np.int32)



packed_weights = pack_weights_cyclic(weight_matrix, 2, 32, 4)



input_fifo_buffer = MATRIX_INNER_ENGINE.stream_fifo4()
output_fifo_buffer = MATRIX_INNER_ENGINE.stream_fifo2()

Weights_cpp = MATRIX_INNER_ENGINE.PACKED_WEIGHTS()

Weights_cpp.load_weights(packed_weights)
input_fifo_buffer.fill_stream(flatten_image(input_matrix))

TUPAC_SHAKUR = MATRIX_INNER_ENGINE.MatrixInnerEngine(input_fifo_buffer, output_fifo_buffer, Weights_cpp, 10)
TUPAC_SHAKUR.Multipy()

TOTAL_PACKETS = 80 

for i in range(16):
    print(output_fifo_buffer.read().view())

print("*******************************")

print(output_matrix_ref)