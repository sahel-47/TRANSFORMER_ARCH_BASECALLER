import numpy as np
import queue
from build.ConvInpGen import hls_stream, ConvInputGen, ConvInputGen_advanced, ConvInputGen_1D




## BOTH HLS_STREAM AND CONVINPUTGEN-WINDOW works

def create_image(seed: int, size: tuple):

    if(len(size) not in (2,3)):
        raise ValueError("The size must be specified as (H, W, C)---> Cant have lesser than 3 arguments")
    
    np.random.seed(seed)
    return np.random.randint(0, 100, size = size, dtype = np.int32)


# img1 = create_image(42, (10, 10 ,3))


def flatten_image(image : np.ndarray):
    return image.flatten(order= 'C')



input_buffer = hls_stream()


image = create_image(42, (10,10,8))

# print(image[0:3,:,:])
print()
image = flatten_image(image)

input_buffer.fill_stream(image)

output_buffer = hls_stream()

window_gen = ConvInputGen_advanced(input_buffer, output_buffer)

window_gen.generator()



# for i in range(2* (9*2)):
#     print(i)
#     print(output_buffer.read().data)

# window_gen = ConvInputGen(input_buffer, output_buffer)

# window_gen.generator()


# for i in range(9*2):
#     output_buffer.read().data

# print("************************************************")

# for i in range(2*(9*2)):
#         print(i)
#         print(output_buffer.read().data)



########################################## 1D TEST ############################################3


signal_1d = create_image(42,(10,8))



print(signal_1d)

input_stream_1d = hls_stream()

signal_1d =  flatten_image(signal_1d)

input_stream_1d.fill_stream(signal_1d)

output_stream_1d = hls_stream()

win_gen_1d = ConvInputGen_1D(input_stream_1d, output_stream_1d)

win_gen_1d.generator()



for i in range(3*6):

    print(output_stream_1d.read().data)


