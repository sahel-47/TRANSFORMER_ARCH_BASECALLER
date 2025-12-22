template <int TOTAL_SIZE>
void RoPE(float q[TOTAL_SIZE], float k[TOTAL_SIZE], const float inv_freq[H_DIM]) // [Q] / [K] ----[R.O.P.E]----> rotated([Q]) / rotated([K])
{
    constexpr auto array_size = TOTAL_SIZE * sizeof(float);
    constexpr int half_head = H_DIM/2;
    constexpr int total_pairs = DIM/2;


    float q_buffer[TOTAL_SIZE];
    float k_buffer[TOTAL_SIZE];

    #pragma HLS ARRAY_PARTITION variable=q_buffer type=cyclic factor = 64
    #pragma HLS ARRAY_PARTITION variable=k_buffer type=cyclic factor = 64

    std::memcpy(q_buffer,q,array_size);
    std::memcpy(k_buffer,k,array_size);


    for(int pos = 0; pos < SEQ_LEN; pos++)
    {
        for(int h = 0; h < HEADS; h++)
        {
            #pragma HLS PIPELINE II = 1
            int offset = (pos*DIM) + (h*H_DIM);
            for(int i = 0; i < H_DIM/2; i++)
            {
                #pragma HLS UNROLL 
                float freq_val = float(pos) * inv_freq[i];
                float cos_val = cosf(freq_val);
                float sin_val = sinf(freq_val);

                int idx1 = offset +i;
                int idx2 = offset +i +H_DIM/2;

                float x = q_buffer[idx1];
                float y = q_buffer[idx2];

                q_buffer[idx1] = x*cos_val - y*sin_val;
                q_buffer[idx2] = y*cos_val + x*sin_val;

                float a = k_buffer[idx1];
                float b = k_buffer[idx2];

                k_buffer[idx1] = a*cos_val - b*sin_val;
                k_buffer[idx2] = b*cos_val + a*sin_val;


            }

        }
    }

    std::memcpy(q,q_buffer,array_size);
    std::memcpy(k,k_buffer,array_size);
}
