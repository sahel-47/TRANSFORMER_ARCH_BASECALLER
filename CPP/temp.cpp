
class MultiHeadAttention
{

private:
    int m_num_heads;
    int m_seq_len;
    float alpha;
    int m_hdim;
    int m_dim;

    float* W_qkv;
    float* W_out; //(DIM, DIM)
    float* W_norm;

    float* resid;
    float* qkv;
    float* q; // [SEQ LEN, DIM]
    float* k;// [SEQ LEN, DIM]
    float* v;// [SEQ LEN, DIM]
    float* attn_output;

    float* wv; // dim


public:

    int total_size = m_seq_len * DIM;


    void linear(float* in, float* out, float* w, int seq_len, int in_dim, int out_dim)
    {
        for(int i = 0; i < seq_len; i++)
        {
            for(int j = 0; j < out_dim; j++)
            {
                float score = 0.0;
                for(int k = 0; k < in_dim; k++)
                {
                    score += in[i*in_dim + k] * w[k*out_dim+j];
                   
                }
                out[i*out_dim + j] = score;
        
            }
        }
    
    }
    template<int MAXSIZE>
    void softmax(float *x, int size)
    {
        float buffer[MAXSIZE];
        float max_val = x[0];
        max:
        for(int i = 1; i < size; i++)
        {
            float x_i = x[i];

            if(x_i > max_val)
            {
                max_val = x_i;
            }
        }

        exp:
        for(int i = 0; i < size; i++)
        {
            float x_i = std::exp(x[i]- max_val);
            buffer[i] = x_i;
        }

        float sum = 0.0;
        sum:
        for(int i = 0; i < size; i++)
        {
            sum += buffer[i];
        }

        float inv_sum = 1.0/sum;

        for(int i = 0; i < size; i++)
        {
            x[i] = buffer[i] * inv_sum;
        }
    }

    void attn_function(float *q, float *k, float *v, float *out)
    {
        for(int i = 0; i < m_seq_len; i++)
        {
            float attn_buffer[WIN_SIZE] ; // HLSTransform uses (numheads * seq len) ---> 

            for(int h = 0; h < m_num_heads; h++)
            {
                int q_offset = i*m_dim + h*m_hdim;
                int attn_offset = h*WIN_SIZE;
                int start_key = i - WIN_SIZE_LEFT;

                for(int at = 0; at < WIN_SIZE; at++) // FIXED WINDOW CHECK DYNAMIC LATER
                {
                    int check = start_key + at;
                    // int k_idx = 0;
                    

                    if(check >= 0 && check < m_seq_len)
                    {
                        int k_idx = (check * m_dim) + h*m_hdim;
                        float score = 0.0;

                        for(int hd = 0; hd < m_hdim; hd++)
                        {
                            // #pragma HLS UNROLL ----->HERE
                            score += q[q_offset+hd] * k[k_idx+hd];
                        }
                        score /= std::sqrt(m_hdim);
                        attn_buffer[at] = score; // [attn_offset + at] if buffer size(WIN LEN * HEADS)
                    }

                    else
                    {
                        attn_buffer[at] = -10000.0;
                    }
                }

                softmax<WIN_SIZE>(attn_buffer, WIN_SIZE);

                int wv_offset = h*m_hdim; // coz im just keeping it dim size

                std::memset(wv+wv_offset, 0, m_hdim*sizeof(float));

                for(int t = 0; t < WIN_SIZE; t++)
                {

                    int check = start_key + t;

                    if(check >= 0 && check < m_seq_len)
                    {
                        int v_offset = check*m_dim + h*m_hdim;
                        float a = attn_buffer[t];

                        for(int ll = 0; ll < m_hdim; ll++)
                        {
                            wv[h*m_hdim+ll] += a*v[v_offset+ll];
                        }


                    }


                }

                for(int ll = 0; ll < m_hdim; ll++)
                {
                    out[q_offset+ll] = wv[h*m_hdim+ll];
                }

            }
               

        }

    }

};
