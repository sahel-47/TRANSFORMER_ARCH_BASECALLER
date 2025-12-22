#include<iostream>
#include<cmath>
#include<cstring>
#include<string.h>


#define DIM 512
#define WIN_SIZE_LEFT 127
#define WIN_SIZE_RIGHT 128
#define WIN_SIZE (WIN_SIZE_LEFT + 1 + WIN_SIZE_RIGHT)  //(127,128)

class RMSNorm // [TENSOR] --- (RMS NOM)/ +=RMS NORM([TENSOR] + [RESIDUAL])-----> RMS_NORMALIZED[TENSOR]
{

private:
    float m_eps; // SMALL VALUE TO BE ADDED TO DENOMINATOR TO MAKE IT NON ZERO in some cases
    // float *weight  // Should be passed to this function -- depends on number of features
    int m_seq_len; //Number of tokens
    int m_dim; //DIM of one token
    float* m_rms_norm_vector; // Depends on sequence length



public:
    void rms_norm(float *x, float *weight, float alpha_norm = 1.0,float* residual = nullptr) //func 1

    // x        -----> [SEQ_LEN, DIM]
    // weight   -----> [DIM]
    // residual -----> [SEQ_LEN, DIM] (optional)
    {

        if(residual)
        {
            for(int i = 0; i < m_seq_len; i++)
            {
                for(int j = 0; j < m_dim; j++)
                {
                    x[i*m_dim + j] =  x[i*m_dim + j] + residual[i*m_dim + j] * alpha_norm;
                }
            }
        }

        float sum = 0.0;

        for(int i = 0; i < m_seq_len; i++) //computes RMS_norm vector for each token
        {
            for(int j = 0; j < m_dim; j++)
            {
                sum += x[i*m_dim + j] * x[i*m_dim + j];
            }

            float rms_norm_val = 1/std::sqrt(sum/m_dim+m_eps);

            m_rms_norm_vector[i] = rms_norm_val;

            sum = 0.0;
        }

        for(int i = 0; i < m_seq_len; i++)
        {
            for(int j = 0; j < m_dim; j++)
            {
                x[i * m_dim + j] = x[i * m_dim + j] * m_rms_norm_vector[i] * weight[j];
            }
        }


    }


    RMSNorm(int seq_len, int dim)
    :m_eps{1e-5},
    m_dim{dim},
    m_seq_len{seq_len}
    {
        m_rms_norm_vector = new float[seq_len];
    }

    ~RMSNorm()
    {
        if(m_rms_norm_vector) delete[] m_rms_norm_vector;
    }

};


class RoPE // [Q,K] -----[ROPE]-----> Rotated[Q,K]
{

private:
    int m_dim;
    float m_base;
    int m_seq_len;
    int m_heads; //or n_heads
    int m_hdim;

    void hdim() // func1
    {
        m_hdim = m_dim/m_heads;
    }

public: 
    float* m_inv_freq;

    
    void compute_inv_freq() //func 2
    {
        for(int i = 0; i < m_hdim/2; i++)
        {
            m_inv_freq[i] = 1/std::pow(m_base, (float)(2*i)/(float)m_hdim);
        }
    }

    void rope(float* query ,float* key) // func 3
    {        

        for(int i = 0; i < m_seq_len; i++)
        {
            for(int j = 0 ; j < m_heads; j++)
            {
                int offset = (i * m_dim) + (j * m_hdim);
                for(int k = 0; k < m_hdim/2; k++)
                {
                    float freq_val = float(i) * m_inv_freq[k];
                    float cos_val = std::cos(freq_val);
                    float sin_val = std::sin(freq_val);

                    int idx1 = offset + k;
                    int idx2 = offset + m_hdim/2 + k;

                    float x1 = query[idx1];
                    float y1 = query[idx2];

                    query[idx1] = x1*cos_val - y1*sin_val;
                    query[idx2] = y1*cos_val + x1*sin_val;

                    float x2 = key[idx1];
                    float y2 = key[idx2];

                    key[idx1] = x2*cos_val - y2*sin_val;
                    key[idx2] = y2*cos_val + x2*sin_val;

                }
            }
        }

    }

    RoPE(int seq_len, int dim, int heads, float base = 10000.0)
    :m_seq_len{seq_len},
    m_dim{dim},
    m_heads{heads},
    m_base{base}
    {


        hdim();
        m_inv_freq = new float[m_hdim/2];
        compute_inv_freq();

    }

    ~RoPE()
    {
        if(m_inv_freq) delete[] m_inv_freq;
    }

};


void mtx_mul(float* in, float* out, float *w, int m, int n, int k) 
// For a [mxn] X [nxk] ---> [mxk] matrix mul
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            float final_val = 0.0;
            for(int l = 0; l < n; l++)
            {
                final_val += in[i * n + l] * w[j * n + l];

            }

            out[i * k + j] = final_val;
        }
    }
}


template<int MAXSIZE>
void softmax(float* x)
{
    float buffer[MAXSIZE];
    float max_val = x[0];

    max:
    for(int i = 1; i < MAXSIZE; i++)
    {
        float x_i = x[i];
        
        if(x_i > max_val)
        {
            max_val = x_i;
        }
    }

    exp:
    for(int i = 0; i < MAXSIZE; i++)
    {
        float x_i = std::exp(x[i] - max_val);
        buffer[i] = x_i;
    }

    float sum = 0.0;
    sum:
    for(int i = 0; i < MAXSIZE; i++)
    {
        sum += buffer[i];
    }

    float inv_sum = 1.0/sum;

    sofmax:
    for(int i = 0; i < MAXSIZE; i++)
    {
        x[i] = buffer[i] * inv_sum;
    }

}

class MultiHeadAttention // 
{

private:
    int m_num_heads;
    int m_seq_len;
    int m_dim;
    int m_hdim;

    void hdim()
    {
        m_hdim = m_dim/m_num_heads;
    }


public:

    void attn_function(float* query, float* key, float* value, float* out)
    {
        for(int i = 0; i < m_seq_len; i++)
        {
            float attn_scores[WIN_SIZE]; // stores scores for one head at a time
            float wv[m_dim]; //Stores one full vector of Final weighted value
            std::memset(wv, 0, m_dim* sizeof(float));

            for(int j = 0; j < m_num_heads; j++)
            {
                int q_offset = i * m_dim + j * m_hdim;
                int start_key = i - WIN_SIZE_LEFT;

                for(int k = 0; k < WIN_SIZE; k++)
                {
                    int check = start_key + k;

                    if(check >= 0 && check < m_seq_len)
                    {
                        int k_idx = check * m_dim + j * m_hdim;
                        float attention_score = 0.0;

                        for(int l = 0; l < m_hdim; l++)
                        {
                            attention_score += query[q_offset + l] * key[k_idx + l];
                            
                        }

                        attention_score = attention_score/std::sqrt(m_hdim);
                        attn_scores[k] = attention_score;

                    }

                    else
                    {
                        attn_scores[k] = -10000.0;
                    }

                }

                softmax<WIN_SIZE>(attn_scores);
                int wv_idx = j * m_hdim;
                

                for(int k = 0; k < WIN_SIZE; k++)
                {
                    int check = start_key + k;

                    if(check >= 0 && check < m_seq_len)
                    {
                        int v_idx = check * m_dim + j * m_hdim;
                        float a = attn_scores[k];

                        for(int l = 0; l < m_hdim; l++)
                        {
                            wv[wv_idx + l] += a * value[v_idx + l];
                            
                        }
                    }

                }



            }

            for(int j = 0 ; j < m_dim; j++)
            {
                out[i*m_dim + j] = wv[j];
            }

        }



    }


    MultiHeadAttention(int num_heads, int seq_len, int dim)
    :m_dim{dim},
    m_seq_len{seq_len},
    m_num_heads{num_heads}
    {
        hdim();
    }

};

class AttentionBlock
{   

private:
    int m_seq_len;
    int m_dim;
    int m_num_heads;
    // float alpha;

    RoPE rope_layer;
    MultiHeadAttention MHA;
    RMSNorm rmsnorm;



    //--------------------------------------------------------
    float* W_q;
    float* W_k;
    float* W_v;
    float* W_out;

    float* intermediate_tensor_q;
    float* intermediate_tensor_k;
    float* intermediate_tensor_v;

    float* out1;


    float* rms_weight;


public:

    void attention_forward(float *x, float* out)
    {
        mtx_mul(x, intermediate_tensor_q, W_q, m_seq_len, m_dim, m_dim);
        mtx_mul(x, intermediate_tensor_k, W_k, m_seq_len, m_dim, m_dim);
        mtx_mul(x, intermediate_tensor_v, W_v, m_seq_len, m_dim, m_dim);

        rope_layer.rope(intermediate_tensor_q, intermediate_tensor_k);
        MHA.attn_function(intermediate_tensor_q, intermediate_tensor_k, intermediate_tensor_v, out1);

        mtx_mul(out1, out, W_out, m_seq_len, m_dim, m_dim );

        rmsnorm.rms_norm(out, rms_weight, 1.0, x);

    }


    AttentionBlock(int seq_len, int dim, int num_heads)
    :m_seq_len{seq_len},
    m_dim{dim},
    m_num_heads{num_heads},
    rope_layer(seq_len, dim, num_heads),
    MHA(num_heads, seq_len, dim),
    rmsnorm(seq_len, dim)

    {
        int weight_size = seq_len * dim ;

        intermediate_tensor_q = new float[weight_size];
        intermediate_tensor_k = new float[weight_size];
       intermediate_tensor_v = new float[weight_size];
        out1 = new float[weight_size];

    }

    ~AttentionBlock()
    {
        if(intermediate_tensor_q) delete[] intermediate_tensor_q;
        if(intermediate_tensor_k) delete[] intermediate_tensor_k;
        if(intermediate_tensor_v) delete[] intermediate_tensor_v;
        if(out1) delete[] out1;
    }

};


class Gated_FFN
{
private:
    int m_seq_len;
    int m_in_features;
    int m_hidden_features;
    int m_out_features;
    // std::string m_activation;

    // Weights
    float* W_hidden;
    float* W_gate;
    float* W_output;

    //Intermediates
    float* matrix_proj;
    float* matrix_gate;
    float* FFN_out;

    //layers

    RMSNorm rms_layer;
    float* rms_weight;
    float alpha_norm;


    void FFN(float* x, float* out)
    {
        mtx_mul(x, matrix_proj, W_hidden, m_seq_len, m_in_features, m_hidden_features);
        mtx_mul(x, matrix_gate, W_gate, m_seq_len, m_in_features, m_hidden_features);

        swiglu(matrix_proj, matrix_gate);
        mtx_mul(matrix_proj, FFN_out, W_output, m_seq_len, m_hidden_features, m_out_features);
        rms_layer.rms_norm(FFN_out, rms_weight, alpha_norm, x);




    }

    void swiglu(float* matrix, float* gate)
    {
        for(int i = 0; i < m_seq_len; i++)
        {
            for(int j = 0; j < m_hidden_features; j++)
            {
                gate[i * m_hidden_features+j] *= (1.0/(1+std::exp(-gate[i * m_hidden_features+j])));
                matrix[i*m_hidden_features+j] *=gate[i*m_hidden_features+j];
            }
        }

    }

    // Gated_FFN(int seq_len, int in_features, int hidden_features, int out_features)
    // :m_seq_len{seq_len},
    // m_in_features{in_features},
    // m_hidden_features{hidden_features},
    // m_out_features{out_features} {}
    // rms_layer()
    // {

    // }   
};































// int main()
// {
//     float input_tensor[]= {-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920,
//                             -0.3160, -2.1152, 0.3223, -1.2633,  0.3500,  0.3081,
//                         0.5667,  0.7935,  0.5988, -1.5551, -0.3414,  1.8530,
//                       0.7502, -0.5855, -0.1734,  0.1835, 1.3894,  1.5863,
//                       0.4397,  0.1124,  0.5433, -0.3952, 0.2055, -0.4503,
//                      -0.5731, -0.5554,  0.5943,  1.5419, 1.8197, -0.5515};

//     RoPE rope_layer(6,10000,3,3);

//     rope_layer.forward(input_tensor);

//     for(int i = 0; i < 2*3*6; i++)
//     {
//         std::cout<< input_tensor[i]<<std::endl;
//     }


// }


// int main()
// {
//     float x[] = {0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901, 0.8964};

//     RMSNorm rms_layer(4,2);

//     rms_layer.rms_norm(x);

//     for(int i = 0; i < 4*2; i++)
//     {
//         std::cout << x[i] <<std::endl;
//     }

//     return 0;

// 


// int main()
// {
// //     float input_tensor[]= {-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920,
// // //                             -0.3160, -2.1152, 0.3223, -1.2633,  0.3500,  0.3081,
// // //                         0.5667,  0.7935,  0.5988, -1.5551, -0.3414,  1.8530,
// // //                       0.7502, -0.5855, -0.1734,  0.1835, 1.3894,  1.5863,
// // //                       0.4397,  0.1124,  0.5433, -0.3952, 0.2055, -0.4503,
// // //                      -0.5731, -0.5554,  0.5943,  1.5419, 1.8197, -0.5515};

//     RoPE rope_layer(512,10000.0,1000,8);

//     for(int i = 0; i < 32; i++)
//     {
//         std::cout<<rope_layer.inv_freq[i]<<", ";

//     }

//     return 0;
// }


// 1, 0.749894, 0.562341, 0.421697, 0.316228, 0.237137, 0.177828, 0.133352, 0.1, 0.0749894, 0.0562341, 0.0421696, 0.0316228, 0.0237137, 0.0177828, 0.0133352, 0.01, 0.00749894, 0.00562341, 0.00421696, 0.00316228, 0.00237137, 0.00177828, 0.00133352, 0.001, 0.000749894, 0.000562341, 0.000421696, 0.000316228, 0.000237137, 0.000177828, 0.000133352, [1] + Done    