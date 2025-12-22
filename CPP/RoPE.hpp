#include<iostream>
#include<cmath>


#define DIM  512
#define H_DIM 64
#define HEADS 8
#define THETA 10000
#define SEQ_LEN 10



class RoPE
{
private:
    int m_dim;
    int m_hdim;
    float m_base_freq;
    int m_seq_len;


public:


    float* m_freq_table {nullptr}; // Only ROM required probably

    void pre_compute_freq_table()
    {
        if(m_freq_table)
        {
            for(int i = 0; i < m_hdim/2; i++)
            {
                float freq =1.0/std::pow(m_base_freq,(float)(2*i)/(float)(m_hdim));
                m_freq_table[i] = freq;
            }
        }
    }

    void forward(float* query, float* key)
    {
        for(int pos = 0; pos< m_seq_len; pos++)
        {
            for(int i = 0; i < m_dim; i+=2)
            {
                int index = i % m_hdim;
                float val = pos * m_freq_table[index/2];
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                
                int idx1 = (pos * m_dim) + i;
                int idx2 = (pos * m_dim) + i+1;

                float q_i1 = query[idx1];
                float q_i2 = query[idx2];

                query[idx1] = q_i1 * cos_val - q_i2 * sin_val;
                query[idx2] = q_i1*sin_val + q_i2 * cos_val;

                float k_i1 = key[idx1];
                float k_i2 = key[idx2];

                key[idx1] = k_i1 * cos_val - k_i2 * sin_val;
                key[idx2] = k_i1*sin_val + k_i2 * cos_val;

            }
        }
        
    }

    void forward_half_split(float* query, float* key)
    {

        
    }


    RoPE(int dim = DIM, int hdim = H_DIM, float base_freq = THETA, int seq_len = SEQ_LEN, bool interleaved = true)
    : m_dim{dim},
      m_hdim{hdim},
      m_base_freq{base_freq},
      m_seq_len{seq_len}
    {
        if(interleaved)
        {
            m_freq_table = new float[m_hdim/2];
            pre_compute_freq_table();
        }

    }

    ~RoPE()
    {
        if(m_freq_table) delete[] m_freq_table;
    }

};


// class RoPE{

// private:
//     int m_dim;
//     int m_seq_len;
//     float m_theta_base;

// public:

//     float* m_Wk;
//     float* m_pos_Wk;
//     float* m_cos_emb;
//     float* m_sin_emb;


//     void inverse_frequency()
//     {

//         for(int i = 0 ; i < m_dim/2; i++)
//         {
//             m_Wk[i] = 1.0/std::pow(m_theta_base,(float)(2*i)/(float)(m_dim));
//         }

//     }
//   void postionwise_inverse_frequency()
//     {
//         for(int i = 0; i < m_seq_len; i++)
//         {
//             for(int j = 0; j < m_dim/2; j++)
//             {
//                 m_pos_Wk[i*(m_dim/2)+j] = i * m_Wk[j];
//             }
//         }

//     }

//   float get_angle(int m, int i)
//   {
//     return m_pos_Wk[m * (m_dim/2) + i];
//   }

//   void sin_cos_embeddings()
//   {
//     for(int i = 0; i < m_seq_len * (m_dim/2); i++)
//     {
//         m_cos_emb[i] = std::cos(m_pos_Wk[i]);
//         m_sin_emb[i] = std::sin(m_pos_Wk[i]);
//     }
//   }

//   void compute_RoPE(float* input_embedding)
//   {
//     for(int i =0; i < m_seq_len; i++)
//     {
//         for(int j = 0; j < m_dim/2; j++)
//         {

//         float x = input_embedding[i*(m_dim)+2*j];
//         float y = input_embedding[i*(m_dim)+2*j+1];


//         float cos_val = m_cos_emb[i*m_dim/2+j];
//         float sin_val = m_sin_emb[i*m_dim/2+j];

//         float x_final = x*cos_val - y*sin_val;
//         float y_final = y*cos_val + x*sin_val;

//         input_embedding[i*(m_dim)+2*j] = x_final;
//         input_embedding[i*(m_dim)+2*j+1] = y_final;

//         }
//     }
//   }

//     RoPE(int d = DIM, int t = SEQ_LEN, float freq = THETA)
//         :m_dim{d},
//         m_seq_len{t},
//         m_theta_base{freq}
        
//     {   
//         m_Wk = new float[m_dim/2];
//         m_pos_Wk = new float[m_seq_len * (m_dim/2)];
//         m_cos_emb = new float[m_seq_len * (m_dim/2)];
//         m_sin_emb = new float[m_seq_len * (m_dim/2)];

//         inverse_frequency();
//         postionwise_inverse_frequency();
//         sin_cos_embeddings();

//     }

//     ~RoPE()
//     {
//         if (m_Wk) delete[] m_Wk;
//         if(m_pos_Wk) delete[] m_pos_Wk;
//         if (m_cos_emb) delete[] m_cos_emb;
//         if(m_sin_emb) delete[] m_sin_emb;
//     }

// };


