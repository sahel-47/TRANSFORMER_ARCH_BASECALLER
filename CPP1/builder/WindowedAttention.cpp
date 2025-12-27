#include<iostream>
#include"stream.hpp"

#define DEBUG_MODE 0;


template<typename T, unsigned int SEQ_LEN, unsigned int HDIM, unsigned int SIMD, unsigned int WIN_SIZE = 256, unsigned int WIN_SIZE_LEFT = 127, unsigned int WIN_SIZE_RIGHT = 128>

class WindowedAttentionEngine
{
    public:
        
        hls_stream<T,1>& Q_stream;
        hls_stream<T,1>& K_stream;
        hls_stream<T,1>& O_stream;

        WindowedAttentionEngine(hls_stream<T,1>& QS, hls_stream<T,1>& KS, hls_stream<T,1>& OS)
        :Q_stream{QS},
        K_stream{KS},
        O_stream{OS}
        {
        
        }

        static_assert(HDIM % SIMD == 0, "HDIM MUST BE DIVISIBLE BY SIMD");

        T K_buf[WIN_SIZE+1][HDIM]; //+1 to enable simulatneously reading and writing keys
        T Q_buf[2][HDIM]; // Ping-pong buffer design
        // #pragma ARRAY_PARTITION_ var = k_buf type = cyclic  factor = SIMD  
        // #pragma ARRAY_PARTITION_ var = q_buf type = cyclic  factor = SIMD 


         static constexpr unsigned int total_ops = [](unsigned int seq_len = SEQ_LEN, unsigned int win_size_R = WIN_SIZE_RIGHT, unsigned int win_size_L = WIN_SIZE_LEFT) constexpr {

            unsigned int total_ops{};
            for(int i = 0; i < seq_len; i++)
            {
            int start = (i - (int)WIN_SIZE_LEFT < 0) ? 0 : (i - WIN_SIZE_LEFT);
            int end   = (i + WIN_SIZE_RIGHT + 1 > (int)SEQ_LEN) ? SEQ_LEN : (i + WIN_SIZE_RIGHT + 1);
              unsigned int valid_keys = end - start;
              total_ops += valid_keys;
                
            }

            return total_ops;

        }();

        static constexpr int compute_factor = HDIM/SIMD; //ASSUMING II=1 !!
        static constexpr unsigned int write_cycles = compute_factor;
        static constexpr unsigned int read_cycles = HDIM; //should add SIMD packets if any

        static constexpr unsigned int total_iter = (total_ops * compute_factor) + read_cycles + 10000;

        T attn_scores[WIN_SIZE+1];

        void attn()
        {
            //flags
            bool unrestricted_compute = false; //Once set true will compute unconditionally
            bool load_k_enable = true; // Will be set false once the window is full
            bool load_q_enable = true; // decides when to load the (n+1)th query
            bool active_q_load = false;
            bool active_k_load = false;

            //K-buffer pointers
            unsigned int kwx = 0;
            unsigned int kwy = 0;
            unsigned int krx = 0;
            unsigned int kry = 0;

            // Q-buffer pointers
            unsigned int qwx = 0;
            bool qwy = 0;
            unsigned int qrx = 0;
            bool qry = 0;

            unsigned int ky_idx = 0; // kry + key_start - points from the start key in the K-buffer till the end

            
            unsigned int key_start = 0;// start of initial k-row in buffer
            unsigned int key_row_count = 0;
            unsigned int query_row_count = 0;
            unsigned int internal_counter = 0;
            unsigned int query_load_count = 0;
            unsigned int initial_read_count = 0;

            unsigned int k_read_start = 0;
            unsigned int k_read_end = WIN_SIZE_RIGHT+1;

            T acc_value = 0;
            unsigned int attn_score_counter = 0;


            for(int i = 0; i < total_iter; i++)
            {
                if(initial_read_count < read_cycles)
                {
                    T q_val = Q_stream.read();
                    T k_val = K_stream.read();

                    K_buf[kwy][kwx] = k_val;
                    Q_buf[qwy][qwx] = q_val;

                    kwx++;
                    qwx++;
                    initial_read_count++;

                    if(initial_read_count==read_cycles)
                    {
                        kwx = 0;
                        qwx = 0;
                        kwy++;
                        qwy = !qwy;
                        key_row_count++;
                        query_load_count++;
                    }
                }

                else
                {
                    if(internal_counter < write_cycles || unrestricted_compute)
                    {
                        for(int sd = 0; sd < SIMD; sd++)
                        //#PRAGMA UNROLL --> HERE
                        {
                            acc_value += Q_buf[qry][qrx+sd] * K_buf[ky_idx][krx+sd];
                        }

                        qrx += SIMD;
                        krx += SIMD;

                        if(qrx >= HDIM)
                        {
                            qrx = 0;
                            krx = 0;
                            unsigned int attn_score_width = k_read_end - k_read_start;

                            attn_scores[attn_score_counter] = acc_value;
                            acc_value = 0;

                            attn_score_counter++;
                            kry++;

                            ky_idx = key_start + kry;


                            if(ky_idx >= WIN_SIZE+1)
                            {
                                ky_idx -= WIN_SIZE+1;
                            }


                            if(attn_score_counter == attn_score_width)
                            {
                                unrestricted_compute = true;

                                if(load_q_enable == false)
                                {
                                    load_q_enable = true;
                                }

                                if(query_row_count > WIN_SIZE_LEFT)
                                {
                                    load_k_enable = true;
                                }

                                k_read_end++;
                                query_row_count++;
                                qry = !qry;
                                kry = 0;

                                if(k_read_end > SEQ_LEN)
                                {
                                    k_read_end = SEQ_LEN;
                                }

                                if(query_row_count > WIN_SIZE_LEFT)
                                {
                                    k_read_start++;
                                    key_start++;

                                    if(key_start == WIN_SIZE+1)
                                    {
                                        key_start = 0;
                                    }
                                }

                                ky_idx = key_start;


                                // ONLY FOR DEBUGGING - WILL REMOVE IT LATER:
                                for(int ro = 0; ro < attn_score_counter; ro++)
                                {
                                    O_stream.write(attn_scores[ro]);
                                }

                                attn_score_counter = 0; // MOVE IT EARLIER PROBABLY
                            }

                            
                        }
                    }


                    if(internal_counter < read_cycles)
                    {

                        if(load_k_enable && internal_counter == 0 && key_row_count < SEQ_LEN)
                        {
                            active_k_load = true;
                        }

                        if(active_k_load)
                        {

                            T k_val = K_stream.read();
                            K_buf[kwy][kwx] = k_val;
                            kwx++;
                        }

                        if(load_q_enable && internal_counter==0 && query_load_count < SEQ_LEN)
                        {
                            active_q_load = true;
                        }

                        if(active_q_load)
                        {
                            T q_val = Q_stream.read();
                            Q_buf[qwy][qwx] = q_val;
                            qwx++;
                        }

                        if(internal_counter == read_cycles-1)
                        {
                            if(active_k_load)
                            {
                                kwx = 0;
                                kwy++;
                                key_row_count++;
                                active_k_load = false;

                                if(key_row_count >= WIN_SIZE+1)
                                {
                                    load_k_enable = false;
                                }

                                if(kwy == WIN_SIZE+1)
                                {
                                    kwy = 0;
                                }
                            }

                            if(active_q_load)
                            {
                                query_load_count++;
                                qwx = 0;
                                qwy = !qwy;
                                load_q_enable = false;
                                active_q_load = false;

                            }
                            
                        }
                    }



                }

            internal_counter++;

            if(internal_counter == read_cycles)
            {
                internal_counter = 0;
            }
            }


        }


};
