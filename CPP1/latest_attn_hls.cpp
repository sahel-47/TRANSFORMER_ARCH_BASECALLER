#include<hls_stream.h>


template<typename T, unsigned int SEQ_LEN, unsigned int HDIM, unsigned int SIMD, unsigned int WIN_SIZE = 256, unsigned int WIN_SIZE_LEFT = 127, unsigned int WIN_SIZE_RIGHT = 128>


void WindowedAttention(hls::stream<int>& Q_stream,
         hls::stream<int>& K_stream,
         hls::stream<int>& O_stream)
         {
             static_assert(HDIM % SIMD == 0, "HDIM MUST BE DIVISIBLE BY SIMD");

        T K_buf[WIN_SIZE+1][HDIM]; //+1 to enable simulatneously reading and writing keys
        T Q_buf[2][HDIM]; 
        #pragma HLS ARRAY_PARTITION variable=K_buf dim=2 type=cyclic factor = SIMD

        #pragma HLS ARRAY_PARTITION variable=Q_buf dim=2 type=cyclic factor = SIMD
        static constexpr int compute_factor = HDIM/SIMD; //ASSUMING II=1 !!
        static constexpr unsigned int write_cycles = compute_factor;
        static constexpr unsigned int read_cycles = HDIM; //should add SIMD packets if any


        // 1. Calculate the compute factor (Cycles per dot product)
static constexpr unsigned int OPS_PER_SCORE = HDIM / SIMD;

// 2. Calculate operations during the "Ramp Up" phase (Triangle: 1+2+...+WIN_SIZE)
//    Formula: n*(n+1)/2
static constexpr unsigned int RAMP_UP_SCORES = (WIN_SIZE * (WIN_SIZE + 1)) / 2;

// 3. Calculate operations during the "Steady State" phase (Rectangle)
//    Tokens remaining after ramp up * Full Window Size
static constexpr unsigned int STEADY_STATE_TOKENS = (SEQ_LEN > WIN_SIZE) ? (SEQ_LEN - WIN_SIZE) : 0;
static constexpr unsigned int STEADY_STATE_SCORES = STEADY_STATE_TOKENS * WIN_SIZE;

// 4. Total Compute Cycles needed
static constexpr unsigned int COMPUTE_CYCLES = (RAMP_UP_SCORES + STEADY_STATE_SCORES) * OPS_PER_SCORE;

// 5. Add a small safety margin for pipeline flush/fill (Overhead)
//    Adding HDIM covers the initial read phase.
const unsigned int total_iter = COMPUTE_CYCLES + HDIM + 100;

        // const unsigned int total_iter = 3942224;
        T attn_scores[WIN_SIZE+1];

            bool unrestricted_compute = false; //Once set true will compute unconditionally
            bool load_k_enable = true; // Will be set false once the window is full
            bool load_q_enable = true; // decides when to load the (n+1)th query
            bool active_q_load = false;
            bool active_k_load = false;
            bool ping_pong = false;

            T acc_even = 0;
            T acc_odd = 0;
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
    #pragma HLS PIPELINE II=1

    #pragma HLS DEPENDENCE variable=K_buf type=intra dependent=false
    #pragma HLS DEPENDENCE variable=K_buf type=inter dependent=false

    #pragma HLS DEPENDENCE variable=Q_buf type=intra dependent=false
    #pragma HLS DEPENDENCE variable=Q_buf type=inter dependent=false
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
                        T partial_sum = 0;
                        for(int sd = 0; sd < SIMD; sd++)
                        #pragma HLS UNROLL
                        {
                            partial_sum += Q_buf[qry][qrx+sd] * K_buf[ky_idx][krx+sd];
                        }

                        if(ping_pong == false)
                        {
                            acc_even += partial_sum;
                        }

                        else{
                            acc_odd +=  partial_sum;
                        }

                        ping_pong = !ping_pong;

                        qrx += SIMD;
                        krx += SIMD;

                        if(qrx >= HDIM)
                        {
                            qrx = 0;
                            krx = 0;
                            unsigned int attn_score_width = k_read_end - k_read_start;

                            T final_score = acc_even +acc_odd;

                            attn_scores[attn_score_counter] = final_score;
                            O_stream.write(final_score);
                            // acc_value = 0;


                            acc_even = 0;
                            acc_odd = 0;
        
                            ping_pong = false;

                            attn_score_counter++;
                            kry++;
                            acc_value = 0;

                            ky_idx = key_start + kry;


                            if(ky_idx >= WIN_SIZE+1)
                            {
                                ky_idx -= WIN_SIZE+1;
                            }


                            if(attn_score_counter == attn_score_width)
                            {
                                unrestricted_compute = true;
                                attn_score_counter = 0; 

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


                                // // ONLY FOR DEBUGGING - WILL REMOVE IT LATER:
                                // for(int ro = 0; ro < attn_score_counter; ro++)
                                // {
                                //     O_stream.write(attn_scores[ro]);
                                // }

                                // MOVE IT EARLIER PROBABLY
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
                                
                                active_k_load = false;

                                if(key_row_count >= WIN_SIZE+1)
                                {
                                    load_k_enable = false;
                                }
                                key_row_count++;

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


void WAB(hls::stream<int> &in, hls::stream<int>& in2, hls::stream<int>& out)
{

    #pragma HLS INTERFACE ap_fifo port=in depth=1024
    #pragma HLS INTERFACE ap_fifo port=in2 depth=1024
    #pragma HLS INTERFACE ap_fifo port=out depth=1024
    WindowedAttention<int, 1024, 64, 64, 256,127,128>
    (in, in2, out);
}


// made changes to make it fast
