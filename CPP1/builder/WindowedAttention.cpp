#include<iostream>
#include"stream.hpp"


template<unsigned int SEQ_LEN, unsigned int HDIM, unsigned int SIMD, unsigned int WIN_SIZE = 256, unsigned int WIN_SIZE_RIGHT = 128, unsigned int WIN_SIZE_LEFT = 127> // First with just plain attention no packed

class WindowedAttention
{
    public:
        
        hls_stream<int,1>& query_stream;
        hls_stream<int,1>& key_stream;
        hls_stream<int,1>& out_stream;



        WindowedAttention(hls_stream<int,1>& QS, hls_stream<int,1>& KS, hls_stream<int,1>& OS)
        : query_stream{QS},
         key_stream{KS},
         out_stream{OS}
         {

         }

    // SIMD ---> number of multipliers parallely reading data from a head (Must be HDIM % SIMD == 0)
    // I have to store KEYS right?????
        // constexpr unsigned int total_k_size = HDIM * (WIN_SIZE + 1); 
        int k_buf[WIN_SIZE+1][HDIM]; //+1 for buffering
        int q_buf[2][HDIM];
        // #pragma ARRAY_PARTITION_ var = k_buf type = cyclic  factor = SIMD  
        // #pragma ARRAY_PARTITION_ var = q_buf type = cyclic  factor = SIMD 

        // constexpr unsigned int total_v_size = HDIM * (WIN_SIZE+1);
        
        // const unsigned int calculate_total_ops = [](unsigned int seq_len = SEQ_LEN, unsigned int win_size_R = WIN_SIZE_RIGHT, unsigned int win_size_L = WIN_SIZE_LEFT){

        //     unsigned int total_ops{};
        //     for(int i = 0; i < seq_len; i++)
        //     {
        //       unsigned int  start_index = std::max(0, static_cast<int>(i - win_size_L) );
        //       unsigned int end_index = std::min(seq_len, i + win_size_R + 1);
        //       unsigned int valid_keys = end_index - start_index;
        //       total_ops += valid_keys;
                
        //     }

        //     return total_ops;
        // };
        // const unsigned int read_cycles = HDIM; // read cycles to load a row
        // const unsigned int multiplying_factor = HDIM/SIMD; // Cycles to write one attention score between heads of one query and key ALWAYS MORE THAN read_cycles
        const unsigned int write_cycles = multiplying_factor; //FOr now assume its int 
        // const unsigned int fill_full_buffer = total_k_size; 
        
        // const unsigned int free_flow_start_cycle = read_cycles * WIN_SIZE_RIGHT; // after this we have plenty of time to read because the new query starts from the begenning

        // static constexpr unsigned int Total_iter = (calculate_total_ops() * multiplying_factor) + read_cycles;        Total_iter *=multiplying_factor 


        static constexpr unsigned int calculate_total_ops() {
        unsigned int total_ops = 0;
        for(unsigned int i = 0; i < SEQ_LEN; i++) {
            // Use signed int for the subtraction to avoid underflow before max()
            int start_idx = ( (int)i - (int)WIN_SIZE_LEFT > 0 ) ? (i - WIN_SIZE_LEFT) : 0;
            unsigned int end_index = (i + WIN_SIZE_RIGHT + 1 < SEQ_LEN) ? (i + WIN_SIZE_RIGHT + 1) : SEQ_LEN;
            total_ops += (end_index - (unsigned int)start_idx);
        }
        return total_ops;
    }

    // 2. Define constants correctly
    static constexpr unsigned int multiplying_factor = HDIM / SIMD;
    static constexpr unsigned int read_cycles = HDIM;
    
    // 3. Initialize Total_iter using the helper
    static constexpr unsigned int Total_iter = (calculate_total_ops() * multiplying_factor) + read_cycles;
        
        int attn_scores[WIN_SIZE];


        
        void attn_function()
        {

            bool free_flow_start = false;
            bool load_enable = true;
            bool load_q_enable = true;
        

            unsigned int kwx = 0;
            unsigned int kwy = 0;
            unsigned int krx = 0;
            unsigned int kry = 0; //SOME POINTER
            
            unsigned int qrx = 0;
            
            unsigned int qwx = 0;
            bool qwy = 0;
            bool qry = 0;
            

            unsigned int ky_index = 0; // kry + key_start
          


            // unsigned int simd_count = 0; //----> For counting how much of input is processed

            unsigned int key_start = 0;// Where does a query start reading from 
            unsigned int write_dest = 0;
            unsigned int key_row_count = 0;
            unsigned int keys_processed_count = 0;
            unsigned int query_row_count = 0;
            unsigned int internal_counter = 0; 


            unsigned int inp = 0; // Counts upto first read_cycles to fill the very first Q and K

            unsigned int finish_read = 0;
            unsigned int read_till = WIN_SIZE_RIGHT;
            int acc_value = 0;


            unsigned int attn_score_counter  = 0;


            for(int i = 0; i < 10000; i++) //TOMORROW : THIS LOOP COUNT GAVE SEG FAULT
            {

                if(inp < read_cycles)
                {
                     int q_val = query_stream.read();
                     int k_val = key_stream.read();

                    k_buf[kwy][kwx] = k_val;
                    q_buf[qwy][qwx] = q_val;

                    kwx++;
                    qwx++;
                    inp++; 

                    if(inp == read_cycles)
                    {
                        kwx = 0;
                        kwy++; 
                        qwx = 0;
                        qwy = !qwy;
                        key_row_count++; // FIRST ROW READDD
                    }
                    
                }

                else
                {
                    //Case1: layer is still loading but we compute N-1th row which was previously fetched while loading Nth row
                    //Case2: free flow we restarted the computation
                    //Case3: Full buffer is filled !!!

                    if(internal_counter < write_cycles || free_flow_start)
                    {

                        for(int sd = 0; sd < SIMD; sd++)
                        //pragma UNROLL
                        {
                            acc_value += q_buf[qry][qrx+sd] * k_buf[ky_index][krx+sd];

                        }
                        qrx += SIMD;
                        krx += SIMD;

                        if(qrx >= HDIM)
                        {
                            qrx = 0;
                            krx = 0;
                            attn_scores[attn_score_counter] = acc_value;
                            acc_value = 0;
                            attn_score_counter++;
                            keys_processed_count++;
                            kry++;
                            ky_index = key_start + kry;

                            if(ky_index >= WIN_SIZE+1)
                            {
                                ky_index -= WIN_SIZE+1;
                            }
                            // finish_read++;

                            if(attn_score_counter == read_till - key_start)
                            {
                                free_flow_start = true;

                                if(load_enable == false)
                                {
                                    load_enable = true;
                                    load_q_enable = true;
                                }
                                // finish_read = 0;
                                read_till++;
                                
                                query_row_count++;
                                qry = !qry;
                                keys_processed_count = 0;

                                if(read_till == SEQ_LEN)
                                {
                                    read_till = SEQ_LEN-1;
                                }

                                if(query_row_count > WIN_SIZE_LEFT) //same as >= WIN_SIZE_LEFT+1
                                {
                                    key_start++;

                                    if(key_start >= WIN_SIZE+1)
                                    {
                                        key_start = 0;
                                    }
                                }

                                for(int pr = 0; pr < attn_score_counter ; pr++)
                                {
                                    out_stream.write(attn_scores[pr]); // THIS IS CAUSING PROBLEMS I FEEL
                                }
                                attn_score_counter = 0;
                                
                            }
                        }
                    }

                    if(internal_counter < read_cycles && load_enable) // ONE ROW FULL FILL
                    {
                        int k_val = key_stream.read();
                        k_buf[kwy][kwx] = k_val;
                        kwx++;

                        if(load_q_enable)
                        {
                            int q_val = query_stream.read();
                            q_buf[qwy][qwx] = q_val;
                            qwx++;
                        }



                        if(internal_counter == read_cycles-1)
                        {
                            
                            kwx = 0;
                            kwy++;
                            key_row_count++;
                           
                            if(load_q_enable)
                            {
                                qwx = 0;
                                qwy = !qwy;
                                load_q_enable = false; //TODO will be set true in compute engine
                            }

                            if(key_row_count >= WIN_SIZE+1)
                            {   
                                
                                load_enable = false; //TODO once set false will be set true in the compute engine
                            }

                            if(kwy == WIN_SIZE+1)
                            {
                                kwy -= WIN_SIZE+1;
                            }


                        }


                    }

                    internal_counter++;

                    if(internal_counter==read_cycles)
                    {
                        internal_counter = 0;
                    }


             }

        }

    }
};


//ONLY READ