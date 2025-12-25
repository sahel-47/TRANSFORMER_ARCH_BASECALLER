#include<queue>
#pragma once

template<typename T, int SIMD>
class hls_stream 
{
    private:
        std::queue<T> q;

    public:
        hls_stream()
        {

        }
        void write(const T& data)
        {
            q.push(data);
        }

        T read()
        {
            if(q.empty())
            {
                throw std::runtime_error("QUEUE IS EMPTY");

            }

            T data = q.front();
            q.pop();

            return data;

        }

        bool empty()
        {
            return q.empty();
        }

        size_t size()
        {
            return q.size();
        }


        void fill_stream(int* input_array, int size)
        {
            // array uncheck 

            int simd_check = 0;
            T channel_batch;
            for(int i = 0 ; i < size; i++)
            {
                int pixel = input_array[i];
                channel_batch.data[simd_check] = pixel;
                simd_check++;

                if(simd_check==SIMD)
                {
                    simd_check = 0;
                    q.push(channel_batch);
                }
            }
        }

};