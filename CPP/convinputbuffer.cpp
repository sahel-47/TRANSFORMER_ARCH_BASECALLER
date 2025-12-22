#include<iostream>
#include<queue>
#include<stdexcept>
#include<pybind11/numpy.h>
#include<pybind11/pybind11.h>

// Class to mimic hls stream
namespace py = pybind11;

template<int SIMD>
struct SimdPacket
{
    int data[SIMD];
};

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


template<
        unsigned int KernelDim,
        unsigned int IFMChannels,
        // unsigned int inputPrecision,
        unsigned int IPDim,
        unsigned int OPDim,
        unsigned int SIMD,
        unsigned int Stride
        >
class ConvInputGen
{

    private:
        static constexpr unsigned int multiplying_factor = IFMChannels/SIMD;
        static constexpr int numberBlocks = KernelDim/Stride + 1;

        SimdPacket<SIMD> input_buffer[numberBlocks][Stride * IPDim * multiplying_factor];

        const unsigned int cycles_read_block = Stride * IPDim * multiplying_factor;
        const unsigned int cycles_write_block = OPDim * KernelDim * KernelDim * multiplying_factor;
        const unsigned int max_cycles = std::max(cycles_read_block, cycles_write_block);
        const unsigned int fullIter = (KernelDim * IPDim * multiplying_factor) + (OPDim * max_cycles);

        unsigned int counter = 0;
        unsigned int current_block_write = 0;
        unsigned int current_line = 0;
        unsigned int read_block = 0;
        unsigned int inp = 0, ofm_x = 0, ofm_y = 0, k_x =0, k_y = 0, count_simd = 0;


    public:
        hls_stream<SimdPacket<SIMD>, SIMD> &in;
        hls_stream<SimdPacket<SIMD>, SIMD> &out;
        ConvInputGen(hls_stream<SimdPacket<SIMD>, SIMD>& input, hls_stream<SimdPacket<SIMD>, SIMD>& output)
        :in{input},
         out{output}
         {

         }
        void generator()
        {
            for(int i = 0; i < fullIter; i++)
            {
                if(inp<KernelDim * IPDim * multiplying_factor)
                {
                    SimdPacket<SIMD> inElem;
                    inElem = in.read();
                    input_buffer[current_block_write][current_line] = inElem;
                    current_line++;
                    inp++;
                    if(current_line == Stride * IPDim * multiplying_factor)
                    {
                        current_line = 0;
                        current_block_write++;
                    

                        if(current_block_write == numberBlocks)
                        {
                            current_block_write = 0;
                        }
                        read_block++;
                        counter = 0;

                    }
                    
                }

                else
                {
                    if(counter < cycles_write_block-1)
                    {
                        unsigned int current_block_read = (current_block_write + 1 + k_y/Stride);

                        if(current_block_read >= numberBlocks)
                        {
                            current_block_read -= numberBlocks;

                        }

                        unsigned int current_line_in_block = ((k_y%Stride)*IPDim + ofm_x*Stride+k_x)*multiplying_factor+count_simd;
                        SimdPacket<SIMD> outElem;
                        outElem = input_buffer[current_block_read][current_line_in_block];
                        out.write(outElem);
                        count_simd++;

                        if(count_simd == multiplying_factor)
                        {
                            count_simd = 0;
                            k_x++;
                            if(k_x==KernelDim)
                            {
                                k_x = 0;
                                k_y++;
                                if(k_y==KernelDim)
                                {
                                    k_y = 0;
                                    ofm_x++;
                                    if(ofm_x == OPDim)
                                    {
                                        ofm_x = 0;
                                        ofm_y++;
                                        if(ofm_y == OPDim)
                                        {
                                            ofm_y = 0;
                                            inp = 0;
                                        }
                                    }
                                }
                            }
                        }



                    }

                    if((counter < cycles_read_block-1) && (read_block < IPDim/Stride))
                    {
                        SimdPacket<SIMD> inElem;
                        inElem = in.read();
                        input_buffer[current_block_write][current_line] = inElem;
                        current_line++;

                        if(current_line == Stride * IPDim * multiplying_factor)
                        {
                            current_line = 0;
                            read_block++;
                            current_block_write++;
                            if(current_block_write == numberBlocks)
                            {
                                current_block_write = 0;
                                
                            }
                        }


                    }
                    counter++;
                    if(counter  == max_cycles-1)
                    {
                        counter = 0;
                    }
                }
            }
        }

};


template<
        unsigned int KernelDim,
        unsigned int IFMChannels,
        // unsigned int inputPrecision,
        unsigned int IPDim,
        unsigned int OPDim,
        unsigned int SIMD,
        unsigned int Stride
        >

class ConvInputGen_advanced
{

    private:
        static const unsigned int multiplying_factor = IFMChannels/SIMD;
        static const unsigned int number_blocks = KernelDim + Stride;

        SimdPacket<SIMD> input_buffer[number_blocks][IPDim * multiplying_factor];

        const unsigned int cycles_read_block = Stride * IPDim * multiplying_factor;
        const unsigned int cycles_write_block = OPDim * KernelDim * KernelDim * multiplying_factor;
        const unsigned int max_cycles = std::max(cycles_write_block, cycles_read_block);
        const unsigned int fullIter = KernelDim * IPDim * multiplying_factor + OPDim * max_cycles;
        const unsigned int initial_time = KernelDim * IPDim * multiplying_factor;

        unsigned int counter_internal_block = 0;
        unsigned int read_block = 0;
        unsigned int current_write_block = 0;
        unsigned int current_line = 0;
        unsigned int inp = 0, ofm_x= 0, ofm_y = 0, k_x = 0, k_y = 0, count_simd = 0;
    
        unsigned int ceil_val = number_blocks;
        unsigned int floor_val = 0;

    public:
        hls_stream<SimdPacket<SIMD>,SIMD>& in;
        hls_stream<SimdPacket<SIMD>,SIMD>& out;

        ConvInputGen_advanced(hls_stream<SimdPacket<SIMD>,SIMD>& input , hls_stream<SimdPacket<SIMD>,SIMD>& output)
        : in{input},
          out{output}
        {

        }


        void generator()
        {

            for(int i = 0; i < fullIter; i++)
            {
                if(inp < initial_time)
                {
                     SimdPacket<SIMD> inElem;
                     inElem = in.read();
                     input_buffer[current_write_block][current_line] = inElem;
                     current_line++;
                     inp++;
                     if(current_line == IPDim * multiplying_factor)
                     {
                        current_line = 0;
                        current_write_block++;
                        read_block++;
                        if(current_write_block == number_blocks)
                        {
                            current_write_block = 0;
                        }

                        counter_internal_block = 0;
                     }

                }
                else
                {
                    if((counter_internal_block < cycles_write_block-1) || read_block == IPDim)
                    {
                        unsigned int current_read_block = (ofm_y*Stride + k_y);

                        if(current_read_block >= ceil_val)
                        {
                            floor_val += number_blocks;
                            ceil_val += number_blocks;
                        }
                        else if(current_read_block < floor_val) { // <--- ADDED THIS -- NEEDED
                             floor_val -= number_blocks;
                             ceil_val -= number_blocks;
                        }

                        current_read_block -= floor_val; //does % operation

                        SimdPacket<SIMD> outElem;
                        unsigned int current_writeline = (ofm_x*Stride+k_x)*multiplying_factor + count_simd;

                        outElem = input_buffer[current_read_block][current_writeline];
                        out.write(outElem);
                        count_simd++;

                        if(count_simd == multiplying_factor)
                        {
                            count_simd = 0;
                            k_x++;
                            if(k_x == KernelDim)
                            {
                                k_x = 0;
                                k_y++;
                                if(k_y == KernelDim)
                                {
                                    k_y = 0;
                                    ofm_x++;
                                    if(ofm_x == OPDim)
                                    {
                                        ofm_x = 0;
                                        ofm_y++;
                                        if(ofm_y == OPDim)
                                        {
                                            ofm_y = 0;
                                            // inp = 0;

                                           return; // Has problem with reset.... keeps reading it for some reason
                                        }
                                    }
                                }
                            }
                        }


                    }

                    if((counter_internal_block < cycles_read_block-1) && (read_block < IPDim))
                    {
                        SimdPacket<SIMD> inElem;

                        inElem = in.read();
                        input_buffer[current_write_block][current_line] = inElem;
                        current_line++;
                        if(current_line == IPDim * multiplying_factor)
                        {
                            current_line = 0;
                            current_write_block++;
                            read_block++;
                            if(current_write_block == number_blocks)
                            {
                                current_write_block = 0;
                            }
                        }
                    }
                    counter_internal_block++;
                    if(counter_internal_block == (max_cycles-1))
                    {
                        counter_internal_block = 0;
                    }
                }
            }




        }



};



template<
        unsigned int KernelDim,
        unsigned int IFMChannels,
        // unsigned int inputPrecision,
        unsigned int IPDim,
        unsigned int OPDim,
        unsigned int SIMD,
        unsigned int Stride
        >

class ConvInputGen_1D
{
    private:
        static constexpr unsigned int SIMD_MULTIPLE = IFMChannels/SIMD;
        static constexpr unsigned int BUFFER_SIZE = (KernelDim - 1) * SIMD_MULTIPLE;
        const unsigned int OUTPUT_SIZE = OPDim * KernelDim * SIMD_MULTIPLE;
        const unsigned int INPUT_SIZE = IPDim * SIMD_MULTIPLE;
        const unsigned int INITIAL_OCNT = BUFFER_SIZE + (Stride - 1);

        const unsigned int WINDOW_SIZE = KernelDim * SIMD_MULTIPLE;


        SimdPacket<SIMD> input_buffer[BUFFER_SIZE];


    public:
        hls_stream<SimdPacket<SIMD>,SIMD>& in;
        hls_stream<SimdPacket<SIMD>,SIMD>& out;

        ConvInputGen_1D(hls_stream<SimdPacket<SIMD>,SIMD>& input, hls_stream<SimdPacket<SIMD>,SIMD>& output)
        :in{input},
         out{output}
         {

         }

         void generator()
         {



            unsigned int rp = 0;
            unsigned int wp = 0;
            unsigned int inp = 0;
            unsigned int offset = 0;
            unsigned int ocnt = (INITIAL_OCNT < WINDOW_SIZE ? INITIAL_OCNT:-1);

            for(int i = 0; i < OUTPUT_SIZE+1; i++)
            {
                bool const re  = i > 0;
                bool const we = ((i < WINDOW_SIZE) || (ocnt < SIMD_MULTIPLE * Stride));

                if(re)
                {
                    out.write(input_buffer[rp]);

                    if(++offset == WINDOW_SIZE)
                    {
                        rp += 1 + (Stride-1) * SIMD_MULTIPLE;
                        offset = 0;

                        if(rp >= BUFFER_SIZE)
                        {
                            rp -= BUFFER_SIZE;
                        }
                    }
                    else
                    {
                        if(++rp >= BUFFER_SIZE)
                        {
                            rp -= BUFFER_SIZE;
                        }
                    
                    }

                    if(++ocnt == WINDOW_SIZE)
                    {
                        ocnt = 0;
                    }
                    
                }

                if(we)
                {
                    if(++inp <= INPUT_SIZE)
                    {
                        input_buffer[wp] = in.read();
                        if(++wp >= BUFFER_SIZE)
                        {
                            wp = 0;
                        }
                    }
                }
            }

         }


        
};


PYBIND11_MODULE(ConvInpGen, m , py::mod_gil_not_used())
{
    py::class_<SimdPacket<4>>(m, "SimdPacket4")
    .def(py::init<>())
    .def_property("data",
    [](SimdPacket<4> &p){
        return py::make_tuple(p.data[0], p.data[1], p.data[2], p.data[3]);
    }, nullptr
    );
    

    py::class_<hls_stream<SimdPacket<4>,4>>(m, "hls_stream")
    .def(py::init<>())
    .def("empty", &hls_stream<SimdPacket<4>,4>::empty)
    .def("size", &hls_stream<SimdPacket<4>,4>::size)
    .def("read", &hls_stream<SimdPacket<4>,4>::read)
    .def("write", &hls_stream<SimdPacket<4>,4>::write)
    .def("fill_stream",([](hls_stream<SimdPacket<4>,4>& stream, py::buffer input)
                        {
                            py::buffer_info info = input.request();

                            int* ptr = static_cast<int*>(info.ptr);
                            int size = static_cast<int>(info.size);

                            stream.fill_stream(ptr, size);
                        }
                       )
        );


    using ConvInputGen3x3_8_10_10_4_1 = ConvInputGen<3,8,10,10,4,1>;

    py::class_<ConvInputGen3x3_8_10_10_4_1>(m, "ConvInputGen")
    .def(py::init<hls_stream<SimdPacket<4>,4>& , hls_stream<SimdPacket<4>,4>&>())
    .def("generator", &ConvInputGen3x3_8_10_10_4_1::generator);

    using ConvInputGen3x3_8_10_4_4_2 = ConvInputGen_advanced<3,8,10,4,4,2>;

    py::class_<ConvInputGen3x3_8_10_4_4_2>(m, "ConvInputGen_advanced")
    .def(py::init<hls_stream<SimdPacket<4>,4>& , hls_stream<SimdPacket<4>,4>&>())
    .def("generator", &ConvInputGen3x3_8_10_4_4_2::generator);


    
    using ConvInputGen3_8_10_8_4_1 =  ConvInputGen_1D<3, 8, 10, 4, 4, 2>;


    py::class_<ConvInputGen3_8_10_8_4_1>(m, "ConvInputGen_1D")
    .def(py::init<hls_stream<SimdPacket<4>,4>& , hls_stream<SimdPacket<4>,4>&>())
    .def("generator", &ConvInputGen3_8_10_8_4_1::generator);


}






