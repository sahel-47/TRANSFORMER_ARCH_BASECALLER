#include"weight_format.hpp"
#include"stream.hpp"
#include"mac.hpp"

struct Identity
{
    template<typename T>
    const T& operator()(const T& object) const
    {
        return object;
    }
};

template<
         unsigned int  MATRIX_W,
         unsigned int MATRIX_H,
         unsigned int SIMD,
         unsigned int NES,
         
         typename TI,
         typename TO, 
         typename TW,
         typename TA ,// ADD LATER
         typename srcInterpreter = Identity,
         typename destInterpreter = Identity,
         typename weightInterpreter = Identity
        >

class MatrixInnerEngine
{
    public:
        hls_stream<TI,SIMD>& in_stream; //HERE that 1 doesnt matter ... its only for manually filling the stream
        hls_stream<TO,2>& out_stream; //HARDCODING TO 2 change later accourdingly // SHOULD BE NES IG
        unsigned int rows;
        TW& weightMatrix;

        static constexpr unsigned int SF = MATRIX_W/SIMD;
        static constexpr unsigned int NF = MATRIX_H/NES;
        static constexpr unsigned int TOTAL_TILES = NF * SF;

        TI input_buffer[SF];

        TA acc_buffer[NES];
        TO outElem;




        MatrixInnerEngine(hls_stream<TI, SIMD>& in, hls_stream<TO, 2>& out, TW& WM, unsigned int Rows = 1)//HARDCODING TO 2 change later accourdingly // SHOULD BE NES IG
        :in_stream{in},
         out_stream{out},
         rows{Rows},
         weightMatrix{WM}
        {

        }

        void Multiply()
        {
            unsigned int sf = 0;
            unsigned int nf = 0;
            unsigned int tile = 0;

            for(int i = 0; i < rows*TOTAL_TILES; i++)
            {
                TI inElem;
                if(nf == 0)
                {
                    inElem = in_stream.read();
                    input_buffer[sf] = inElem;
                }
                else
                {
                    inElem = input_buffer[sf];
                }

                if(sf==0)
                {
                    for(int ne = 0; ne < NES; ne++)
                    {
                       acc_buffer[ne] = 0;
                    }
                }

                  const auto& weights = weightMatrix.weight_idx(tile);

                  for(int ne = 0; ne < NES; ne++)
                  {
                     const auto mac_input = srcInterpreter()(inElem);
                     acc_buffer[ne] = mac<SIMD, TA, TI>(acc_buffer[ne],mac_input, weights[ne]); 
                  }


                ++tile;
                ++sf;
                if(sf == SF)
                {
                    TO outElem;
                    for(int ne = 0; ne < NES; ne++)
                    {
                        outElem[ne] = acc_buffer[ne];

                    }

                    outElem = destInterpreter()(outElem);

                    out_stream.write(outElem);

                    sf = 0;
                    ++nf;
                    if(nf == NF)
                    {
                        nf = 0;
                        tile = 0;
                    }
                }
            }

        }
};















































/*
In your original code, you incremented tile manually (++tile). This works, but using tile = nf * SF + sf at the start of the loop is more robust in hardware because it doesn't rely on the history of the previous cycle—it's stateless logic, which is easier for the synthesizer to optimize.
*/

