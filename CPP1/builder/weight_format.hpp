#include<iostream>
#include<array>
#pragma once
/*

 a (16x16) divided into 4x4 (SIMD = 4, NES = 4)

# #*************************16***************************#
# || || || || + || || || || + || || || || + || || || || +   (1)
* || || || || + || || || || + || || || || + || || || || +   (2)
* || || || || + || || || || + || || || || + || || || || +   (3)
* || || || || + || || || || + || || || || + || || || || +   (4)
* &  &  &  &  + &  &  &  &  + &  &  &  &  + &  &  &  &  +

* || || || || + || || || || + || || || || + || || || || +   (5)
* || || || || + || || || || + || || || || + || || || || +   (6)
* || || || || + || || || || + || || || || + || || || || +   (7)
1 || || || || + || || || || + || || || || + || || || || +   (8)
6 &  &  &  &  + &  &  &  &  + &  &  &  &  + &  &  &  &  +

* || || || || + || || || || + || || || || + || || || || +   (9)
* || || || || + || || || || + || || || || + || || || || +   (10)
* || || || || + || || || || + || || || || + || || || || +   (11)
* || || || || + || || || || + || || || || + || || || || +   (12)
* &  &  &  &  + &  &  &  &  + &  &  &  &  + &  &  &  &  +

* || || || || + || || || || + || || || || + || || || || +   (13)
* || || || || + || || || || + || || || || + || || || || +   (14)
* || || || || + || || || || + || || || || + || || || || +   (15)
# || || || || + || || || || + || || || || + || || || || +   (16)

*/


template<typename T, unsigned int SIMD>
struct ElementsPacked
{
    T data[SIMD];

    T operator[](const int idx) const
    {
        return (*this).data[idx];
    }
    T& operator[](const int idx) 
    {
        return (*this).data[idx];
    }

};


template<typename T,unsigned int SIMD>
struct WeightsPacked
{
    T data[SIMD];

    T operator[](const int idx) const 
    {
        return (*this).data[idx];
    }
    
    T& operator[](const int idx) 
    {
        return (*this).data[idx];
    }
};

//Didnt use width because all floats are 32bit 
template<unsigned int SIMD, unsigned int NES, unsigned TILES>

class FloatingPackedWeights
{

    public:
        WeightsPacked<int, SIMD> m_weights[NES][TILES]; // ADDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD FLOAT BACKKKK

        // for tomorrow: TILES is holding weights for more than one neuron btw

    
    class Tidx
    {
        private:
            FloatingPackedWeights const &m_weight_ref;
            unsigned int const m_idx;

        public: 
            Tidx(FloatingPackedWeights const &weight_ref, unsigned int const idx)
            : m_weight_ref{weight_ref},
              m_idx{idx}
            {

            }

        std::array<int,SIMD> operator[](unsigned const int pe) const
        {
            std::array<int,SIMD> ret;  // ADDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD FLOAT BACKKKK

            for(int i = 0; i < SIMD; i++)
            {
                int const temp = m_weight_ref.m_weights[pe][m_idx].data[i];     // ADDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD FLOAT BACKKKK
                ret[i] = temp;
            }

            return ret;
        }

    
    
    };

    Tidx weight_idx(unsigned const int tile) const 
    {
        return Tidx(*this, tile);
    }

    void load_weights(const int* input_array)
    {
        int total = SIMD * NES * TILES;

        int* wt_ptr = reinterpret_cast<int*>(m_weights);

        for(int i = 0; i < total;i++)
        {
            wt_ptr[i] = input_array[i];
        }
    }

};




// int main()
// {
//     FloatingPackedWeights<4, 6, 24> array1;
    
//     auto const& w = array1.weight_idx(2);

//     std::cout << "The value of w[3] is "<< w[3][1] <<std::endl;

//     return 0;
// }


