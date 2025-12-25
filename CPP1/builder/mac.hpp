
#pragma once
//N - Number of MAC units
// AT - Accumulator data type
// T1 - First operand data type
// T2 - Second operand data type

template<unsigned int N, typename AT, typename T1, typename T2>

AT mac(const AT& acc, const T1& in1, const T2& in2)
{
    AT res = acc;

    for(unsigned int i = 0; i < N; i++)
    {
        res += in1[i] * in2[i];
    }

    return res;
}