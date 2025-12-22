#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
#include<cmath>
#include"RoPE.hpp"
#include<string>


// void load_tensor_from_file(const std::string& filename, float* buffer, int size)
// {
//     std::ifstream file(filename);

//      if (!file.is_open()) {
//         std::cerr << "Error: Could not open file " << filename << std::endl;
//         exit(1);
//     }

//     for (int i = 0; i < size; i++)
//     {
//         if(!(file >> buffer[i]))
//         {
//             std::cerr << "Error reading value at index " << i << " from " << filename << std::endl;
//             exit(1);

//         }
//     }

//     file.close();
//     std::cout << "Loaded" <<size <<" values from " << filename <<std::endl;
// }


int main()
{
    return 0;
}
// int main()
// {
//     const int seq_len = 10;
//     const int dim = 16;
//     const int total_elements = seq_len * dim;

//     float* input_data = new float[total_elements];
//     float* golden_results = new float[total_elements];

//     std::cout <<"------LOADING DATA-------"<<std::endl;
//     load_tensor_from_file("q_input.txt", input_data, total_elements);
//     load_tensor_from_file("q_result_golden.txt", golden_results, total_elements);


//     std::cout<<std::endl;

//     RoPE rope_layer;
//     // rope_layer.compute_RoPE(input_data);

//        std::cout << "\n--- Verification ---" << std::endl;
    
//     int errors = 0;
//     float max_error = 0.0f;
//     int half_dim = dim / 2;

//     // Iterate over Time Steps
//     for (int t = 0; t < seq_len; t++) {
//         // Iterate over Feature Pairs
//         for (int j = 0; j < half_dim; j++) {
            
//             std::cout << input_data[t * dim+j]<<" ";
//         }
//     }
//     // Cleanup
//     delete[] input_data;
//     delete[] golden_results;
    
//     return 0;

// }