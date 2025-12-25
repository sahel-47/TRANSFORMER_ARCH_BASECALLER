#include"mac.hpp"
#include"MatrixInnerEngine.hpp"
#include"stream.hpp"
#include"weight_format.hpp"
#include<nanobind/nanobind.h>
#include<nanobind/ndarray.h>

namespace nb = nanobind;


using ElementsPacked_int_4 = ElementsPacked<int,4>;
using ElementsPacked_int_2 = ElementsPacked<int, 2>;
using WeightsPacked_int_4 = WeightsPacked<int, 4>;
using input_fifo_stream = hls_stream<ElementsPacked_int_4, 4>;
using output_fifo_stream = hls_stream<ElementsPacked_int_2, 2>;
using WEIGHTS =  FloatingPackedWeights<4,2,32>;
using Matrix_Engine = MatrixInnerEngine<16, 16, 4, 2, ElementsPacked_int_4, ElementsPacked_int_2,WEIGHTS, int>; //16x16

NB_MODULE(MATRIX_INNER_ENGINE, m)
{
    nb::class_<ElementsPacked_int_4>(m, "ElementsPacked4")
    .def(nb::init<>())
    .def("__getitem__",[](ElementsPacked_int_4& self, int i){

            return self[i];
        }
    )
    .def("__setitem__",[](ElementsPacked_int_4& self, int i, int val){


        self[i] = val;
        }
    )
    .def("view",[](ElementsPacked_int_4& self){

        return nb::ndarray<nb::numpy, int>(
            self.data,
            {4},
            nb::cast(&self)
        );
        }
    );

     nb::class_<ElementsPacked_int_2>(m, "ElementsPacked2")
    .def(nb::init<>())
    .def("__getitem__",[](ElementsPacked_int_2& self, int i){

            return self[i];
        }
    )
    .def("__setitem__",[](ElementsPacked_int_2& self, int i, int val){


        self[i] = val;
        }
    )
    .def("view",[](ElementsPacked_int_2& self){

        return nb::ndarray<nb::numpy, int>(
            self.data,
            {2},
            nb::cast(&self)
        );
        }
    );

    nb::class_<WeightsPacked_int_4>(m , "WeightsPacked")
    .def(nb::init<>())
    .def("__getitem__",[](WeightsPacked_int_4& self, int i){
                return self[i];
            }
        )
    .def("__setitem__",[](WeightsPacked_int_4& self, int i, int val){

                self[i] = val;
            }

        )

    .def("view",[](WeightsPacked_int_4& self){
                return nb::ndarray<nb::numpy, int>(
                    self.data,
                    {4},
                    nb::cast(&self)
                );
            }

        );

    nb::class_<WEIGHTS>(m, "PACKED_WEIGHTS")
    .def(nb::init<>())
    // .def("weight_idx",&WEIGHTS::weight_idx) // not required to expose to py
    .def("load_weights",[](WEIGHTS& w,nb::ndarray<>& inp){

        int* ptr = static_cast<int*>(inp.data());

        w.load_weights(ptr);
 
    });

    // nb::class_<WEIGHTS::Tidx>(m, "Tidx")

    nb::class_<input_fifo_stream>(m, "stream_fifo4")
    .def(nb::init<>())
    .def("write", &input_fifo_stream::write)
    .def("read", &input_fifo_stream::read)
    .def("empty", &input_fifo_stream::empty)
    .def("size", &input_fifo_stream::size)
    .def("fill_stream",[](input_fifo_stream& fifo,const nb::ndarray<>& stream){

                int* ptr = static_cast<int*>(stream.data());
                int size = static_cast<int>(stream.size());
                std::cout<<"SIZE IS: "<<size;

                fifo.fill_stream(ptr, size);

            }

        );

    nb::class_<output_fifo_stream>(m, "stream_fifo2")
    .def(nb::init<>())
    .def("write", &output_fifo_stream::write)
    .def("read", &output_fifo_stream::read)
    .def("empty", &output_fifo_stream::empty)
    .def("size", &output_fifo_stream::size)
    .def("fill_stream",[](output_fifo_stream& fifo,const nb::ndarray<>& stream){

                int* ptr = static_cast<int*>(stream.data());
                int size = static_cast<int>(stream.size());

                fifo.fill_stream(ptr, size);

            }

        );

    nb::class_<Matrix_Engine>(m, "MatrixInnerEngine")
    .def(nb::init<input_fifo_stream&, output_fifo_stream&, WEIGHTS&, int>())
    .def("Multipy", &Matrix_Engine::Multiply);

}