#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <cmath>
#include <math.h>


namespace py = pybind11;

double radius(py::array_t<double> A, size_t i, size_t j, double box_length)
{
	py::buffer_info A_info = A.request();

	double r = 0.0;

	size_t num = A_info.shape[0];

	size_t row_stride = A_info.strides[0] / sizeof(double);
	size_t col_stride = A_info.strides[1] / sizeof(double);
	
	const double * A_data = static_cast<double *>(A_info.ptr);

	double rx, ry, rz;
	// Here all the i should be i*row_stride, but to row_stride is 1 so I'm omiting it in ry and rz.
	rx = A_data[i * row_stride] - A_data[j * row_stride];
	ry = A_data[i + col_stride] - A_data[j +  col_stride];
	rz = A_data[i + 2 * col_stride] - A_data[j + 2 * col_stride];

	rx = rx - round(rx/box_length) * box_length;
	ry = ry - round(ry/box_length) * box_length;
	rz = rz - round(rz/box_length) * box_length;				
			
	r = sqrt(pow(rx,2) + pow(ry,2) + pow(rz,2));

	return r;

}



double single_LJ(double r)
{
	
	return 4.0 * (1/pow(r,12) - 1/pow(r,6));

}


double total_pair_e(py::array_t<double> A, double box_length, double cutoff)
{
	py::buffer_info A_info = A.request();

	double e = 0.0;
	double r = 0.0;


	size_t num = A_info.shape[0];


	for (size_t i = 0; i < num; i++)
	{
		for (size_t j = i + 1; j < num; j++)
			{
				
				r = radius(A, i, j, box_length);


				if(r < cutoff)
					e += single_LJ(r); 
			}


	}		
	return e;
}

double get_mol_energy(py::array_t<double> A, size_t i,double box_length,double cutoff)
{
	double e_i = 0.0;
	double r;
	
	py::buffer_info A_info = A.request();
	size_t num = A_info.shape[0];

	for (size_t j = 0; j < num; j++)
	{	if (j != i)		
		{	r = radius(A, i, j, box_length);

			if (r < cutoff) 	
			e_i += single_LJ(r);				
		}	
	
	}
	return e_i;

}



PYBIND11_PLUGIN(LJ_potential)
{
	py::module m("LJ_potential", "Lennard Jones Potential");
	m.def("total_pair_e", &total_pair_e, "Input coordinate set and box length");
	m.def("get_mol_energy", &get_mol_energy, "Input coordinate, which particle, boxlegth and cutoff");	
	
	return m.ptr();

}

