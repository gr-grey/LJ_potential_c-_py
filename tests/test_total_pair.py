"""
	Testing the lennard Jones c++ python binding module
"""
import pytest

import numpy as np

import LJ_potential as lj

import Monte_Carlo


def test_total_pair():
	
	e1 = total_potential_energy(coordinate, box_length)
	e2 = lj.total_pair_e(coordinate, box_length)

	assert np.allclose(e1,e2)


