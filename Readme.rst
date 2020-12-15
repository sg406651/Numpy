***************
 numpy library
***************

Numpy is default python library to work with n-dimensional array.
Almost all popular python libraries for manipulate n-dimensional array in python
allow to export data an numpy array or implement same interface.

Numpy like libraries:

* Zarr – https://zarr.readthedocs.io/en/stable/
* Dask arrays – https://dask.org/

Sample of libraries with numpy integration:

* matplotlib – plotting utility for python https://matplotlib.org/
* Pillow – image processing libraries https://pillow.readthedocs.io/en/stable/
* scipy – numerical effective routines https://www.scipy.org/scipylib/index.html
* scikit-learn – base machine learning methods https://scikit-learn.org/stable/#
* cython – optimising static compiler for python https://cython.org/
* Pandas (next classes) – dataframes for python https://pandas.pydata.org/

Basics
######
In comparison to python list numpy arrays have fixed size and type (called ``dtype``).

.. code-block::

    >>> np.empty((3, 3), dtype=np.uint16)
    array([[    0,     0,     0],
           [36864, 31715, 60221],
           [ 2046, 57344,     2]], dtype=uint16)


Creating arrays
###############

* ``empty(shape, dtype=float, order='C')`` – array of given shape without data initialization
* ``zeros(shape, dtype=float, order='C')`` – array of given shape filled with zeros
* ``ones(shape, dtype=float, order='C')`` – array of given shape filled with ones
* ``arange([start, ]stop, [step, ]dtype=None)`` - similar to ``range`` but returns ``ndarray`` instance
* ``linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`` – evenly spaced numbers over a specified interval

Slicing
#######



Exercises
#########

Exercise 1
~~~~~~~~~~
