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
* scikit-learn – base machine learning methods https://scikit-learn.org/stable/
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


Types
~~~~~
Numpy in basic uses primitive types.

=============================== ================================================
Type                            Description
=============================== ================================================
``bool``                        ``True`` or ``False`` stored as single bit
``int``                         signed 32 or 64 bits int
``int8/16/32/64``               respectively 8/16/32/64 signed integer
``uint8/16/32/64``              respectively 8/16/32/64 unsigned integer
``float16``                     half precision float
``float32``                     single precision float
``float64`` or ``float``        double precision float
``complex64``                   complex number represented as two ``float32``
``complex128`` or ``complex``   complex number represented as two ``float64``
``object``                      arbitrary python object, possible but rare used
=============================== ================================================

Please remember about bit overflow.

Creating arrays
###############

* ``array(object,dtype=None, *, copy=True, order='K', subok=False, ndmin=0)`` - create numpy array from object
* ``empty(shape, dtype=float, order='C')`` – array of given shape without data initialization
* ``zeros(shape, dtype=float, order='C')`` – array of given shape filled with zeros
* ``ones(shape, dtype=float, order='C')`` – array of given shape filled with ones
* ``np.full(shape, fill_value, dtype=None, order='C')`` – array of given shape filled with ``fill_value``
* ``arange([start, ]stop, [step, ]dtype=None)`` - similar to ``range`` but returns ``ndarray`` instance
* ``linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`` – evenly spaced numbers over a specified interval

Like list Numpy array support slicing.

.. code-block:: python

    array = np.zeros(10, dtype=np.uint8)
    array[2] = 1
    array[3] = 2
    array[4:7] = [3, 4, 5]
    print(array)

Its also support assigment single value to multiple positions

.. code-block:: python

    array = np.zeros(10, dtype=np.uint8)
    array[2:7] = 1
    print(array)

Mathematical operations
#######################

By default mathematical operation (like ``+``, ``-``, ``*`` and ``**``) are done positional.
Full list could be found here: https://numpy.org/doc/stable/reference/routines.math.html

Operation like matrix multiplication are implemented in ``numpy.linalg`` module
https://numpy.org/doc/stable/reference/routines.linalg.html

Exercise 1
~~~~~~~~~~

Fix array creation in bellow code to satisfy all ``assert``.

.. code-block:: python

    arr1 = []
    arr2 = []
    assert len(arr1) == 10
    assert len(arr2) == 10
    assert np.all(arr1 == 100)
    assert np.all(arr1 == 156)
    assert np.all(arr1 + arr2 == 0)

Array properties
################

* ``shape`` – tuple with shape of array
* ``size`` – size of arrays, equal to multiplication ``shape`` elements
* ``dtype`` – data type used for storage
* ``T`` – transpose of array

Array manipulation
##################
Numpy has multiple functions for manipulate shape of array:

* ``reshape`` – new shape have to had same number of elements.
* ``squeeze`` – remove single dimensions axes
* ``flatten`` - single dimension copy of array
* ``ravel`` – contiguous flattened array

* ``astype`` – allow to change array dtpe

For more read https://numpy.org/doc/stable/reference/routines.array-manipulation.html

Slicing
#######

Numpy arrays allow for slicing in multiple dimension. For example:

.. code-block:: python

    >>> import numpy as np
    >>> arr = np.zeros((4, 4), dtype=np.uin16)
    >>> arr
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=uint16)
    >>> arr[1:-1, 1:-1] = 1
    >>> arr
    array([[0, 0, 0, 0],
           [0, 1, 1, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 0]], dtype=uint16)
    >>> arr[:2, :2]
    array([[0, 0],
           [0, 1]], dtype=uint16)

Remember that ``arr[:2, :2]`` is faster than ``arr[:2][:2]``

IO operations
#############

* ``loadtxt`` – load array from text file
* ``savetxt`` – save array to text file
* ``load`` – load data from binary file (``.npy`` or ``.npz``)
* ``save`` – save array to ``.npy`` binary file
* ``savez`` and ``savez_compressed`` – save multiple arrays in uncompressed or compressed binary file.

More: https://numpy.org/doc/stable/reference/routines.io.html

Statistics
##########

* ``min``/``amin``
* ``max``/``amax``
* ``median``
* ``std``
* ``var``

More: https://numpy.org/doc/stable/reference/routines.statistics.html

Many of numpy functions have ``axis`` argument which allows to perform such operation along given axis.

.. code-block:: python

    >>> import numpy as np
    >>> arr = np.random.uniform(size=(10, 20))
    >>> np.std(arr)
    0.289538402318112
    >>> np.std(arr, axis=1)
    array([0.28590859, 0.29832191, 0.29218063, 0.29722575, 0.26979703,
           0.24772888, 0.28394164, 0.24025019, 0.29967281, 0.32325727])

Exercise 2
~~~~~~~~~~

Load data from ``data/sample.csv`` calculate mean, median and std for each column separately.
Write it using numpy and without it (also without pandas etc).

For each method measure time of it execution (using ``%time`` magic or ``time()`` function from ``time`` module)

Masking
#######

Comparison of two proper sized numpy array or comparison numpy array to number produces array of ``bool``.

.. code-block:: python

    >>> np.arange(9) > 4
    array([False, False, False, False, False,  True,  True,  True,  True])

So it cannot be used in ``if`` without conversion too bool using ``np.all`` or ``np.any``. So instead

.. code-block:: python

    if arr1 == arr2:
        do_something()

do:

.. code-block:: python

    if np.all(arr1 == arr2):
        do_something()

or best:

.. code-block:: python

    if np.array_equal(arr1, arr2):
        do_something()

Boolean masks could be used for indexing existing array.
Lest randomize 1000 variables from ``N(2, 1)`` then change all values bellow 0 to 0.

.. code-block:: python

    >>> arr = np.random.normal(2, 1, size=1000)
    >>> np.sum(arr < 0)
    27
    >>> arr[arr < 0] = 0
    >>> np.sum(arr < 0)
    0

Exercise 3
~~~~~~~~~~
Load data from ``data/ex3_data.npy`` and filter out rows with ``nan`` values.
Report how many rows are dropped during filtering. Globally or per column.

Exercise 3
~~~~~~~~~~
Using ``loadtxt`` form ``numpy`` load data from ``data/iris.csv``. Skip header and name column.
For each column calculate `mean`, `median` and `std`

Exercise 4
~~~~~~~~~~
Load data from ``data/sample_treated.npz``. Assume that each row of ``outputs`` array contains information
about size of some structure traced in time. We would like to know which object grows at least two times during
observation.
