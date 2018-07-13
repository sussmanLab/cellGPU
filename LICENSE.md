# Licensing information {#license}

CellGPU is released under the MIT license

Copyright (c) 2016 - 2018 Daniel M. Sussman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

All publications and presentations based on this software will acknowledge its use according to the terms posted at the time of submission on the code homepage.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

# External code used

Additionally, some files and functionality draw from existing open-source code, as described below.

(1) Two files (gpuarray.h and indexer.h) are largely based on parts of the HOOMD-blue project, released
under the BSD 3-Clause License.
https://glotzerlab.engin.umich.edu/hoomd-blue

HOOMD-blue Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer both in the code and prominently in any materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may be used to enorse or promote products derived from this software without specific prior written permission

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

(2) The file HilbertSort.h calls, but does not modify, code released under the GNU LGPL licence by John
Burkardt. The "HilberSort.h" wrapper is used by the program to call the functions defined in the
library, and the source code from which the library can be built is in the "hilbert_sort.hpp" and
"hilbert_sort.cpp" files. Thus, this repository contains everything a user would need to relink the
application with a different version of Burkardts LGPL source code. As such CellGPU can be distributed
under a non-(L)GPL license. Credit for this library, of course, goes to John Burkardt:
https://people.sc.fsu.edu/~jburkardt/cpp_src/hilbert_curve/hilbert_curve.html

(3) eigenMatrixInterface.h and .cpp interfaces with the Eigen library. Eigen is Free Software,
available from eigen.tuxfamily.org/. It is licensed under the MPL2. See https://www.mozilla.org/en-US/MPL/2.0/ for more details.

(4) Finally, the cellGPU logo was made by using the ``Lincoln Experiments'' project,
https://snorpey.github.io/triangulation/ released by Georg Fischer under the MIT license
(Copyright 2013 Georg Fischer). The image used was taken by Torsten Wittmann and is public domain,
available from http://www.cellimagelibrary.org/images/240
