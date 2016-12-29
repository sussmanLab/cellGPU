DMS has not decided under what license to release the main branch of the code... some files and functionality draw from existing open-source code, as described below.

Two files (gpuarray.h and indexer.h) are part of the HOOMD-blue project, released under the BSD 3-Clause License:

HOOMD-blue Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer both in the code and prominently in any materials provided with the distribution.
3. Neither the name ofthe copyright holder nor the names of its contributors may be used to enorse or promote products derived from this software without specific prior written permission

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The file HilbertSort.h calls, but does not modify, code released under the GNU LGPL licence by John Burkardt. My undertanding is that DelGPU thus does not constitute a derivate work, and thus can be distributed under a non-(L)GPL license
https://people.sc.fsu.edu/~jburkardt/cpp_src/hilbert_curve/hilbert_curve.html

Finally, the logo used, DelFrog.jpg, was generated using the ``Lincoln Experiments'' project, released by Georg Fischer under the MIT license (Copyright 2013 Georg Fischer)


