# Contributing to cellGPU {#contrib}

Contributions are welcomed via pull requests on the repository homepage! Before beginning work and
submitting a pull request, please contact the lead developer (currently Daniel Sussman) to see
if your intended feature is under development by anyone else at the moment and to make sure your plans
match up well with the existing code base.

# Currently planned features

The developers are actively thinking about or interested in adding the following features. If you
are interested in contributing to any of these development branches please contact the lead developer!
Please also see the public gitlab cellGPU page for more detailed issues

## Self-propelled Voronoi branch

1. The current implementation of the test-and-repair scheme should eventually be ported over to a
fully GPU-accelerated scheme; this, after all, was the initial motivation for using the Chen and
Gotsman "candidate 1-ring" approach instead of a more conventional star-splaying method. High
priority.
    - As an interim feature, add multi-threaded CPU support to the test-and-repair scheme via OpenMP or equivalent. Medium priority.
2. The current implementation is restricted to square periodic domains. Extensions to non-square
periodic domains should be trivial; extensions to fixed boundary conditions would be interesting.
Medium priority.

3. The SPV model has a natural extension to 3D models (see the work of M. Merkel). The CPU-branch
implementation would be straightforward; the GPU branch may take a bit more thought. Medium
priority.

## Vertex model branch

1. Extend to allow more general changes to the network topology. High priority.

2. Allow fixed boundaries and edges. High priority.

3. Extend to 3D models. Medium priority.


# New features

New, widely applicable features are favored over the creation of niche code. For example,
introducing entirely new boundary conditions would be a much stronger submission than adding a
slight modification of an existing energy functional. All code should have a functioning CPU-
only branch as well as a GPU-accelerated branch, and the user should be able to select either
branch at will.

Pull requests that introduce new, external dependencies for compilation and execution are
disfavored, but will be considered if the new feature is sufficiently general.

# Source code conventions

## Coding

Code should be written in a style consistent with the existing code base. As a brief summary, the
Whitesmith indentation style should be used, and 4 spaces, and not tabs, should be used to indent
lines. A soft maximum line length of 120 characters should be used, with very long lines of code
broken at some natural point.

Variable names should be descriptive; prefer lower camelCase names to using other delimiters. When
using ArrayHandles stick to the h_variableName and d_variableName convention for accessing GPUArrays
on either the host or device.

## Documentation

Every class, member, function, etc., should be documented with doxygen comments.

## Directory structure

This repository currently follows a simple structure. The main executables, Vertex.cpp and
voronoi.cpp are in the base directory and can be used to reproduce timing information. Other example files are in examples/
headers and .cu/.cpp files are in inc/ and src/ directories, respectively, with a subdirectory
structure for updaters, models, utilities, analysis, and database creators. Object files get compiled
in the obj/ directory. A simple makefile is used to compile everything and all cpp files in the root
directory.

### Optimizations

There is always a tension between optimizing performance of the program and easy code readability.
Since we are going to the trouble of writing the first GPU-accelerated code, this project errs on
the side of optimizations, in particular with regard to very flat data structures, while still
trying to maintain an OO-perspective for code growth and maintainability. In costly functions
unexpected optimizations are allowed, but a straightforward and less optimized function that does
the same thing should be provided for testing and debugging. Once the optimized functions are
thoroughly vetted, the unoptimized code paths can be relegated to in-code documentation.
