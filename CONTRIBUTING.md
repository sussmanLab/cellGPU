Contributions are welcomed via pull requests on the repository homepage! Before beginning work and
submitting a pull request, please contact the lead developer (currently Daniel M. Sussman) to see
if your intended feature is currently under development by anyone else and to make sure your plans
match up well with the existing code base.

#New features

New, widely applicable features are favored over the creation of niche code. For example,
introducing entirely new boundary conditions would be a much stronger submission than adding a
slight modification of an existing energy functional. All code should have a functioning CPU-
only branch as well as a GPU-accelerated branch, and the user should be able to select either
branch at will.

Pull requests that introduce new, external dependencies for compilation and execution are
disfavored, but will be considered if the new feature is sufficiently general

#Source code conventions

## Coding

Code should be written in a style consistent with the existing code base. As a brief summary, 4
spaces, and not tabs, should be used to indent lines, and the Whitesmith style should be used. A
A soft maximum line length of 120 characters should be used, with very long lines of code broken
at some natural point.

##Documentation

Every class, member, function, etc., should be documented with doxygen comments.
