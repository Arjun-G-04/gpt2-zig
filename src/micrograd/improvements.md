# Further Improvements

- [ ] MLP Layers can be just a slice, since its fixed for the lifetime of the program.
- [ ] optionality in neurons to be removed. in zig, it should be optional only for actual use case. so here b and w is definitely needed, so they cant be optional.
- [ ] children need not be a null array. it can be an array of null. if its a null array, extra memory needed to know whether its null or not
- [ ] less verbose in backward fns
- [ ] unnecessary optionality in topo sort
- [ ] make all compute in nn read only