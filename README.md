This repo contains various projects that I tackled with in MSc thesis. Some are still private while I still wrap up the final paper. This repo contains the functions and headers needed by the individual projects. The projects are offered in git submodules, so you will need to use a command like this to update the submodules:

```
$ git submodule update --init --recursive
```

All of the projects make use of the [Madagascar Seismic Processing](http://www.ahay.org/wiki/Main_Page) suite. All of the public ones also use CUDA.
The Makefiles are still not optimized with portability in mind. You will most likely have to modify then for your hardware. I am working on a fork with a centralized build system using [meson](https://mesonbuild.com/index.html).
