# stereolabs
Here is my submission for the technical test.

### Dependencies
- TensorRT >= 10.2.0
- CUDA >= 11.8
- Opencv >= 4.8.0

### Build
(Optional) build the docker image to avoid managing dependencies

```bash
docker build -t stereolabs . # takes a long time bc of the opencv build (~1h on my modest machine)
docker run --gpus all -it --name stereolabs -v /local/folder:/workspace/stereolabs stereolabs
```

```bash
mkdir build && cd build
cmake .. 
make -j4
```

### Usage
```bash
./yolov_trt

```