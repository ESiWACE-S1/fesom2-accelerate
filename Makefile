
COMPILER = nvcc
INCLUDES = -Iinclude
LIBRARIES = -lcudart
OBJECTS = build/reference.o build/fesom2-accelerate.o build/fct_ale_a1.o build/fct_ale_a2.o build/fct_ale_a3.o build/fct_ale_b1_vertical.o build/fct_ale_b1_horizontal.o build/fct_ale_b2.o build/fct_ale_b3_vertical.o build/fct_ale_b3_horizontal.o build/fct_ale_c_vertical.o build/fct_ale_c_horizontal.o
# CFLAGS
GPU_ARCH="sm_60"
CFLAGS = --std=c++14 --gpu-architecture ${GPU_ARCH}
ifndef DEBUG
	CFLAGS += -O3
else
	CFLAGS += -O0 -g
endif
ifdef PROFILE
	CFLAGS += -pg
endif

all: fesom2-accelerate

clean:
	rm -rf build/*.o
	rm -rf build/*.so

build/reference.o: src/reference.cpp
	mkdir -p build
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/reference.cpp -o build/reference.o

build/fct_ale_a1.o: kernels/fct_ale_a1.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_a1.cu -o build/fct_ale_a1.o

build/fct_ale_a2.o: kernels/fct_ale_a2.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_a2.cu -o build/fct_ale_a2.o

build/fct_ale_a3.o: kernels/fct_ale_a3.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_a3.cu -o build/fct_ale_a3.o

build/fct_ale_b1_vertical.o: kernels/fct_ale_b1_vertical.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_b1_vertical.cu -o build/fct_ale_b1_vertical.o

build/fct_ale_b1_horizontal.o: kernels/fct_ale_b1_horizontal.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_b1_horizontal.cu -o build/fct_ale_b1_horizontal.o

build/fct_ale_b2.o: kernels/fct_ale_b2.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_b2.cu -o build/fct_ale_b2.o

build/fct_ale_b3_vertical.o: kernels/fct_ale_b3_vertical.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_b3_vertical.cu -o build/fct_ale_b3_vertical.o

build/fct_ale_b3_horizontal.o: kernels/fct_ale_b3_horizontal.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_b3_horizontal.cu -o build/fct_ale_b3_horizontal.o

build/fct_ale_c_vertical.o: kernels/fct_ale_c_vertical.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_c_vertical.cu -o build/fct_ale_c_vertical.o

build/fct_ale_c_horizontal.o: kernels/fct_ale_c_horizontal.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c kernels/fct_ale_c_horizontal.cu -o build/fct_ale_c_horizontal.o

build/fesom2-accelerate.o: src/fesom2-accelerate.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/fesom2-accelerate.cu -o build/fesom2-accelerate.o

fesom2-accelerate: ${OBJECTS}
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC --shared -o build/libfesom2-accelerate.so ${OBJECTS} ${LIBRARIES}
