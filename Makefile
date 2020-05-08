
COMPILER = nvcc
INCLUDES = -Iinclude
LIBRARIES = -lcudart
OBJECTS = build/reference.o build/fesom2-accelerate.o build/fct_ale_a1.o build/fct_ale_a2.o
# CFLAGS
CFLAGS = --std=c++14
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

build/fesom2-accelerate.o: src/fesom2-accelerate.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/fesom2-accelerate.cu -o build/fesom2-accelerate.o

fesom2-accelerate: ${OBJECTS}
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC --shared -o build/libfesom2-accelerate.so ${OBJECTS} ${LIBRARIES}
