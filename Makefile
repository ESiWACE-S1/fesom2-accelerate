
COMPILER = nvcc
INCLUDES = -Iinclude
LIBRARIES = -lcudart
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

build/fesom2-accelerate.o: src/fesom2-accelerate.cu
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/fesom2-accelerate.cu -o build/fesom2-accelerate.o

fesom2-accelerate: build/reference.o build/fesom2-accelerate.o
	${COMPILER} ${CFLAGS} -Xcompiler -fPIC --shared -o build/libfesom2-accelerate.so build/reference.o build/fesom2-accelerate.o ${LIBRARIES}
