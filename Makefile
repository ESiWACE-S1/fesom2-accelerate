
COMPILER = nvcc
INCLUDES = -Iinclude
LIBRARIES = -lcudart

all: fesom2-accelerate

clean:
	rm -rf build/*.o
	rm -rf build/*.so

build/reference.o: src/reference.cpp
	mkdir -p build
	${COMPILER} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/reference.cpp -o build/reference.o

build/fesom2-accelerate.o: src/fesom2-accelerate.cu
	${COMPILER} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/fesom2-accelerate.cu -o build/fesom2-accelerate.o

fesom2-accelerate: build/reference.o build/fesom2-accelerate.o
	${COMPILER} -Xcompiler -fPIC --shared -o build/libfesom2-accelerate.so build/reference.o build/fesom2-accelerate.o ${LIBRARIES}
