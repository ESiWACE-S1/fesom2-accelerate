
NVCC = nvcc
INCLUDES = -Iinclude
LIBRARIES = -lcudart


fesom2-accelerate:
	mkdir -p build
	${NVCC} -Xcompiler -fPIC -x cu -rdc=true ${INCLUDES} -c src/fesom2-accelerate.cu -o build/fesom2-accelerate.o
	${NVCC} -Xcompiler -fPIC --shared -o build/libfesom2-accelerate.so build/fesom2-accelerate.o ${LIBRARIES}


clean:
	rm -rf build/*.o
	rm -rf build/*.so
