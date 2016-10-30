 
all: token_trace.cpp ocl_base.o ocl_ttrace.o
	g++ -o token_trace token_trace.cpp ocl_base.o ocl_ttrace.o -lopencv_core -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lOpenCL -lrt -lm

ocl_base.o: ocl/ocl_base.h ocl/ocl_base.cpp
	g++ -c ocl/ocl_base.cpp
	
ocl_ttrace.o: ocl/ocl_ttrace.h ocl/ocl_ttrace.cpp
	g++ -c ocl/ocl_ttrace.cpp

clean:
	rm *.o


