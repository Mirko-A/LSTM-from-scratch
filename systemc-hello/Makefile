all: hello_s hello_d

hello_s: hello.cpp
	g++ -std=c++17 -I. -I${SYSTEMC_HOME}/include -L. -L${LD_LIBRARY_PATH} -lsystemc -lm -lpthread -o hello_s hello.cpp


hello_d: hello.cpp
	g++ -std=c++17 -I. -I${SYSTEMC_HOME}/include -L. -L${LD_LIBRARY_PATH} -lsystemc -o hello_d hello.cpp

.PHONY: clean
clean:
	rm hello_*
