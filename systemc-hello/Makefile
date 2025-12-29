all: hello_s hello_d

hello_s: hello.cpp
	g++ −std=c++11 −I${SYSTEMC_HOME}/include −L${SYSTEMC_HOME}/lib−linux64 −l:libsystemc.a −lpthread −o hello_s hello.cpp

hello_d: hello.cpp
	g++ −std=c++11 −I${SYSTEMC_HOME}/include −L${SYSTEMC_HOME}/lib−linux64 −lsystemc −o hello_d hello.cpp

.PHONY: clean
clean:
	rm hello_∗
