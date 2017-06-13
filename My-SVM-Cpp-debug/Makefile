CXX = g++
CXXFLAGS = -g -Wall
CXXFLAGS += -g -std=c++11 -Wall -pedantic
SRCS = main.cpp mnist.cpp
PROG = main

$(PROG):$(SRCS)
	$(CXX) $(CXXFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean: ; rm -f *.o $(PROG)