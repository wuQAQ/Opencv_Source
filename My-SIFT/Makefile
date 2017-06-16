CXX = g++
CXXFLAGS = -g -Wall
CXXFLAGS += -g -std=c++11 -Wall -pedantic
SRCS = main.cpp
PROG = main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CXX) $(CXXFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean: ; rm -f *.o $(PROG)