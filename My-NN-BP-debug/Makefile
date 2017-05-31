OBJ = main.o NNet.o

CXXFLAGS += -g -std=c++11 -Wall -pedantic

SVision : $(OBJ)
	g++ -o SVision $(OBJ) `pkg-config --libs opencv`  ${CXXFLAGS}

$(OBJ) : NNet.h

.PHONY : cleanall cleanobj
cleanall:
	rm SVision $(OBJ)
cleanobj:
	rm *.o
