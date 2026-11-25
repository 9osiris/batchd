CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
SRC = src/model.cpp src/sched.cpp src/main.cpp
TEST_SRC = src/model.cpp src/sched.cpp tests/test_batchd.cpp

batchd: $(SRC) src/model.h src/sched.h
	$(CXX) $(CXXFLAGS) -o batchd $(SRC)

test_batchd: $(TEST_SRC) src/model.h src/sched.h
	$(CXX) $(CXXFLAGS) -o test_batchd $(TEST_SRC)

test: batchd test_batchd
	./test_batchd
	./batchd verify

demo: batchd
	./batchd demo

bench: batchd
	./batchd bench

clean:
	rm -f batchd test_batchd

.PHONY: test demo bench clean
