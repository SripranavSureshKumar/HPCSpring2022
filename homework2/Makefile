CFLAGS = -std=c++11 -fopenmp -O3
CC = g++

all: val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

val_test01_solved: val_test01_solved.cpp
	${CC} -o val_test01_solved ${CFLAGS} val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	${CC} -o val_test02_solved ${CFLAGS} val_test02_solved.cpp

MMult1: MMult1.cpp
	${CC} -o MMult1 ${CFLAGS} MMult1.cpp

omp_solved2: omp_solved2.c 
	${CC} -o omp_solved2 ${CFLAGS} omp_solved2.c 

omp_solved3: omp_solved3.c 
	${CC} -o omp_solved3 ${CFLAGS} omp_solved3.c 

omp_solved4: omp_solved4.c 
	${CC} -o omp_solved4 ${CFLAGS} omp_solved4.c 

omp_solved5: omp_solved5.c 
	${CC} -o omp_solved5 ${CFLAGS} omp_solved5.c 

omp_solved6: omp_solved6.c 
	${CC} -o omp_solved6 ${CFLAGS} omp_solved6.c 

jacobi2D-omp: jacobi2D-omp.cpp
	${CC} -o jacobi2D-omp ${CFLAGS} jacobi2D-omp.cpp

gs2D-omp: gs2D-omp.cpp
	${CC} -o gs2D-omp ${CFLAGS} gs2D-omp.cpp
