CXX=nvcc
CFLAGS=-std=c++11 -Wno-deprecated-gpu-targets
COMPILE = $(CXX) $(CFLAGS) 

SRC = main.cc
SRC += helper.cc
SRC += LinearAlgebra.cc 
SRC += LinearSolver.cc
SRC += Timer.cc 

SRC_GPU = MatrixVectorProduct.cu

# convert cc 
OBJ = $(subst .cc,.o,$(SRC))

# convert .cu 
OBJ_GPU = $(subst .cu,.o,$(SRC_GPU))

OBJDIR = obj/

OBJ := $(addprefix $(OBJDIR), $(OBJ))
OBJ_GPU := $(addprefix $(OBJDIR), $(OBJ_GPU))

exe = x 

all: $(OBJDIR) $(exe)

$(OBJDIR): 
	mkdir $(OBJDIR) 

$(exe) : $(OBJ_GPU) $(OBJ)
	$(COMPILE) -o $(exe) $(OBJ_GPU) $(OBJ)

$(OBJDIR)%.o : %.cc
	$(COMPILE) -c $< -o $@ 

$(OBJDIR)%.o : %.cu 
	$(COMPILE) -c $< -o $@ 

.PHONY : clean 
clean : 
	rm -rf $(OBJDIR)
	rm -f $(exe) 