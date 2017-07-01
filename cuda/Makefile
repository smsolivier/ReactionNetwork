CXX=nvcc
CFLAGS=-std=c++11 
CFLAGS += -Wno-deprecated-gpu-targets -Xcompiler 
CFLAGS += -fopenmp

LIBLOC = /home/sam/cpp/
LIBS = MyLib
CLIB = $(addprefix -l, $(LIBS))

COMPILE = $(CXX) $(CFLAGS) -I $(LIBLOC) -L $(LIBLOC) $(CLIB)

SRC = main.cc

# convert cc 
OBJ = $(subst .cc,.o,$(SRC))

OBJDIR = obj/

OBJ := $(addprefix $(OBJDIR), $(OBJ))

exe = x 

all: $(OBJDIR) $(exe)

$(OBJDIR): 
	mkdir $(OBJDIR) 

$(exe) : $(OBJ)
	$(COMPILE) -o $(exe) $(OBJ)

$(OBJDIR)%.o : %.cc
	$(COMPILE) -c $< -o $@ 

.PHONY : clean 
clean : 
	rm -rf $(OBJDIR)
	rm -f $(exe) 