
# Compilers
CXX = g++
LIBER = ar

# Flags
CXXFLAGS = -Ofast -ffast-math # -fsanitize=address -O0 -g -march=native #-ffast-math   #-std=c++10 -O3 -Wall -g -g -fsanitize=address 

# Paths and files
APPNAME = bin/libZI.a
SRCPATH = src
OBJPATH = obj
CPP_SRC = $(wildcard $(SRCPATH)/*.cpp)

#object file paths
CPP_OBJ = $(patsubst $(SRCPATH)/%.cpp, $(OBJPATH)/%.o, $(CPP_SRC))

# Targets
all: $(APPNAME)

#build from .o
$(APPNAME): $(CPP_OBJ) $(CUDA_OBJ)
	$(LIBER) rcs -o $@ $^  

#bulid .cpp files
$(OBJPATH)/%.o: $(SRCPATH)/%.cpp
	mkdir -p $(OBJPATH)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(APPNAME) $(OBJPATH)



#konsole "make" -> (may be Vscode "make" as vell) Vscode "./out"
