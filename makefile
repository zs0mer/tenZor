
# Compilers
CXX = g++
AR = ar

# Flags
APP_INCLUDE = -Iinclude
LIB_FLAGS = -Ofast -ffast-math $(APP_INCLUDE)
APP_FLAGS = -Ofast -ffast-math -pthread $(APP_INCLUDE)
APP_LIBS = -Lbin -lTenzor -Iinclude $(APP_INCLUDE)

# Paths and files
LIB = bin/libTenzor.a
L_SRCPATH = src
L_OBJPATH = bin/obj
L_CPP_SRC = $(wildcard $(L_SRCPATH)/*.cpp)

APPNAME = Demo/targets/out
A_SRCPATH = Demo/src
A_OBJPATH = Demo/targets/obj
A_CPP_SRC = $(wildcard $(A_SRCPATH)/*.cpp)

#object file paths
L_CPP_OBJ = $(patsubst $(L_SRCPATH)/%.cpp, $(L_OBJPATH)/%.o, $(L_CPP_SRC))
A_CPP_OBJ = $(patsubst $(A_SRCPATH)/%.cpp, $(A_OBJPATH)/%.o, $(A_CPP_SRC))

# Targets
all: $(APPNAME)

app: $(APPNAME)

lib: $(LIB)

# Demo
$(APPNAME): $(A_CPP_OBJ) $(LIB)
	$(CXX) -o $@ $(A_CPP_OBJ) $(APP_LIBS)

$(A_OBJPATH)/%.o: $(A_SRCPATH)/%.cpp
	mkdir -p $(A_OBJPATH)
	$(CXX) $(APP_FLAGS) -c $< -o $@

# Libary
$(LIB): $(L_CPP_OBJ)
	$(AR) rcs $@ $^ 

$(L_OBJPATH)/%.o: $(L_SRCPATH)/%.cpp
	mkdir -p $(L_OBJPATH)
	$(CXX) $(LIB_FLAGS) -c $< -o $@	 

clean: capp clib

capp:
	rm -rf $(APPNAME) $(A_OBJPATH) 

clib:
	rm -rf $(LIB) $(L_OBJPATH)
