# Compiler
CXX = g++

# Flags
APP_INCLUDE = -Iinclude -Ilib
# using c++17
APP_FLAGS = -O0 -g $(APP_INCLUDE) # -fsanitize=address -pthread 
_APP_FLAGS = -Ofast -ffast-math $(APP_INCLUDE)

# Paths and files
APPNAME = Demo/targets/out
SRCPATH = Demo/src
LIBSRC = lib
OBJPATH = Demo/targets/obj

CPP_SRC = $(wildcard $(SRCPATH)/*.cpp)
LIB_SRC = $(wildcard $(LIBSRC)/*.cpp)

CPP_OBJ = $(patsubst $(SRCPATH)/%.cpp, $(OBJPATH)/%.o, $(CPP_SRC))
LIB_OBJ = $(patsubst $(LIBSRC)/%.cpp, $(OBJPATH)/%.o, $(LIB_SRC))

# Target
all: clean $(APPNAME)

$(APPNAME): $(CPP_OBJ) $(LIB_OBJ)
	mkdir -p $(dir $@)
	$(CXX) -o $@ $(CPP_OBJ) $(LIB_OBJ) $(APP_FLAGS)

$(OBJPATH)/%.o: $(SRCPATH)/%.cpp
	mkdir -p $(OBJPATH)
	$(CXX) $(APP_FLAGS) -c $< -o $@

$(OBJPATH)/%.o: $(LIBSRC)/%.cpp
	mkdir -p $(OBJPATH)
	$(CXX) $(APP_FLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(APPNAME) $(OBJPATH)
