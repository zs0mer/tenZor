# Compiler
CXX = g++

# Flags
APP_INCLUDE = -Iinclude
# using c++17
APP_FLAGS = -O0 -g $(APP_INCLUDE) # -fsanitize=address -pthread 
_APP_FLAGS = -Ofast -ffast-math $(APP_INCLUDE)

# Paths and files
APPNAME = Demo/targets/out
SRCPATH = Demo/src
OBJPATH = Demo/targets/obj
CPP_SRC = $(wildcard $(SRCPATH)/*.cpp)
CPP_OBJ = $(patsubst $(SRCPATH)/%.cpp, $(OBJPATH)/%.o, $(CPP_SRC))

# Target
all: clean $(APPNAME)

$(APPNAME): $(CPP_OBJ)
	mkdir -p $(dir $@)
	$(CXX) -o $@ $(CPP_OBJ) $(APP_FLAGS)

$(OBJPATH)/%.o: $(SRCPATH)/%.cpp
	mkdir -p $(OBJPATH)
	$(CXX) $(APP_FLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(APPNAME) $(OBJPATH)
