# generate_instantiations.py

types_file = "src/instantiate_helper/types.txt"
functions_file = "src/instantiate_helper/functions.txt"
output_file = "src/tensor-inst.cpp"

# Read types
with open(types_file) as f:
    types = [line.strip() for line in f if line.strip()]

# Read functions
with open(functions_file) as f:
    functions = [line.strip() for line in f if line.strip()]

with open(output_file, "w") as f:
    f.write('#include "Tenzor.hpp"\n\n')
    f.write("namespace TZ {\n")

    # Instantiate class templates
    for t in types:
        f.write(f"template class Tensor<{t}>;\n")
    f.write("\n\n")

    # Instantiate member functions
    for t in types:
        f.write(f"// {t}\n")
        for func in functions:
            # Check if it’s a constructor
            if func.startswith("Tensor("):
                # Explicit constructor instantiation
                f.write(f"template Tensor<{t}>::{func.replace('<T>', f'<{t}>')}\n")
            else:
                # Normal member function
                first_space = func.find(" ")
                func = f"template {func[:first_space]} Tensor<T>::{func[first_space+1:]};"
                f.write(f"{func.replace('<T>', f'<{t}>')}\n")
        f.write("\n\n")

    f.write("};\n")

print(f"Generated {output_file} for types: {types} and functions: {functions}")
