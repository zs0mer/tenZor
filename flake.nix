{
  description = "C/C++ environment";

  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable"; };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; };
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gnumake
          pkg-config
          gnumake
          gcc
          gdb
          autoconf
          automake
          libtool
          direnv
          cmake
          doctest
          nixfmt
          perf
          valgrind
          python3
          cudaPackages.cuda_nvcc
          cudaPackages.cudatoolkit
        ];

        shellHook = "";
      };
    };
}
