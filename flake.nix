{
  description = "C++ / Cuda environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    oldNixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  };

  outputs =
    {
      nixpkgs,
      oldNixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
      };

      old = import oldNixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gnumake
          pkg-config
          cmake

          old.gcc11
          gdb
          python3

          doctest

          perf
          valgrind
          clang-tools

          old.cudaPackages.cuda_nvcc
          old.cudaPackages.cudatoolkit

          neocmakelsp
        ];

        shellHook = "";
      };
    };
}
