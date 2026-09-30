{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:

    let
      forEachSystem =
        makeFlakeWithSystem:
        nixpkgs.lib.genAttrs [
          "aarch64-darwin"
          "aarch64-linux"
          "x86_64-darwin"
          "x86_64-linux"
        ] makeFlakeWithSystem;
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            buildInputs =
              let
                python = pkgs.python313;
                pyPkgs = pkgs.python313Packages;
                inherit (pkgs.lib) callPackageWith;
              in
              [
                pkgs.zig
                python
                # pyPkgs.pip
                pyPkgs.numpy

                (callPackageWith (pyPkgs) ./nix/taipy/default.nix {
                  inherit (pkgs) lib stdenv fetchFromGitHub;
                  kthread = callPackageWith (pyPkgs // pkgs) ./nix/kthread/default.nix { };
                  rdp = callPackageWith (pyPkgs // pkgs) ./nix/rdp/default.nix { };
                })
              ];
          };
        }
      );
    };
}
