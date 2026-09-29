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
            buildInputs = with pkgs; [
              zig
              python313
              # python313Packages.pip
              python313Packages.numpy
              (lib.callPackageWith (pkgs // pkgs.python313Packages) ./nix/taipy/default.nix {
                kthread = lib.callPackageWith (pkgs // pkgs.python313Packages) ./nix/kthread/default.nix { };
                rdp = lib.callPackageWith (pkgs // pkgs.python313Packages) ./nix/rdp/default.nix { };
              })
            ];
          };
        }
      );
    };
}
