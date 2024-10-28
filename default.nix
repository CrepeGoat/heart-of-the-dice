let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
  pkgs = import nixpkgs { };

  # taipy = pkgs.callPackage ./nix/taipy.nix { };
in
# pkgs.callPackage ./package.nix { inherit taipy; }
pkgs.callPackage ./nix/taipy.nix { }