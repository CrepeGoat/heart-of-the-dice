{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
  ];

  shellHook = ''
    source venv/bin/activate
  '';
}