{
  # nix stdlib
  python3Packages,
  # dependencies
  taipy,
}:
python3Packages.buildPythonPackage {
  pname = "heart of the dice";
  version = "0.0.1";

  src = ./.;

  buildInputs = [
    # from pyproject.toml
    python3Packages.numpy
    taipy
  ];
}
