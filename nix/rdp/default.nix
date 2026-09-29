# https://nixos.org/manual/nixpkgs/stable/#python
# https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md

{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,
  setuptools,
}:

buildPythonPackage rec {
  pname = "rdp";
  version = "0.8";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "fhirschmann";
    repo = "rdp";
    tag = "v${version}";
    hash = "";
  };

  build-system = [ setuptools ];

  nativeCheckInputs = [
    pytestCheckHook # https://nixos.org/manual/nixpkgs/stable/#using-pytestcheckhook
  ];

  pythonImportsCheck = [ "rdp" ];

  meta = {
    description = "Pure Python implementation of the Ramer-Douglas-Peucker algorithm";
    homepage = "https://github.com/fhirschmann/rdp";
    license = lib.licenses.mit;
  };
}
