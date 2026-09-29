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
  pname = "kthread";
  version = "0.2.3";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "munshigroup";
    repo = "kthread";
    tag = "v${version}";
    hash = "";
  };

  build-system = [ setuptools ];

  nativeCheckInputs = [
    pytestCheckHook # https://nixos.org/manual/nixpkgs/stable/#using-pytestcheckhook
  ];

  pythonImportsCheck = [ "kthread" ];

  meta = {
    description = "Killable threads in Python!";
    homepage = "https://github.com/munshigroup/kthread";
    license = lib.licenses.mit;
  };
}
