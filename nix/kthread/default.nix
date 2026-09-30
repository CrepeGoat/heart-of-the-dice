# https://nixos.org/manual/nixpkgs/stable/#python
# https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md

{
  lib,
  buildPythonPackage,
  fetchPypi,
  pytestCheckHook,
  setuptools,
}:

buildPythonPackage rec {
  pname = "kthread";
  version = "0.2.3";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-kOGU5qf/kDBAxBM9PqkDfJCMQpa/X1gsf9z2MloE+bQ=";
  };

  build-system = [ setuptools ];

  nativeCheckInputs = [
    # pytestCheckHook # https://nixos.org/manual/nixpkgs/stable/#using-pytestcheckhook
  ];

  pythonImportsCheck = [ "kthread" ];

  meta = {
    description = "Killable threads in Python!";
    homepage = "https://github.com/munshigroup/kthread";
    license = lib.licenses.mit;
  };
}
