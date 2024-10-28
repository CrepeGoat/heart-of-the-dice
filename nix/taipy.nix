{
  fetchFromGitHub,
  python3Packages,
  # dependencies
  nodePackages,
}:

let
  name = "taipy";
  version = "4.0.0";
in
python3Packages.buildPythonPackage {
  pname = name;
  inherit version;

  src = fetchFromGitHub {
    owner = "avaiga";
    repo = "taipy";
    rev = version;
    hash = "sha256-wuI5OY2IRpAMNx6ovXSTqhDXgkKkyLpk17tm5a4RmIc=";
  };

  buildInputs = [ nodePackages.npm ];

}
