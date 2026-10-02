# https://nixos.org/manual/nixpkgs/stable/#python
# https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md

{
  lib,
  buildPythonPackage,
  # fetchFromGitHub,
  fetchPypi,
  setuptools,

  # package deps
  apispec,
  apispec-webframeworks,
  boto3,
  charset-normalizer,
  cookiecutter,
  deepdiff,
  flask,
  flask-cors,
  flask-restful,
  flask-socketio,
  gevent,
  gevent-websocket,
  gitignore-parser,
  kthread,
  markdown,
  marshmallow,
  networkx,
  openpyxl,
  pandas,
  passlib,
  pyarrow,
  pymongo,
  python-dotenv,
  pytz,
  requests,
  simple-websocket,
  sqlalchemy,
  tabulate,
  toml,
  twisted,
  tzlocal,
  watchdog,

  # optional deps
  # pyarrow,
  pyngrok,
  pyodbc,
  python-magic,
  rdp,
}:

buildPythonPackage rec {
  pname = "taipy";
  version = "4.1.1";
  # pyproject = true;
  format = "wheel";

  __darwinAllowLocalNetworking = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-rRSFiEp3Ln468P4uaN46jYZGS/BsjQVaLEDxWJEo9yg=";
  };

  build-system = [ setuptools ];

  dependencies = [
    apispec # [ yaml ]
    apispec-webframeworks
    boto3
    charset-normalizer
    cookiecutter
    deepdiff
    flask
    flask-cors
    flask-restful
    flask-socketio
    gevent
    gevent-websocket
    gitignore-parser
    kthread
    markdown
    marshmallow
    networkx
    openpyxl
    pandas
    passlib
    pyarrow
    pymongo # [ srv ]
    python-dotenv
    pytz
    requests
    simple-websocket
    sqlalchemy
    tabulate
    toml
    twisted
    tzlocal
    watchdog
  ];

  optional-dependencies = {
    ngrok = [ pyngrok ];
    image = [ python-magic ];
    rdp = [ rdp ];
    arrow = [ pyarrow ];
    mssql = [ pyodbc ];
  };

  # pypi package does not include the tests, but cannot be built with fetchFromGitHub
  doCheck = false;
  # nativeCheckInputs = [
  #   pytestCheckHook # https://nixos.org/manual/nixpkgs/stable/#using-pytestcheckhook
  # ];

  pythonImportsCheck = [
    "taipy.gui.Gui"
    "taipy.gui.builder"
  ];

  meta = {
    description = "A 360° open-source platform from Python pilots to production-ready web apps.";
    homepage = "https://www.taipy.io/";
    license = lib.licenses.asl20;
  };
}
