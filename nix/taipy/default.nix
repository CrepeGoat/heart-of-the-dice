# https://nixos.org/manual/nixpkgs/stable/#python
# https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md

{
  lib,
  stdenv,
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,
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
  pyproject = true;

  __darwinAllowLocalNetworking = true;

  src = fetchFromGitHub {
    owner = "Avaiga";
    repo = "taipy";
    tag = "v${version}";
    hash = "sha256-cQnCTMmpdkvWwt7RFAIhAfmhVwGVn0Y8Z5Tr6lzDmS8=";
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

  nativeCheckInputs = [
    pytestCheckHook # https://nixos.org/manual/nixpkgs/stable/#using-pytestcheckhook
  ];

  disabledTests = [
  ];

  disabledTestPaths = lib.optionals (stdenv.hostPlatform.isDarwin && stdenv.hostPlatform.isAarch64) [
  ];

  # pythonImportsCheck = [ "requests" ];

  meta = {
    description = "A 360° open-source platform from Python pilots to production-ready web apps.";
    homepage = "https://www.taipy.io/";
    license = lib.licenses.asl20;
  };
}
