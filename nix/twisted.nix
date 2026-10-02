{
  lib,
  stdenv,
  buildPythonPackage,
  pythonAtLeast,
  pythonOlder,
  fetchPypi,
  fetchpatch,

  # build-system
  hatchling,
  hatch-fancy-pypi-readme,

  # dependencies
  attrs,
  automat,
  constantly,
  hyperlink,
  incremental,
  typing-extensions,
  zope-interface,

  # optional-dependencies
  appdirs,
  bcrypt,
  cryptography,
  h2,
  idna,
  priority,
  pyopenssl,
  pyserial,
  service-identity,

  # tests
  tox,
  cython-test-exception-raiser,
  gitMinimal,
  glibcLocales,
  pyhamcrest,
  hypothesis,

  # for passthru.tests
  cassandra-driver,
  httpx,
  klein,
  magic-wormhole,
  scrapy,
  treq,
  txaio,
  txamqp,
  txrequests,
  txtorcon,
  thrift,
  nixosTests,
}:

buildPythonPackage rec {
  pname = "twisted";
  version = "26.4.0";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    extension = "tar.gz";
    hash = "sha256-2/0P4e5AnQJD/dempv8U9JSM7B/XjgN2KR+AXhUB+uk=";
  };

  __darwinAllowLocalNetworking = true;

  build-system = [
    hatchling
    hatch-fancy-pypi-readme
    incremental
  ];

  dependencies = [
    attrs
    automat
    constantly
    hyperlink
    incremental
    typing-extensions
    zope-interface
  ];

  # https://discourse.nixos.org/t/packaging-python-for-nixpkgs-w-dependency-from-github/40415
  # https://github.com/NixOS/nixpkgs/issues/285234
  dontCheckRuntimeDeps = true;

  # Generate Twisted's plug-in cache. Twisted users must do it as well. See
  # http://twistedmatrix.com/documents/current/core/howto/plugin.html#auto3
  # and http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=477103 for details.
  postFixup = lib.optionalString (stdenv.buildPlatform.canExecute stdenv.hostPlatform) ''
    $out/bin/twistd --help > /dev/null
  '';

  nativeCheckInputs = [
    gitMinimal
    glibcLocales
  ]
  ++ optional-dependencies.test
  ++ optional-dependencies.conch
  ++ optional-dependencies.http2
  ++ optional-dependencies.serial
  ++ optional-dependencies.tls;

  preCheck = ''
    export SOURCE_DATE_EPOCH=315532800
    export PATH=$out/bin:$PATH
  '';

  checkPhase = ''
    runHook preCheck
    ${tox}/tox -e nocov
    runHook postCheck
  '';

  optional-dependencies = {
    conch = [
      appdirs
      bcrypt
      cryptography
    ];
    http2 = [
      h2
      priority
    ];
    serial = [ pyserial ];
    test = [
      cython-test-exception-raiser
      pyhamcrest
      hypothesis
      httpx
    ]
    ++ optional-dependencies.http2;
    tls = [
      idna
      pyopenssl
      service-identity
    ];
  };

  passthru = {
    tests = {
      inherit
        cassandra-driver
        klein
        magic-wormhole
        scrapy
        treq
        txaio
        txamqp
        txrequests
        txtorcon
        thrift
        ;
      inherit (nixosTests) buildbot matrix-synapse;
    };
  };

  meta = {
    changelog = "https://github.com/twisted/twisted/blob/twisted-${version}/NEWS.rst";
    homepage = "https://github.com/twisted/twisted";
    description = "Asynchronous networking framework written in Python";
    license = lib.licenses.mit;
    maintainers = [ ];
  };
}
