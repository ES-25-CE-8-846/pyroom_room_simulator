# save this as shell.nix or default.nix
{ pkgs ? import <nixpkgs> { } }:

let
  pythonPackages = pkgs.python311Packages;

  pyroomacoustics =
    let
      pname = "pyroomacoustics";
      version = "0.8.3";
    in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "5619bcab3d03ffa4221383cf1ebaa2f078335c08d45601e10864612c8cbbc7d3";
      };
      propagatedBuildInputs = with pythonPackages; [
        cython
        numpy
        scipy
        pybind11
      ];
      doCheck = false;
    };

in
pkgs.mkShell {
  buildInputs = with pythonPackages; [ pyroomacoustics ]
    ++ (with pkgs; [
    python311
  ])
    ++ (with pythonPackages; [
    numpy
    scipy
    matplotlib
    jupyter-core
    ipykernel
    jupyterlab
    sounddevice
    soundfile
    tqdm
  ]);
}
