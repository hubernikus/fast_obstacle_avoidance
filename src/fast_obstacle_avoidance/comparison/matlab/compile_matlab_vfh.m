appFile = fullfile('vfh_func.m');

% Build a Python package using the compiler.build.pythonPackage command.
compiler.build.pythonPackage(appFile);