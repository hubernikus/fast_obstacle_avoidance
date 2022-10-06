clc; clear all; close all;

my_vfh = controllerVFH;

target_dir = 0.1;
angles = linspace(0, 2*pi, 10);
ranges = 1 - cos(angles);

output = my_vfh(ranges, angles, target_dir)
