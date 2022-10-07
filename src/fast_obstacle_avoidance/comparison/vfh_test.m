clc; clear all; close all;

angles = linspace(-pi/2, pi/2, 1000);
ranges = 2.0 - cos(angles);
%ranges = ranges;

input_direction = 0.1;

% vfh = controllerVFH('NumAngularSectors', 10);
vfh = controllerVFH();
vfh.NumAngularSectors = 20;
vfh.NumAngularSectors = [3 5];
% vfh.RobotRadius = vfh_options.RobotRadius;

output_direction = vfh(ranges, angles, input_direction)

% output_direction = vfh_func(ranges, angles, input_direction)

h = figure;
set(h,'Position',[50 50 800 400])
show(vfh)