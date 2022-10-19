clc; 
% clear all; 
close all;
datestr(now, 'HH:MM:SS')

angles = linspace(-pi/2, pi/2, 10);
ranges = 2 - cos(angles);
ranges = 0.5*ranges;

input_direction = 0.1;

% vfh = controllerVFH('NumAngularSectors', 10);
vfh = controllerVFH( ...
    'NumAngularSectors', 20, ...
    'HistogramThresholds', [1, 2]);

% vfg.NumAngularSectors = 10;
% vfh.RobotRadius = vfh_options.RobotRadius;

output_direction = vfh(ranges, angles, input_direction)

% output_direction = vfh_func(ranges, angles, input_direction)

% h = figure;
% set(h,'Position',[50 50 800 400])
% show(vfh)