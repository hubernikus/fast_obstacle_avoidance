function [output_direction] = vfh_func(ranges, angles, input_direction, vfh_options)
%vfg_func - A wrapper for the controllerVFH, in order to allow use with
%python.
vfh = controllerVFH;

if nargin > 3
    % Assign values / options
    if isfield(vfh_options,'RobotRadius')
        vfh.RobotRadius = vfh_options.RobotRadius;
    end

    if isfield(vfh_options,'HistogramThresholds')
        vfh.HistogramThresholds = vfh_options.HistogramThresholds;
    end

    if isfield(vfh_options,'NumAngularSectors')
        vfh.NumAngularSectors = vfh_options.NumAngularSectors;
    ;end
end

output_direction = vfh(ranges, angles, input_direction);
end
