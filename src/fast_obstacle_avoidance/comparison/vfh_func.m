function [output_direction] = vfh_func(ranges, angles, input_direction, vfh_options)
%vfg_func - A wrapper for the controllerVFH, in order to allow use with
%python.
vfh = controllerVFH;

if nargin > 3
    if isfield(vfh_options,'RobotRadius')
        vfh.RobotRadius = vfh_options.RobotRadius;
    end
end

output_direction = vfh(ranges, angles, input_direction);
end