import math
import numpy as np

from fast_obstacle_avoidance.comparison.histogram_base import HistogramBase


class controllerVFH(HistogramBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # classdef (StrictDefaults) controllerVFH < nav.algs.internal.VectorFieldHistogramBase
    # controllerVFH Avoid obstacles using vector field histogram
    #   The vector field histogram (VFH) algorithm is used for obstacle
    #   avoidance based on range sensor data. Given a range sensor reading
    #   in terms of ranges and angles, and a target direction to drive
    #   towards, the controllerVFH computes an obstacle-free steering
    #   direction using the VFH+ algorithm.
    #
    #   VFH = controllerVFH returns a vector field histogram
    #   object, VFH, that computes a steering direction using the VFH+ algorithm.
    #
    #   VFH = controllerVFH('PropertyName', PropertyValue, )
    #   returns a vector field histogram object, VFH, with each specified
    #   property set to the specified value.
    #
    #   Step method syntax:
    #
    #   STEERINGDIR = step(VFH, SCAN, TARGETDIR) finds an obstacle
    #   free steering direction STEERINGDIR, using the VFH+ algorithm for
    #   a laser scan, SCAN, as a lidarScan object and a scalar input TARGETDIR.
    #   The output STEERINGDIR is in radians. The vehicle's forward direction
    #   is considered zero radians.
    #   The angles measured clockwise from the forward direction are negative
    #   angles and angles measured counter-clockwise from the forward direction
    #   are positive angles.
    #
    #   STEERINGDIR = step(VFH, RANGES, ANGLES, TARGETDIR) allows
    #   you to pass range sensor readings as RANGES and ANGLES. The input
    #   RANGES are in meters, the ANGLES and TARGETDIR are in radians.
    #
    #   System objects may be called directly like a function instead of using
    #   the step method. For example, y = step(obj, x) and y = obj(x) are
    #   equivalent.
    #
    #   controllerVFH methods:
    #
    #   step        - Compute steering direction using the range data
    #   clone       - Create controllerVFH object with same property values
    #   show        - Display controllerVFH information in a figure window
    #   <a href="matlab:help matlab.System/reset   ">reset</a>       - Reset the internal states of controllerVFH System object
    #
    #   controllerVFH properties:
    #
    #   NumAngularSectors       - Number of angular sectors in histogram
    #   DistanceLimits          - Ranges within these limits are considered
    #   RobotRadius             - Radius of the circle circumscribing the vehicle
    #   SafetyDistance          - Safety distance around the vehicle
    #   MinTurningRadius        - Minimum turning radius at current speed
    #   TargetDirectionWeight   - Weight for moving in target direction
    #   CurrentDirectionWeight  - Weight for moving in current direction
    #   PreviousDirectionWeight - Weight for moving in previous direction
    #   HistogramThresholds     - Upper and lower thresholds for histogram
    #   UseLidarScan            - Use lidarScan object instead of ranges and angles
    #
    #   Example:
    #
    #       # Create a vector field histogram object
    #       vfh = controllerVFH('UseLidarScan', True);
    #
    #       # Example laser scan data input
    #       ranges = 10*ones(1, 300);
    #       ranges(1, 130:170) = 1.0;
    #       angles = linspace(-pi/2, pi/2, 300);
    #       scan = lidarScan(ranges, angles);
    #       targetDir = 0;
    #
    #       # Compute obstacle-free steering direction
    #       steeringDir = vfh(scan, targetDir)
    #
    #       # Visualize the controllerVFH computation
    #       show(vfh);
    #
    #   See also controllerPurePursuit, mobileRobotPRM.

    #   Copyright 2015-2020 The MathWorks, Inc.
    #
    #   References:
    #
    #   [1] I. Ulrich and J. Borenstein, "VFH+: reliable obstacle avoidance
    #       for fast mobile robots", Proceedings of IEEE International
    #       Conference on Robotics and Automation, 1998.
    # codegen

    # properties (Nontunable)
    #     #UseLidarScan Use lidarScan object instead of ranges and angles
    #     #   By default, you pass ranges and angles as numeric arrays into
    #     #   the step function. Setting UseLidarScan to "True" allows you to
    #     #   pass a lidarScan object instead.
    #     #
    #     #   Default: false
    #     #
    #     #   See also: lidarScan
    #     UseLidarScan (1, 1) logical = false
    # end

    # properties(Access = {?nav.algs.internal.VectorFieldHistogramBase,
    #                      ?nav.algs.internal.InternalAccess})
    #     #IsShowBeforeStep Flag to prevent calls to show before step
    #     IsShowBeforeStep = True;
    # end

    # methods (Access = protected)
    #     function steeringDir = stepImpl(obj, varargin)
    #     #step Compute control commands and steering direction

    #         steeringDir = stepImpl@nav.algs.internal.VectorFieldHistogramBase(
    #             obj, varargin{:});

    #         # Allow show method to be called
    #         obj.IsShowBeforeStep = false;
    #     end

    #     function resetImpl(obj)
    #     #resetImpl Reset internal states

    #         resetImpl@nav.algs.internal.VectorFieldHistogramBase(obj);

    #         # Prevent calls to show method before step
    #         obj.IsShowBeforeStep = True;
    #     end

    #     function [scan, target, classOfRanges] = parseAndValidateStepInputs(obj, varargin)
    #     #parseAndValidateStepInputs Validate inputs to step function

    #     # Parse and validate
    #         if obj.UseLidarScan
    #             # Only lidarScan input
    #             scan = robotics.internal.validation.validateLidarScan(
    #                 varargin{1}, 'step', 'scan');

    #             target = varargin{2};
    #         else
    #             # Scan as ranges and angles
    #             scan = robotics.internal.validation.validateLidarScan(
    #                 varargin{1}, varargin{2}, 'step', 'ranges', 'angles');

    #             target = varargin{3};
    #         end

    #         classOfRanges = class(scan.Ranges);

    #         # Validate the target direction
    #         validateattributes(target, {'double', 'single'}, {'nonnan', 'real',
    #                             'scalar', 'nonempty', 'finite'}, 'step', 'target direction');

    #     end

    #     function validateInputsImpl(obj, varargin)
    #     #validateInputsImpl Validate inputs before setupImpl is called
    #         [scan, target, classOfRanges] = obj.parseAndValidateStepInputs(varargin{:});

    #         isDataTypeEqual = isequal(classOfRanges, class(target));

    #         coder.internal.errorIf(~isDataTypeEqual,
    #                                'nav:navalgs:vfh:DataTypeMismatch',
    #                                classOfRanges, class(scan.Angles), class(target));
    #     end

    #     function num = getNumInputsImpl(obj)
    #     #getNumInputsImpl Get number of inputs

    #         if obj.UseLidarScan
    #             num = 2;
    #         else
    #             num = 3;
    #         end
    #     end

    #     function flag = isInputSizeMutableImpl(obj, index)
    #     #isInputSizeMutableImpl Mutable input size status
    #     #   This function will be called once for each input of the
    #     #   system block.

    #         if obj.UseLidarScan
    #             # All inputs are fixed size in this case
    #             flag = false;
    #         else
    #             # First two inputs, i.e. ranges and angles are variable sized
    #             # signals.
    #             if (index == 1 || index  == 2)
    #                 flag = True;
    #             else
    #                 flag = false;
    #             end
    #         end
    #     end

    #     function loadObjectImpl(obj, svObj, wasLocked)
    #     #loadObjectImpl Custom load implementation

    #         obj.IsShowBeforeStep = svObj.IsShowBeforeStep;
    #         loadObjectImpl@nav.algs.internal.VectorFieldHistogramBase(obj,svObj,wasLocked);
    #     end

    #     function s = saveObjectImpl(obj)
    #     #saveObjectImpl Custom save object action

    #         s = saveObjectImpl@nav.algs.internal.VectorFieldHistogramBase(obj);
    #         s.IsShowBeforeStep = obj.IsShowBeforeStep;
    #     end
    # end

    # methods
    #     function obj = controllerVFH(varargin)
    #     #controllerVFH Constructor
    #         setProperties(obj,nargin,varargin{:});
    #     end
    # end

    # methods
    #     function ax = show(obj, name, value)
    #     #show Show the histograms in a figure
    #     #   show(VFH) shows different histograms in a figure along with
    #     #   the parameters of the controllerVFH and range values
    #     #   from the last step input.
    #     #
    #     #   AH = show(VFH) returns handles of the axes used by the show
    #     #   function.
    #     #
    #     #   show(VFH,___,Name,Value) provides additional options specified
    #     #   by one or more Name,Value pair arguments. Name must appear
    #     #   inside single quotes (''). You can specify several name-value
    #     #   pair arguments in any order as Name1,Value1,,NameN,ValueN:
    #     #
    #     #       'Parent'        - A two-element array of axes handles
    #     #                         that specifies the parent axes for
    #     #                         the objects created by show.
    #     #
    #     #   Example:
    #     #       # Create a vector field histogram
    #     #       vfh = controllerVFH('UseLidarScan', True);
    #     #
    #     #       # Example laser scan data
    #     #       ranges = 10*ones(1, 300);
    #     #       ranges(1, 130:170) = 1.0;
    #     #       angles = linspace(-math.pi/2, math.pi/2, 300);
    #     #       scan = lidarScan(ranges, angles);
    #     #       targetDir = 0;
    #     #
    #     #       # Compute control inputs and steering direction
    #     #       steeringDir = vfh(scan, targetDir)
    #     #
    #     #       # Visualize histograms using show
    #     #       show(vfh);
    #     #
    #     #       # Specify axes using Name-Value pair
    #     #       fh = figure;
    #     #       ax(1) = subplot(1,2,1);
    #     #       ax(2) = subplot(1,2,2);
    #     #       show(vfh, 'Parent', ax);
    #     #
    #     #   See also controllerPurePursuit

    #     # If step has not been called then error out
    #         if obj.IsShowBeforeStep
    #             error(message('nav:navalgs:vfh:ShowBeforeStep'));
    #         end

    #         # Plot polar obstacle density histogram
    #         if nargin > 1
    #             validatestring(name,{'Parent'}, 'show', 'Name');
    #             validateattributes(value,{'matlab.graphics.axis.Axes'},
    #                                {'numel', 2}, 'show', 'Value');
    #             axHandle1 = newplot(value(1));
    #             axHandle2 = newplot(value(2));
    #         else
    #             axHandle1 = newplot(subplot(1,2,1));
    #             axHandle2 = newplot(subplot(1,2,2));
    #         end

    #         # Preserve the hold state of axes
    #         holdState1 = ishold(axHandle1);
    #         holdState2 = ishold(axHandle2);

    #         # Prepare polar background for the histogram
    #         if max(obj.HistogramThresholds) == 0
    #             polarAxesLimit = 1;
    #         else
    #             polarAxesLimit = 4*max(obj.HistogramThresholds);
    #         end

    #         obj.preparePolarPlot(axHandle1, polarAxesLimit);

    #         polarAxesLimit = axHandle1.XLim(2);
    #         hold(axHandle1,'on');
    #         polarData = obj.PolarObstacleDensity;
    #         polarData(polarData > polarAxesLimit) =
    #             polarAxesLimit;

    #         polarData = [polarData; polarData];
    #         polarDataToPlot = polarData(:).T;

    #         # Plot the polar obstacle density
    #         [x,y] = pol2cart(obj.AngularSectorEndPoints, polarDataToPlot);
    #         c = zeros(1, length(x));
    #         cnan = nan(1, length(x));
    #         xPts = [x;c;cnan];
    #         yPts = [y;c;cnan];
    #         plot(axHandle1, -yPts(:), xPts(:),'b-');

    #         # Connect polar lines
    #         plot(axHandle1, -y, x, 'b');

    #         # Draw circles representing the thresholds
    #         [xhigh, yhigh] = obj.circle(0,0,obj.HistogramThresholds(2));
    #         [xlow, ylow] = obj.circle(0,0,obj.HistogramThresholds(1));

    #         ptL = plot(axHandle1, xlow, ylow, 'm');
    #         plot(axHandle1, xhigh, yhigh, 'm');

    #         # Add legend and title to the first plot
    #         legend(axHandle1, ptL(1), message(
    #             'nav:navalgs:vfh:HistThresholds').getString,
    #                'Location','northwest');
    #         title(axHandle1,
    #               message('nav:navalgs:vfh:PODTitle').getString);

    #         # Set the aspect ratio and reset the hold
    #         set(axHandle1, 'DataAspectRatio', [1, 1, 1]);
    #         axis(axHandle1, 'off');
    #         set(axHandle1, 'NextPlot', lower(get(axHandle1, 'NextPlot')));

    #         if ~holdState1
    #             hold(axHandle1, 'off');
    #         end

    #         set(get(axHandle1, 'XLabel'), 'Visible', 'on');
    #         set(get(axHandle1, 'YLabel'), 'Visible', 'on');

    #         # Create masked polar histogram with range sensor data

    #         # The maximum axis limit is slightly above active region
    #         axisLimit = obj.DistanceLimits(2) + 1;

    #         rangeIdx = obj.Ranges < axisLimit &
    #             (obj.Ranges >=obj.DistanceLimits(1)) &
    #             (obj.Ranges <=obj.DistanceLimits(2));

    #         # Prepare the polar plot with maximum axis limit
    #         obj.preparePolarPlot(axHandle2, axisLimit);
    #         hold(axHandle2,'on');

    #         # Plot the masked histogram
    #         maskedLength = 0.5*axisLimit;
    #         maskedHist = [obj.MaskedHistogram; obj.MaskedHistogram];
    #         maskedHistToPlot = maskedLength*maskedHist(:).T;

    #         [xMask,yMask] = pol2cart(obj.AngularSectorEndPoints,
    #                                  maskedHistToPlot);
    #         cMask = zeros(1, length(xMask));
    #         cnanMask = nan(1, length(xMask));
    #         xPtsMask = [xMask;cMask;cnanMask];
    #         yPtsMask = [yMask;cMask;cnanMask];
    #         plot(axHandle2, -yPtsMask(:), xPtsMask(:),'b-');

    #         # Connect histogram lines on periphery
    #         plot(axHandle2, -yMask, xMask, 'b');

    #         # Percentage (of axisLimit) line lengths for directions
    #         tarLength = 0.5;

    #         # Plot target and last steering direction
    #         [xTd, yTd] = pol2cart(obj.TargetDirection, tarLength*axisLimit);
    #         [xPd, yPd] = pol2cart(obj.PreviousDirection, tarLength*axisLimit);
    #         pHTar = plot(axHandle2, -[yTd,0], [xTd,0], 'm', 'LineWidth', 2.0);
    #         pHPre = plot(axHandle2, -[yPd,0], [xPd,0], 'g--', 'LineWidth', 2.0);

    #         # Plot range readings
    #         [pX,pY] = pol2cart(obj.Angles(rangeIdx), obj.Ranges(rangeIdx));
    #         pHLaser = scatter(axHandle2, -pY, pX, 5, 'r', 'filled');

    #         # Plot the active region
    #         [xAct, yAct] = obj.circle(0,0,obj.DistanceLimits(2));
    #         pHAct = plot(axHandle2, -yAct, xAct, 'black');

    #         [xAct2, yAct2] = obj.circle(0,0,obj.DistanceLimits(1));
    #         plot(axHandle2, -yAct2, xAct2, 'black');

    #         # Get legend strings
    #         targetStr =
    #             message('nav:navalgs:vfh:TargetDir').getString;
    #         steerStr =
    #             message('nav:navalgs:vfh:SteeringDir').getString;
    #         rangeStr =
    #             message('nav:navalgs:vfh:RangeReadings').getString;
    #         distLimStr =
    #             message('nav:navalgs:vfh:DistanceLimits').getString;

    #         # Set legends and title
    #         if ~isempty(pX) && ~isnan(obj.PreviousDirection)
    #             # All children exists
    #             legend(axHandle2, [pHTar, pHPre, pHAct, pHLaser],
    #                    targetStr, steerStr, distLimStr, rangeStr,
    #                    'Location','northwest');
    #         elseif ~isempty(pX) && isnan(obj.PreviousDirection)
    #             # Previous direction is nan, so not plotted
    #             legend(axHandle2, [pHTar, pHAct, pHLaser],
    #                    targetStr, distLimStr, rangeStr,
    #                    'Location','northwest');
    #         else
    #             # If no laser points then previous direction cannot be nan
    #             legend(axHandle2, [pHTar, pHPre, pHAct],
    #                    targetStr, steerStr, distLimStr, 'Location','northwest');
    #         end

    #         title(axHandle2,
    #               message('nav:navalgs:vfh:MPHTitle').getString);

    #         # Set the aspect ratio and reset the hold
    #         set(axHandle2, 'DataAspectRatio', [1, 1, 1]);
    #         axis(axHandle2, 'off');
    #         set(axHandle2, 'NextPlot', lower(get(axHandle2, 'NextPlot')));

    #         if ~holdState2
    #             hold(axHandle2, 'off');
    #         end

    #         set(get(axHandle2, 'XLabel'), 'Visible', 'on');
    #         set(get(axHandle2, 'YLabel'), 'Visible', 'on');

    #         # Only return handle if user requested it.
    #         if nargout > 0
    #             ax = [axHandle1 axHandle2];
    #         end
    #     end
    # end

    # methods (Static, Access = private)
    #     function preparePolarPlot(cax, maxrho)
    #     #preparePolarPlot Prepare polar background for polar plot

    #         hold_state = ishold(cax);
    #         # get x-axis text color so grid is in same color
    #         # get the axis gridColor
    #         axColor = get(cax, 'Color');
    #         gridAlpha = get(cax, 'GridAlpha');
    #         axGridColor = get(cax,'GridColor').*gridAlpha +
    #             axColor.*(1-gridAlpha);
    #         tc = axGridColor;
    #         ls = get(cax, 'GridLineStyle');

    #         # Hold on to current Text defaults, reset them to the
    #         # Axes' font attributes so tick marks use them.
    #         fAngle = get(cax, 'DefaultTextFontAngle');
    #         fName = get(cax, 'DefaultTextFontName');
    #         fSize = get(cax, 'DefaultTextFontSize');
    #         fWeight = get(cax, 'DefaultTextFontWeight');
    #         fUnits = get(cax, 'DefaultTextUnits');
    #         set(cax,
    #             'DefaultTextFontAngle', get(cax, 'FontAngle'),
    #             'DefaultTextFontName', get(cax, 'FontName'),
    #             'DefaultTextFontSize', get(cax, 'FontSize'),
    #             'DefaultTextFontWeight', get(cax, 'FontWeight'),
    #             'DefaultTextUnits', 'data');

    #         # only do grids if hold is off
    #         if ~hold_state
    #             # make a radial grid
    #             hold(cax, 'on');

    #             # ensure that Inf values don't enter into the limit calculation.
    #             hhh = line([-maxrho, -maxrho, maxrho, maxrho],
    #                        [-maxrho, maxrho, maxrho, -maxrho], 'Parent', cax);
    #             set(cax, 'DataAspectRatio', [1, 1, 1],
    #                      'PlotBoxAspectRatioMode', 'auto');
    #             v = [get(cax, 'XLim') get(cax, 'YLim')];
    #             ticks = sum(get(cax, 'YTick') >= 0);
    #             delete(hhh);
    #             # check radial limits and ticks
    #             rmin = 0;
    #             rmax = v(4);
    #             rticks = max(ticks - 1, 2);
    #             if rticks > 5   # see if we can reduce the number
    #                 if rem(rticks, 2) == 0
    #                     rticks = rticks / 2;
    #                 elseif rem(rticks, 3) == 0
    #                     rticks = rticks / 3;
    #                 end
    #             end

    #             # define a circle
    #             th = 0 : math.pi / 50 : 2 * math.pi;
    #             xunit = cos(th);
    #             yunit = sin(th);
    #             # now really force points on x/y axes to lie on them exactly
    #             inds = 1 : (length(th) - 1) / 4 : length(th);
    #             xunit(inds(2 : 2 : 4)) = zeros(2, 1);
    #             yunit(inds(1 : 2 : 5)) = zeros(3, 1);
    #             # plot background if necessary
    #             if ~ischar(get(cax, 'Color'))
    #                 patch('XData', xunit * rmax, 'YData', yunit * rmax,
    #                       'EdgeColor', tc, 'FaceColor', get(cax, 'Color'),
    #                       'HandleVisibility', 'off', 'Parent', cax);
    #             end

    #             # draw radial circles
    #             c82 = cos(82 * math.pi / 180);
    #             s82 = sin(82 * math.pi / 180);
    #             rinc = (rmax - rmin) / rticks;
    #             for i = (rmin + rinc) : rinc : rmax
    #                 hhh = line(xunit * i, yunit * i, 'LineStyle', ls,
    #                            'Color', tc, 'LineWidth', 1,
    #                            'HandleVisibility', 'off', 'Parent', cax);
    #                 text((i + rinc / 20) * c82, (i + rinc / 20) * s82,
    #                      ['  ' num2str(i)], 'VerticalAlignment', 'bottom',
    #                      'HandleVisibility', 'off', 'Parent', cax);
    #             end
    #             set(hhh, 'LineStyle', '-'); # Make outer circle solid

    #             # plot spokes
    #             th = (1 : 6) * 2 * math.pi / 12;
    #             cst = cos(th);
    #             snt = sin(th);
    #             cs = [-cst; cst];
    #             sn = [-snt; snt];
    #             line(rmax * cs, rmax * sn, 'LineStyle', ls, 'Color', tc,
    #                  'LineWidth', 1, 'HandleVisibility', 'off', 'Parent', cax);

    #             # annotate spokes in degrees
    #             rt = 1.1 * rmax;
    #             for i = 1 : length(th)
    #                 text(-rt * snt(i), rt * cst(i), int2str(i * 30),
    #                      'HorizontalAlignment', 'center',
    #                      'HandleVisibility', 'off', 'Parent', cax);
    #                 if i == length(th)
    #                     loc = int2str(0);
    #                 else
    #                     loc = int2str(180 + i * 30);
    #                 end
    #                 text(rt * snt(i), -rt * cst(i), loc, 'HorizontalAlignment',
    #                      'center', 'HandleVisibility', 'off', 'Parent', cax);
    #             end

    #             # set view to 2-D
    #             view(cax, 2);

    #             # set axis limits
    #             axis(cax, rmax * [-1, 1, -1.15, 1.15]);
    #         end

    #         # Reset defaults.
    #         set(cax,
    #             'DefaultTextFontAngle', fAngle ,
    #             'DefaultTextFontName', fName ,
    #             'DefaultTextFontSize', fSize,
    #             'DefaultTextFontWeight', fWeight,
    #             'DefaultTextUnits', fUnits );
    #     end

    @staticmethod
    def circle(x, y, r):
        # circle Used for drawing circles
        # th = 0 : pi / 60 : 2 * pi;
        th = np.linspace(0, 2 * math.pi, 60)
        xp = r * np.cos(th) + x
        yp = r * np.sin(th) + y
        return [xp, yp]
