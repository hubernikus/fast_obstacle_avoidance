"""
Inspired by the Matlab class HistogramBase
DISCLAIMER: we do not own the copyrights for this function (!)

"""
import copy
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Scan:
    Ranges: np.ndarray
    Angles: np.ndarray

    @property
    def num_scan(self):
        return self.Angles.shape[0]

    def removeInvalidData(self, distance_limits):
        scan = copy.deepcopy(self)

        ind_valid = scan.Ranges > distance_limits[0]
        scan.Ranges = scan.Ranges[ind_valid]
        scan.Angles = scan.Angles[ind_valid]

        ind_valid = scan.Ranges < distance_limits[1]
        scan.Ranges = scan.Ranges[ind_valid]
        scan.Angles = scan.Angles[ind_valid]

        return scan


def wrap_to_pi(angle: [float, np.ndarray]) -> [float, np.ndarray]:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_difference(angle1, angle2):
    angle_diff = angle1 - angle2
    return wrap_to_pi(angle_diff)


def bisectAngles(theta1, theta2):
    # This function is for internal use only. It may be removed in the future.

    # bisectAngles Find bisection angle between two angles
    #
    #   BISECT = bisectAngles(THETA1, THETA2) computes bisecting angle between
    #   two angles THETA1 and THETA2 and returns the BISECT within the interval
    #   [-pi pi]. Positive, odd multiples of pi map to pi and negative, odd
    #   multiples of pi map to -pi.
    theta1 = wrap_to_pi(theta1)
    theta2 = wrap_to_pi(theta2)

    # Get angle bisection
    deltaAng = theta1 - theta2
    angle = theta1 - deltaAng / 2.0

    # Make sure the output is in the [-pi,pi) range
    return wrap_to_pi(angle)


def angle_difference_abs(angle1, angle2):
    angle_diff = np.abs(angle1 - angle2)

    ind_ang = angle_diff > 2 * math.pi
    if np.sum(ind_ang):
        angle_diff = angle_diff - 2 * math.pi

    return angle_diff


class HistogramBase:
    """
    #This class is for internal use only. It may be removed in the future.

    #VectorFieldHistogramBase Base class for controllerVFH implementation in MATLAB and in Simulink
    #
    #   See also controllerVFH, nav.slalgs.internal.VectorFieldHistogram.

    #   Copyright 2017-2019 The MathWorks, Inc.
    """

    ##codegen
    def __init__(
        self, NumAngularSectors=180, HistogramThresholds=(3, 10), RobotRadius=0.1
    ):
        # properties (Nontunable)
        # NumAngularSectors Number of angular sectors
        #   The number of angular sectors are the number of bins used to
        #   create histograms
        #
        #   Default: 180
        self.NumAngularSectors = NumAngularSectors

        # properties (Hidden)
        # NarrowOpeningThreshold Angular threshold in radians
        #   This is an angular threshold, specified in radians, to consider
        #   an angular region to be narrow. The algorithm selects one
        #   candidate direction for each narrow region, while it selects
        #   two candidate directions in each non-narrow region.
        #
        #   Default: 0.8
        self.NarrowOpeningThreshold = 0.8

        # properties
        # DistanceLimits Range distance limits (m)
        #   The range readings specified in the step function input are
        #   considered only if they fall within the distance limits.
        #   The lower distance limit is specified to ignore false
        #   positive data while the higher distance limit is specified
        #   to ignore obstacles that are too far from the vehicle.
        #
        #   Default: [0.05 2]
        self.DistanceLimits = [0.05, 2]

        # RobotRadius Vehicle radius (m)
        #   This is radius of the smallest circle that can circumscribe the
        #   vehicle geometry. The vehicle radius is used to account for vehicle
        #   size in the computation of the obstacle-free direction.
        #
        #   Default: 0.1
        self.RobotRadius = RobotRadius

        # SafetyDistance Safety distance (m)
        #   This is a safety distance to leave around the vehicle position in
        #   addition to the RobotRadius. The vehicle radius and safety
        #   distance are used in the computation of the obstacle-free
        #   direction.
        #
        #   Default: 0.1
        self.SafetyDistance = 0.1

        # MinTurningRadius Minimum turning radius (m)
        #   This is the minimum turning radius with which the vehicle can turn
        #   while moving at its current speed.
        #
        #   Default: 0.1
        self.MinTurningRadius = 0.1

        # TargetDirectionWeight Target direction weight
        #   This is the cost function weight for moving towards the target
        #   direction. To follow a target direction, the
        #   TargetDirectionWeight should be higher than the sum of
        #   CurrentDirectionWeight and PreviousDirectionWeight. You can
        #   ignore the target direction cost by setting this weight to zero.
        #
        #   Default: 5
        self.TargetDirectionWeight = 5

        # CurrentDirectionWeight Current direction weight
        #   This is the cost function weight for moving in the current
        #   heading direction. Higher values of this weight produces
        #   efficient paths. You can ignore the current direction cost
        #   by setting this weight to zero.
        #
        #   Default: 2
        self.CurrentDirectionWeight = 2

        # PreviousDirectionWeight Previous direction weight
        #   This is the cost function weight for moving in the previously
        #   selected steering direction. Higher values of this weight
        #   produces smoother paths. You can ignore the previous direction
        #   cost by setting this weight to zero.
        #
        #   Default: 2
        self.PreviousDirectionWeight = 2

        # HistogramThresholds Histogram thresholds
        #   These thresholds are used to compute the binary histogram from
        #   the polar obstacle density. Polar obstacle density values higher
        #   than the upper threshold are considered to be occupied (1) in
        #   the binary histogram. Polar obstacle density values smaller
        #   than the lower threshold are considered to be free space (0) in
        #   the binary histogram. The values of polar obstacle density that
        #   fall between the upper and lower thresholds are determined by
        #   the previous binary histogram, with default being free space (0).
        #
        #   Default: [3 10]
        self.HistogramThresholds = HistogramThresholds

        # properties(Access = {?nav.algs.internal.VectorFieldHistogramBase,
        #                      ?matlab.unittest.TestCase})
        #     #PolarObstacleDensity Polar obstacle density histogram
        #     PolarObstacleDensity

        #     #BinaryHistogram Binary polar histogram
        self.BinaryHistogram = np.zeros(self.NumAngularSectors)

        #     #MaskedHistogram Masked polar histogram
        #     MaskedHistogram

        #     #PreviousDirection Steering direction output of the last step call
        #     #
        #     #   Default: 0
        #     PreviousDirection

        #     #TargetDirection Target direction specified in the last step call
        #     #
        #     #   Default: 0
        #     TargetDirection

        #     #AngularSectorMidPoints Angular sectors in radians
        #     #   These are the angular sectors determined based on the angular
        #     #   limits and the number of angular sectors.
        #     AngularSectorMidPoints

        #     #AngularDifference Size of each angular sector
        #     AngularDifference

        #     #AngularSectorStartPoints Start points of angular sectors
        #     AngularSectorStartPoints

        #     #AngularSectorEndPoints Start and end points for angular sectors
        #     AngularSectorEndPoints

        # properties (Access = protected)
        #     #Ranges Range sensor reading from the last step call
        #     Ranges

        #     #Angles Angles corresponding to ranges from the last step call
        #     Angles

        # AngularLimits Minimum and maximum angular limits in radians
        #   A vector [MIN MAX] representing the angular limits to consider
        #   as candidate directions. This is usually the angular limits of
        #   the range sensor. If an empty value is assigned, the angular
        #   limits will be computed from the input angles in the step
        #   function.
        #
        #   Default: [-pi, pi]
        self.AngularLimits = [-math.pi, math.pi]

        # # methods (Access = protected)
        #     function loadObjectImpl(obj, svObj, wasLocked)
        #     #loadObjectImpl Custom load implementation

        #         mco = ?nav.algs.internal.VectorFieldHistogramBase;
        #         propList = mco.PropertyList;

        #         # Re-load all protected properties
        #         for i = 1:length(propList)
        #             propName = mco.PropertyList(i).Name;
        #             if robotics.internal.isProtectedProperty(mco, propList(i)) &&
        #                     isfield(svObj, propName)
        #                 obj.(propName) = svObj.(propName);

        #         # Call base class method
        #         loadObjectImpl@matlab.System(obj,svObj,wasLocked);

        #     function s = saveObjectImpl(obj)
        #     #saveObjectImpl Custom save implementation
        #         s = saveObjectImpl@matlab.System(obj);

        #         # Save all protected properties
        #         mco = ?nav.algs.internal.VectorFieldHistogramBase;
        #         propList = mco.PropertyList;

        #         for i = 1:length(propList)
        #             propName = mco.PropertyList(i).Name;
        #             if robotics.internal.isProtectedProperty(mco, propList(i))
        #                 s.(propName) = obj.(propName);

        #     function outFixedSize = isOutputFixedSizeImpl(~)
        #     #isOutputFixedSizeImpl Return True for each output port with fixed size

        #     # Steering direction is fixed size
        #         outFixedSize = True;

        #     function resetImpl(obj)
        #     #resetImpl Reset internal states

        #         obj.BinaryHistogram = false(1, obj.NumAngularSectors);
        #         obj.PreviousDirection = 0*obj.PreviousDirection;

        #     function num = getNumOutputsImpl(~)
        #     #getNumOutputsImpl Define number of outputs for system with optional outputs
        #         num = 1;

        # methods
        #     function set.DistanceLimits(obj, val)
        #         validateNonnegativeArray(val, 'DistanceLimits');
        #         obj.DistanceLimits = [min(val) max(val)];

        #     function set.NarrowOpeningThreshold(obj, val)
        #         validateattributes(val, {'double'}, {'nonnan', 'real',
        #                             'scalar', 'positive', 'finite'}, 'controllerVFH',
        #                            'NarrowOpeningThreshold');
        #         obj.NarrowOpeningThreshold = val;

        #     function set.RobotRadius(obj, val)
        #         validateNonnegativeScalar(val, 'RobotRadius');
        #         obj.RobotRadius = val;

        #     function set.SafetyDistance(obj, val)
        #         validateNonnegativeScalar(val, 'SafetyDistance');
        #         obj.SafetyDistance = val;

        #     function set.MinTurningRadius(obj, val)
        #         validateNonnegativeScalar(val, 'MinTurningRadius');
        #         obj.MinTurningRadius = val;

        #     function set.TargetDirectionWeight(obj, val)
        #         validateNonnegativeScalar(val, 'TargetDirectionWeight');
        #         obj.TargetDirectionWeight = val;

        #     function set.CurrentDirectionWeight(obj, val)
        #         validateNonnegativeScalar(val, 'CurrentDirectionWeight');
        #         obj.CurrentDirectionWeight = val;

        #     function set.PreviousDirectionWeight(obj, val)
        #         validateNonnegativeScalar(val, 'PreviousDirectionWeight');
        #         obj.PreviousDirectionWeight = val;

        #     function set.HistogramThresholds(obj, val)
        #         validateNonnegativeArray(val, 'HistogramThresholds');
        #         obj.HistogramThresholds = [min(val) max(val)];

        #     function set.NumAngularSectors(obj, val)
        #         validateattributes(val, {'double', 'single'}, {'nonnan', 'integer',
        #                             'scalar', 'positive', 'finite'}, 'controllerVFH',
        #                            'NumAngularSectors');
        #         obj.NumAngularSectors = val;

        #     function val = get.NumAngularSectors(obj)
        #         val = obj.NumAngularSectors;

        # # Abstract methods that should be implemented by all derived
        # # classes.
        # methods (Abstract, Access = protected)
        #     [scan, target, classOfRanges] = parseAndValidateStepInputs(varargin);

        # def setup(self):
        # methods (Access = protected)
        #     function setupImpl(obj, varargin)
        #     #setupImpl Setup for the system object

        #         [~, ~, classOfRanges] = obj.parseAndValidateStepInputs(varargin{:});

        self.PreviousDirection = 0
        angularLimits = self.AngularLimits
        numAngularSectors = self.NumAngularSectors

        # Create angular sectors
        self.AngularSectorMidPoints = np.linspace(
            angularLimits[0] + math.pi / numAngularSectors,
            angularLimits[1] - math.pi / numAngularSectors,
            numAngularSectors,
        )

        if numAngularSectors > 1:
            self.AngularDifference = abs(
                angle_difference(
                    self.AngularSectorMidPoints[0], self.AngularSectorMidPoints[1]
                )
            )
        else:
            self.AngularDifference = 2 * math.pi

        self.AngularSectorStartPoints = (
            self.AngularSectorMidPoints - self.AngularDifference / 2
        )

        sectorEndPoints = self.AngularSectorMidPoints + self.AngularDifference / 2

        sectorPoints = [self.AngularSectorStartPoints, sectorEndPoints]
        self.AngularSectorEndPoints = np.array(sectorPoints).reshape(-1)

        # Pre-allocate the histogram
        self.BinaryHistogram = np.zeros(self.NumAngularSectors)

    # def stepImpl(self, varargin):
    def __call__(self, ranges, angles, target_dir):
        # step Compute control commands and steering directiona
        #   STEERINGDIR = step(VFH, RANGES, ANGLES, TARGETDIR) finds an obstacle
        #   free steering direction STEERINGDIR, using the VFH+ algorithm for
        #   input vectors RANGES and ANGLES of the same number of elements, and
        #   scalar input TARGETDIR. The input RANGES are in meters, the
        #   ANGLES and TARGETDIR are in radians. The output STEERINGDIR is in
        #   radians. The vehicle's forward direction is considered zero radians.
        #   The angles measured clockwise from the forward direction are negative
        #   angles and angles measured counter-clockwise from the forward direction
        #   are positive angles.
        #
        #   Supported syntax
        #   vfh(ranges, angles, target)
        #   vfh(lidarscanObj, target)

        # [scan, target, classOfRanges] = self.parseAndValidateStepInputs(varargin{:})

        scan = Scan(Ranges=ranges, Angles=angles)

        target_dir = wrap_to_pi(target_dir)

        # Compute theta steer
        self.buildPolarObstacleDensity(scan)
        self.buildBinaryHistogram()
        self.buildMaskedPolarHistogram(scan)

        steeringDir = self.selectHeadingDirection(target_dir)
        return steeringDir

    def buildPolarObstacleDensity(self, scan: Scan):
        # buildPolarObstacleDensity Create polar obstacle density
        #   This function creates a polar obstacle density histogram
        #   from the range readings taking into account the vehicle
        #   radius and safety distance.

        validScan = scan.removeInvalidData(self.DistanceLimits)

        # Constants A and B used in Reference [1]
        # constB = cast(1, classRanges)
        # constA = cast(self.DistanceLimits(2), classRanges)
        constB = 1
        constA = self.DistanceLimits[1]

        # Weighted ranges
        weightedRanges = constA - constB * validScan.Ranges

        # # If empty space in front of the vehicle
        # -> deactivated as the sub-function is defined differently
        # if validScan.num_scan == scan.num_scan:
        #     self.PolarObstacleDensity = np.zeros((1, self.NumAngularSectors))
        #     return

        # Special case of one sector
        if self.NumAngularSectors == 1:
            validWeights = np.ones_like(validScan.Ranges)
            self.PolarObstacleDensity = (validWeights * weightedRanges).T

        # If vehicle radius and safety distance both are zero, then use
        # primary histogram
        if self.RobotRadius + self.SafetyDistance == 0:
            # _ , bin  = histc(validScan.Angles, self.AngularSectorMidPoints); ##ok<HISTC>
            bins = np.histcounts(validScan.Angles, self.AngularSectorMidPoints)

            obstacleDensity = zeros((self.NumAngularSectors))
            for i in range(len(bins)):
                obstacleDensity[bins[i]] = obstacleDensity[bin[i]] + weightedRanges[i]

            self.PolarObstacleDensity = obstacleDensity
            return

        # Equation (4) in Reference [1]
        # If the vehicle radius + safety distance is larger than the
        # Ranges, then "ASIN" will give complex values.

        # Using pi/2 as enlargement angle for ranges that are below
        # RobotRadius+SafetyDistance. In the original VFH algorithm
        # this case is not handled.
        # sinOfEnlargement = cast(self.RobotRadius +
        #                         self.SafetyDistance, classRanges)./validScan.Ranges;
        sinOfEnlargement = (self.RobotRadius + self.SafetyDistance) / validScan.Ranges

        # Using 1 - eps, which results in enlargement angles approximately
        # sqrt(eps) smaller than pi/2. This is required because floating point
        # errors can cause nondeterministic behavior in the downstream
        # computation at enlargement angle of pi/2.
        sinOfEnlargement[sinOfEnlargement >= 1] = 1 - np.finfo(np.float64).eps

        enlargementAngle = np.arcsin(sinOfEnlargement)

        # Polar obstacle density computation
        # Equation (5)-(6) in Reference [1]
        higherAng = validScan.Angles + enlargementAngle
        lowerAng = validScan.Angles - enlargementAngle

        # Compute if a sector is within enlarged angle of a range
        # reading
        # If A X B, A X N and N X B have the same sign for Z-dimension,
        # then vector N is in between vectors A and B.

        # Create vectors for cross product computation
        lowerVec = np.vstack(
            (np.cos(lowerAng), np.sin(lowerAng), np.zeros_like(lowerAng))
        ).T

        higherVec = np.vstack(
            (np.cos(higherAng), np.sin(higherAng), np.zeros_like(higherAng))
        ).T

        validWeights = np.ones((self.NumAngularSectors, lowerVec.shape[0]), dtype=bool)
        lh = np.cross(lowerVec, higherVec)
        kalpha = np.vstack(
            (
                np.cos(self.AngularSectorMidPoints),
                np.sin(self.AngularSectorMidPoints),
                np.zeros(self.NumAngularSectors).T,
            )
        ).T

        for i in range(self.NumAngularSectors):
            kalphaVec = np.tile(kalpha[i, :], (lowerVec.shape[0], 1))
            lk = np.cross(lowerVec, kalphaVec)
            kh = np.cross(kalphaVec, higherVec)
            validWeights[i, :] = (
                abs(
                    np.copysign(1, lk[:, 2])
                    + np.copysign(1, kh[:, 2])
                    + np.copysign(1, lh[:, 2])
                )
                > 1
            )

        self.PolarObstacleDensity = validWeights @ weightedRanges

    def buildBinaryHistogram(self):
        # buildBinaryHistogram Create binary histogram
        #   This function creates a binary polar histogram using the
        #   polar obstacle density. The function uses two threshold
        #   values to determine the binary values. The values falling
        #   in between the two threshold are chosen from binary
        #   histogram from the previous step.

        # Using thresholds, determine binary histogram
        # Equation (7) in Reference [1]
        # True means occupied sector
        self.BinaryHistogram[
            self.PolarObstacleDensity > self.HistogramThresholds[1]
        ] = True
        self.BinaryHistogram[
            self.PolarObstacleDensity < self.HistogramThresholds[0]
        ] = False

    def buildMaskedPolarHistogram(self, scan):
        # buildMaskedPolarHistogram Create masked histogram
        #   This function creates the masked polar histogram from the
        #   binary histogram. It considers the vehicle's turning radius
        #   and evaluates if the obstacles are too close restricting
        #   the vehicle movement towards certain direction.

        # Angle ahead =  0    rad
        # Angle left  =  pi/2 rad
        # Angle right = -pi/2 rad

        # Equation (8) in Reference [1]
        # DXr = cast(0, classRanges);
        # DYr = cast(-self.MinTurningRadius, classRanges);
        # DYl = cast(self.MinTurningRadius, classRanges);
        # DXl = cast(0, classRanges);

        DXr = 0
        DYr = -1 * self.MinTurningRadius
        DYl = self.MinTurningRadius
        DXl = 0

        # Only consider indices in active region
        # find function always returns double output, hence it is
        # required to cast it.
        # validScan = removeInvalidData(
        #     scan, 'RangeLimits',
        #     cast([self.DistanceLimits(1) self.DistanceLimits(2)], 'like', scan.Ranges));
        validScan = scan.removeInvalidData(self.DistanceLimits)

        # Equation (9) in Reference [1]
        DXj = (validScan.Ranges) * np.cos(validScan.Angles)
        DYj = (validScan.Ranges) * np.sin(validScan.Angles)

        distR = np.sqrt((DXr - DXj) ** 2 + (DYr - DYj) ** 2)
        distL = np.sqrt((DXl - DXj) ** 2 + (DYl - DYj) ** 2)

        # Equation (10a)-(10b) in Reference [1]
        blockedR = np.logical_and(
            distR < (self.MinTurningRadius + self.RobotRadius + self.SafetyDistance),
            (validScan.Angles <= 0),
        )
        blockedL = np.logical_and(
            distL < (self.MinTurningRadius + self.RobotRadius + self.SafetyDistance),
            (validScan.Angles >= 0),
        )

        # Compute limit angles
        phiR_ind = np.argwhere(blockedR)
        if not len(phiR_ind):
            phiR = self.AngularSectorMidPoints[0]

        else:
            phiR = validScan.Angles[phiR_ind[-1]]
            if phiR[0] <= self.AngularSectorMidPoints[0]:
                # Account for point inside first sector
                phiR = self.AngularSectorMidPoints[0]

        # phiL = validScan.Angles[np.argwhere(blockedL)[0]]
        phiL_ind = np.argwhere(blockedL)
        if not len(phiL_ind):
            phiL = self.AngularSectorMidPoints[-1]

        else:
            phiL = validScan.Angles[phiL_ind[0]]
            if phiL[0] >= self.AngularSectorMidPoints[-1]:
                # Account for point inside last sector
                phiL = self.AngularSectorMidPoints[-2]

        # Equation (11) in Reference [1]
        occupiedAngularSectors = np.logical_or(
            self.AngularSectorMidPoints
            < phiR * np.ones(self.AngularSectorMidPoints.shape),
            self.AngularSectorMidPoints
            > phiL * np.ones(self.AngularSectorMidPoints.shape),
        )
        self.MaskedHistogram = np.logical_or(
            self.BinaryHistogram, occupiedAngularSectors
        )

    def selectHeadingDirection(self, targetDir):
        # selectHeadingDirection Select heading direction
        #   This function selects the heading direction based on a
        #   target direction using a cost function. It first computes
        #   the candidate directions based on the empty sectors in the
        #   masked histogram and then selects one or two candidate
        #   directions for each sector.

        # Find open sectors
        changes = np.diff(
            np.hstack((0, np.logical_not(self.MaskedHistogram), 0))
        ).flatten()

        # Skip everything if there are no open sectors
        if not np.sum(np.abs(changes)):
            self.PreviousDirection = None
            return

        foundSectors = np.argwhere(changes).flatten()

        # Because masked histogram is binary, the foundSectors will
        # always have even elements.
        sectors = foundSectors.reshape((2, -1), order="F")
        sectors[1, :] = sectors[1, :] - np.ones(sectors.shape[1])

        # Get size of different sectors
        angles = np.zeros(sectors.shape, dtype=float)
        angles[0, :] = self.AngularSectorMidPoints[sectors[0, :]]
        angles[1, :] = self.AngularSectorMidPoints[sectors[1, :]]

        sectorAngles = np.reshape(angles, (2, -1))
        sectorSizes = self.AngularDifference * np.diff(sectors, axis=0).flatten()

        # Compute one candidate direction for each narrow sector
        # Equation (12) in Reference [1]
        narrowIdx = sectorSizes < self.NarrowOpeningThreshold * np.ones(
            sectorSizes.shape[0]
        )

        narrowDirs = bisectAngles(
            sectorAngles[0, narrowIdx], sectorAngles[1, narrowIdx]
        )

        ind_notidx = np.logical_not(narrowIdx)
        # Compute two candidates for each non-narrow sector
        # Equation (13) in Reference [1]
        nonNarrowDirs = np.hstack(
            (
                sectorAngles[0, ind_notidx]
                + np.ones(sectorAngles[0, ind_notidx].shape)
                * self.NarrowOpeningThreshold
                / 2,
                sectorAngles[1, ind_notidx]
                - np.ones((sectorAngles[1, ind_notidx].shape))
                * self.NarrowOpeningThreshold
                / 2,
            )
        )

        # Add target, current and previous directions as candidates
        self.TargetDirection = targetDir
        currDir = 0
        if self.PreviousDirection is None:
            self.PreviousDirection = currDir

        # Final list of candidate directions
        # Equation (14) in Reference [1]
        candidateDirs = np.hstack(
            (
                nonNarrowDirs,
                narrowDirs,
                targetDir,
                currDir,
                self.PreviousDirection,
            )
        )

        # Remove occupied directions
        # If the candidate direction falls at the center of two bins
        # then check both the bins for occupancy
        tolerance = math.sqrt(np.finfo(np.float64).eps)

        candToSectDiff = np.abs(
            angle_difference(
                np.tile(self.AngularSectorMidPoints, (candidateDirs.shape[0], 1)),
                np.tile(candidateDirs, (self.AngularSectorMidPoints.shape[0], 1)).T,
            )
        )

        # tempDiff = bsxfun(@minus, candToSectDiff, min(candToSectDiff,[],2));
        tempDiff = (
            candToSectDiff
            - np.tile(np.min(candToSectDiff, axis=1), (candToSectDiff.shape[1], 1)).T
        )

        nearIdx = tempDiff < tolerance

        freeDirs = np.ones(nearIdx.shape[0], dtype=bool)
        for i in range(len(freeDirs)):
            freeDirs[i] = not any(self.MaskedHistogram[nearIdx[i, :]])

        candidateDirections = candidateDirs[freeDirs]

        # Compute cost for each candidate direction
        # Equation (15) in Reference [1]
        costValues = self.computeCost(
            candidateDirections, targetDir, currDir, self.PreviousDirection
        )

        # Decide best direction to steer
        cVal = np.min(costValues)

        # Consider all costs that have very small difference to min
        # value
        cDiff = costValues - cVal
        minCostIdx = cDiff < tolerance

        if not np.sum(minCostIdx):
            # if not len(thetaSteer):
            # thetaSteer = cast(nan, "like", targetDir)
            return targetDir

        thetaSteer = np.min(candidateDirections[minCostIdx])
        self.PreviousDirection = thetaSteer

        return thetaSteer

    def computeCost(self, c, targetDir, currDir, prevDir):
        # computeCost Compute total cost using all cost components

        # tdWeight = cast(self.TargetDirectionWeight, 'like', targetDir);
        # cdWeight = cast(self.CurrentDirectionWeight, 'like', targetDir);
        # pdWeight = cast(self.PreviousDirectionWeight, 'like', targetDir);
        tdWeight = self.TargetDirectionWeight
        cdWeight = self.CurrentDirectionWeight
        pdWeight = self.PreviousDirectionWeight

        totalWeight = tdWeight + cdWeight + pdWeight

        targetDir = targetDir * np.ones_like(c)
        currDir = currDir * np.ones_like(c)
        prevDir = prevDir * np.ones_like(c)

        cost = (
            (
                tdWeight * self.localCost(c, targetDir)
                + cdWeight * self.localCost(c, currDir)
                + pdWeight * self.localCost(c, prevDir)
            )
            / 3
            * totalWeight
        )

        return cost

    def localCost(self, candidateDir, selectDir):
        # localCost: Compute cost for each cost component (for valid candidate indices)
        return np.abs(angle_difference(candidateDir, selectDir))


def validateNonnegativeScalar(val, name):
    # validateNonnegativeScalar Validate non-negative real scalar
    validateattributes(
        val,
        {"double", "single"},
        {"nonnan", "real", "scalar", "nonnegative", "finite"},
        "controllerVFH",
        name,
    )


def validateNonnegativeArray(val, name):
    # validateNonnegativeArray Validate non-negative two element array
    validateattributes(
        val,
        {"double", "single"},
        {"nonnan", "real", "numel", 2, "finite", "nonnegative"},
        "controllerVFH",
        name,
    )
