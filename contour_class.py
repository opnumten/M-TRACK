"""Classes for dealing with data organized as 2D point clouds and contours.

This module provides a base class, PointSet, for simple operations on sets of
2D points. A subclass, Contour, provides specified operations for points
organized into closed contours (aka polygons). Other subclasses deal with
contour data that also includes outside landmark points (ContourAndLandmarks),
or for creating and using PCA-based shape models (PCAContour).
"""

import numpy
import scipy.interpolate.fitpack as fitpack
import copy
#import exceptions #for python 2
import builtins as exceptions#for python 3
import utility_tools as utility_tools
import procustes as procustes
#import path as path
#import numpy_compat as numpy_compat

_pi_over_2 = numpy.pi / 2

class ContourError(RuntimeError):
    pass

def _copymethod(method):
    """Return a function that makes a copy of the self object applys the method to that object, and returns the modified copy."""
    def m(self, *p, **kw):
        c = self.__class__(other = self)
        method(c, *p, **kw)
        return c
    m.__doc__ = 'Make a copy of this object, apply method %s (with the given arguments) to the copy, and then return the modified copy.' % method.__name__
    return m

class PointSet(object):
    """Manage a list of 2D points and provide basic methods to measure properties
    of those points and transform them geometrically.

    The list of points is stored in the 'points' variable, which is an array of
    size Nx2.

    Note that methods starting with 'as_' are equivalent to their similarly-named
    counterparts, except they first make a copy of the object, modify that copy,
    and return the new object. For example, q = p.as_recentered() is equivalent to:
    q = PointSet(other = p)
    q.recenter()
    """

    _instance_data = {'points' : numpy.zeros((0, 2)),
                                        'to_world_transform' : numpy.eye(3),
                                        'units' : None}

    def __init__(self, **kws):
        """Create a new PointSet (or subclass).

        The most important parameter is 'points', which must be convertible to an
        Nx2 array, specifying N data points in (x, y) format.

        Optional arguments to provide data for the new object are allowed; see the
        class's _instance_data attribute for paramter names and their default values.

        In addition, if an 'other' keyword is supplied, that parameter is assumed to
        be an other object of a compatible class, and the relevant attributes are
        copied (if possible) from the other object (AKA copy-construction). A dict
        can also be supplied for this parameter.

        If a keyword parameter and an attriubte from 'other' are both defined, the
        former has precedence. If neither are present, the default value from
        _instance_data is used.
        """
        try:
            other = kws['other']
        except:
            other = None
        for attr, value in self._instance_data.items():
            if attr in kws:
                value = kws[attr]
            elif other is not None:
                try:
                    value = getattr(other, attr)
                except:
                    try:
                        value = other[attr]
                    except:
                        pass
            if isinstance(self._instance_data[attr], numpy.ndarray):
                setattr(self, attr, numpy.array(value, copy=True, subok=True))
            else:
                setattr(self, attr, copy.deepcopy(value))
        self._filename = ''
        if other is not None:
            try:
                self._filename = other._filename
            except:
                pass

    def as_copy(self):
        """Return a copy of this object."""
        return self.__class__(other = self)

    # def simple_name(self):
    #   """Return the base name (no directories, no extension) of the file that this
    #   object was loaded from."""
    #   try:
    #     return path.path(self._filename).namebase
    #   except:
    #     return ''
    def bounding_box(self):
        """Return the bounding box of the data points as [[xmin, ymin], [xmax, ymax]]."""
        mins = self.points.min(axis = 0)
        maxes = self.points.max(axis = 0)
        return numpy.array([mins, maxes])

    def size(self):
        """Return the size of the data point bounding box [x_size, y_size]"""
        return self.points.max(axis = 0) - self.points.min(axis = 0)

    def centroid(self):
        """Return the [x, y] centroid of the data points."""
        return self.points.mean(axis = 0)

    def alignment_angle(self):
        """Return the rotation (in degrees) needed to return the contour to its
        original alignment."""
        rotate_reflect = utility_tools.decompose_homogenous_transform(self.to_world_transform)[0]
        theta = numpy.arctan2(rotate_reflect[0,1], rotate_reflect[0,0])
        return theta * 180 / numpy.pi

    def bounds_center(self):
        """Return the center point of the bounding box. Differs from the centroid
        in that the centroid is weighted by the number of points in any particular
        location."""
        mins, maxes = self.bounding_box()
        return mins + (maxes - mins) / 2.0

    def aspect_ratio(self):
        """Return the aspect ratio of the data as x_size / y_size."""
        size = self.size()
        return float(size[0]) / size[1]

    def recenter(self, center = numpy.array([0,0])):
        """Center the data points about the provided center-point, or the origin if no point is provided."""
        center = numpy.asarray(center)
        self.translate(center - self.centroid())

    def recenter_bounds(self, center = numpy.array([0,0])):
        """Center the data points' bounding-box about the provided center-point, or the origin if no point is provided."""
        center = numpy.asarray(center)
        self.translate(center - self.bounds_center())

    def transform(self, transform):
        """Transform the data points with the provided affine transform.

        The transform should be a 3x3 transform in homogenous coordinates that will
        be used to transform row vectors by pre-multiplication.
        (E.g. final_point = transform * initial_point, where final_point and
        initial_point are 3x1, and * indicates matrix multiplication.)

        The provided transform is inverted and used to update the to_world_transform
        instance variable. This variable keeps track of all of the transforms performed
        so far.
        """
        inverse = numpy.linalg.inv(transform)
        self.to_world_transform = numpy.dot(inverse, self.to_world_transform)
        self.points = utility_tools.homogenous_transform_points(self.points, transform)

    def translate(self, translation):
        """Translate the points by the given [x,y] translation."""
        self.transform(utility_tools.make_homogenous_transform(translation = translation))

    def scale(self, scale):
        """Scale the points by the provide scaling factor (either a constant or an [x_scale, y_scale] pair)."""
        self.transform(utility_tools.make_homogenous_transform(scale = scale))

    def descale(self):
        """Remove any previously-applied scaling factors. If the contour is centered
        at the origin, it will remain so; if it is centered elsewhere then the descaling
        will be applied to its current location as well."""
        rotate_reflect, scale_shear, translation = utility_tools.decompose_homogenous_transform(self.to_world_transform)
        transform = utility_tools.make_homogenous_transform(transform=scale_shear)
        self.transform(transform)

    def rotate(self, rotation, in_radians = True):
        """Rotate the points by the given rotation, which can optionally be specified in degrees."""
        if not in_radians:
            rotation = numpy.pi * rotation / 180.
        s = numpy.sin(rotation)
        c = numpy.cos(rotation)
        self.transform(utility_tools.make_homogenous_transform(transform = [[c,s],[-s,c]]))

    def to_world(self):
        """Return the points to their original ('world') coordinates, undoing all transforms."""
        self.transform(self.to_world_transform)

    def to_file(self, filename):
        """Save the object to a named file.

        The saved file is valid python code which can be executed to re-create all of the
        object's instance variables."""
        old_threshold = numpy.get_printoptions()['threshold']
        numpy.set_printoptions(threshold = numpy.inf)
        file_contents = ['cls = ("%s", "%s")\n'%(self.__class__.__module__, self.__class__.__name__)]
        for var_name in self._instance_data:
            file_contents.append('%s = \\'%var_name)
            file_contents.append(repr(getattr(self, var_name, None)))
            file_contents.append('\n')
        file_contents = '\n'.join(file_contents)
        try:
            f = open(filename, 'w')
        except Exception as e:
            raise IOError('Could not open file "%s" for saving. (Error: %s)'%(filename, e))
        f.write(file_contents)
        f.close()
        numpy.set_printoptions(threshold = old_threshold)

    def rigid_align(self, reference, weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        """Find the best rigid alignment between the data points and those of another PointSet (or subclass) object.

        By default, the alignment can include translation and rotation; reflections
        or scaling transforms can also be allowed, or translation disallowed.
        In addition, the 'weights' parameter allows different weights to be set for
        each data point.

        The best rigid alignment (least-mean-squared-distance between the data points
        and the reference points) is returned as a 3x3 homogenous transform matrix
        which operates on row-vectors.
        """
        T, c, t, new_A = procustes.procustes_alignment(self.points, reference.points, weights,
            allow_reflection, allow_scaling, allow_translation)
        self.transform(utility_tools.make_homogenous_transform(T, c, t))

    def axis_align(self):
        """Align the data points so that the major and minor axes of the best-fit ellpise are along the x and y axes, respectively."""
        self.recenter()
        u, s, vt = numpy.linalg.svd(self.points, full_matrices = 0)
        rotation = -numpy.arctan2(vt[0, 1],vt[0,0])
        # If we're rotating by more than pi/2 radians, just go the other direction.
        if rotation > _pi_over_2 or rotation < -_pi_over_2:
            rotation += numpy.pi
        self.rotate(rotation)

    def rms_distance_from(self, reference):
        """Calculate the RMSD between the data points and those of a reference object."""
        return numpy.sqrt(((self.points - reference.points)**2).mean())

    def procustes_distance_from(self, reference, apply_transform = True,
            weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        """Calculate the procustes distance between the data points and those of a reference object.

        The procustes distance is the RMSD between two point sets after the best rigid transform
        between the object (some of rotation/translation/reflection/scaling, depending on the
        parameters to this function) is taken into account. Weights for the individual points can
        also be specified. (See rigid_align for more details.)

        By default, the rigid transform is applied to the data points as a side-effect
        of calculating the procustes distance, though this can be disabled.
        """
        T, c, t, new_A = procustes.procustes_alignment(self.points, reference.points, weights,
            allow_reflection, allow_scaling, allow_translation)
        if apply_transform:
            self.transform(utility_tools.make_homogenous_transform(T, c, t))
        return numpy.sqrt(((new_A - reference.points)**2).mean())

    as_world = _copymethod(to_world)
    as_recentered = _copymethod(recenter)
    as_recentered_bounds = _copymethod(recenter_bounds)
    as_transformed = _copymethod(transform)
    as_translated = _copymethod(translate)
    as_rotated = _copymethod(rotate)
    as_scaled = _copymethod(scale)
    as_descaled = _copymethod(descale)
    as_rigid_aligned = _copymethod(rigid_align)
    as_axis_aligned = _copymethod(axis_align)


class Contour(PointSet):
    """Class for dealing with an ordered set of points that comprise a contour or polygon.

    This subclass of PointSet provides methods appropriate for closed contours.
    Internally, the contour is stored in the 'points' attribute as an Nx2 array
    of (x, y) points. Note that points[0] != points[-1]; that is, the contour is
    not explicitly closed, though this is implicitly assumed. The constructor will
    take care of any explicitly closed point data, if provided.

    Please also review the PointSet documentation for relevant details, especially
    pertaining to the __init__ method and method with 'as_' names.
    """
    _instance_data = dict(PointSet._instance_data)

    def __init__(self, **kws):
        PointSet.__init__(self, **kws)
        self._make_acyclic()

    def area(self):
        """Return the area inside of the contour."""
        return numpy.abs(self.signed_area())

    def signed_area(self):
        """Return the signed area inside of the contour.

        If the contour points wind counter-clockwise, the area is negative; otherwise
        it is positive."""
        xs = self.points[:,0]
        ys = self.points[:,1]
        y_forward = numpy.roll(ys, -1, axis = 0)
        y_backward = numpy.roll(ys, 1, axis = 0)
        return numpy.sum(xs * (y_backward - y_forward)) / 2.0

    def reverse_orientation(self):
        """Reverse the orientation of the contour from clockwise to counter-clockwise or vice-versa."""
        self.points = numpy.flipud(self.points)

    def point_range(self, begin = None, end = None):
        """Get a periodic slice of the contour points from begin to end, inclusive.

        If 'begin' is after 'end', then the slice wraps around."""
        return utility_tools.inclusive_periodic_slice(self.points, begin, end)

    def length(self, begin = None, end = None):
        """Calculate the length of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.interpoint_distances(begin, end).sum()

    def cumulative_distances(self, begin = None, end = None):
        """Calculate the cumulative distances along the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        interpoint_distances = self.interpoint_distances(begin, end)
        interpoint_distances[0] = 0
        return numpy.add.accumulate(interpoint_distances)

    def interpoint_distances(self, begin = None, end = None):
        """Calculate the distance from each point to the previous point, optionally over only the periodic slice specified by 'begin' and 'end'."""
        offsetcontour = numpy.roll(self.points, 1, axis = 0)
        return utility_tools.inclusive_periodic_slice(utility_tools.norm(self.points - offsetcontour, axis = 0), begin, end)

    def spline_derivatives(self, begin, end, derivatives=1):
        """Calculate derivative or derivatives of the contour using a spline fit,
        optionally over only the periodic slice specified by 'begin' and 'end'."""
        try:
            l = len(derivatives)
            unpack = False
        except:
            unpack = True
            derivatives = [derivatives]
        tck, uout = self.to_spline()
        points = utility_tools.inclusive_periodic_slice(range(len(self.points)), begin, end)
        ret = [numpy.transpose(fitpack.splev(points, tck, der=d)) for d in derivatives]
        if unpack:
            ret = ret[0]
        return ret

    def first_derivatives(self, begin = None, end = None):
        """Calculate the first derivatives of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.spline_derivatives(begin, end, 1)

    def second_derivatives(self, begin = None, end = None):
        """Calculate the second derivatives of the contour, optionally over only the periodic slice specified by 'begin' and 'end'."""
        return self.spline_derivatives(begin, end, 2)

    def curvatures(self, begin = None, end = None):
        """Calculate the curvatures of the contour (1/r of the osculating circle at each point), optionally over only the periodic slice specified by 'begin' and 'end'."""
        d1, d2 = self.spline_derivatives(begin, end, [1,2])
        x1 = d1[:,0]
        y1 = d1[:,1]
        x2 = d2[:,0]
        y2 = d2[:,1]
        return (x1*y2 - y1*x2) / (x1**2 + y1**2)**(3./2)

    def normalized_curvature(self, begin = None, end = None):
        """Return the mean of the absolute values of the curvatures over the given
        range, times the path length along that range. For a circle, this equals
        the angle (in radians) swept out along the range. For less smooth shapes,
        the value is higher."""
        return numpy.absolute(self.curvatures(begin, end)).mean() * self.length(begin, end)

    def inward_normals(self, positions=None):
        """Return unit-vectors facing inwards at the points specified in the
        positions variable (of all points, if not specified). Note that fractional
        positions are acceptable, as these values are calculated via spline
        interpolation."""
        if positions is None:
            positions = numpy.arange(len(self.points))
        tck, uout = self.to_spline()
        points = numpy.transpose(fitpack.splev(positions, tck))
        first_der = numpy.transpose(fitpack.splev(positions, tck, 1))
        inward_normals = numpy.empty_like(first_der)
        inward_normals[:,0] = -first_der[:,1]
        inward_normals[:,1] = first_der[:,0]
        if self.signed_area() > 0:
            inward_normals *= -1
        inward_normals /= numpy.sqrt((inward_normals**2).sum(axis=1))[:, numpy.newaxis]
        return inward_normals

    def interpolate_points(self, positions):
        """Use spline interpolation to determine the spatial positions at the
        contour positions specified (fracitonal positions are thus acceptable)."""
        tck, uout = self.to_spline()
        return numpy.transpose(fitpack.splev(positions, tck))

#  def _make_cyclic(self):
#    if not self.is_cyclic():
#      self.points = numpy.resize(self.points, [self.points.shape[0] + 1, self.points.shape[1]])
#      self.points[-1] = self.points[0]

    def _make_acyclic(self):
        """If the contour is cyclic (last point == first point), strip the last point off."""
        if numpy.alltrue(self.points[-1] == self.points[0]):
            self.points = numpy.resize(self.points, [self.points.shape[0] - 1, self.points.shape[1]])

    def offset_points(self, offset):
        """Offset the point ordering forward or backward.

        Example: if the points are offset by 1, then the old points[0] is now at points[1],
        the old points[-1] is at points[0], and so forth. This doesn't change the spatial
        position of the contour, but it changes how the points are numbered.
        """
        self.points = numpy.roll(self.points, offset, axis = 0)

    def to_spline(self, smoothing = 0, spacing_corrected = False):
        """Return the best-fit periodic parametric 3rd degree b-spline to the data points.

        The smoothing parameter is an upper-bound on the mean squared deviation between the
        data points and the points produced by the smoothed spline. By default it is 0,
        forcing an interpolating spline.

        The returned spline is valid over the parametric range [0, num_points+1], where
        the value at 0 is the same as the value at num_points+1. If spacing_corrected
        is True, then the intermediate points will be placed according to the physical
        spacing between them, which is useful in some situations like resampling.
        However, it means that the spline evalueated at position N will not necessarialy
        give the same point as contour.points[n], unless all of the points are exactly
        evenly spaced. Similarly, non-zero values of smoothing will disrupt this property
        as well.

        Two values are returned: tck and u. 'tck' is a tuple containing the spline c
        oefficients, the knots (two lists; x-knots and y-knots), and the degree of
        the spline. This 'tck' tuple can be used by the routines in scipy.fitpack.
        'u' is a list of the parameter values corresponding to the points in the range.
        """
        # the fitpack smoothing parameter is an upper-bound on the TOTAL squared deviation;
        # ours is a bound on the MEAN squared deviation. Fix the mismatch:
        l = len(self.points)
        smoothing = smoothing * l
        if spacing_corrected:
            interpoint_distances = self.interpoint_distances()
            last_to_first = interpoint_distances[0]
            interpoint_distances[0] = 0
            cumulative_distances = numpy.add.accumulate(interpoint_distances)
            u = numpy.empty(l+1, dtype=float)
            u[:-1] = cumulative_distances
            u[-1] = cumulative_distances[-1] + last_to_first
            u *= l / u[-1]
        else:
            u = numpy.arange(0, l+1)
        points = numpy.resize(self.points, [l+1, 2])
        points[-1] = points[0]
        tck, uout = fitpack.splprep(x = points.transpose(), u = u, per = True, s = smoothing)
        return tck, uout

    def to_bezier(self, match_curves_to_points = False, smooth = True):
        """Convert the contour into a sequence of cubic Bezier curves.

        NOTE: There may be fewer Bezier curves than points in the contour, if the
        contour is sufficiently smooth. To ensure that each point interval in the
        contour corresponds to a returned curve, set 'match_curves_to_points' to
        True.

        Output:
            A list of cubic Bezier curves.
            Each Bezier curve is an array of shape (4,2); thus the curve includes the
            starting point, the two control points, and the endpoint.
        """
        if smooth:
            size = self.size().max()
            s = 0.00001 * size
        else:
            s = 0
        tck, u = self.to_spline(smoothing=s)
        if match_curves_to_points:
            #to_insert = numpy.setdiff1d(u, numpy_compat.unique(tck[0]))#-------note wwk------the numpy_compat.py are removed
            to_insert = numpy.setdiff1d(u, numpy.unique(tck[0]))
            for i in to_insert:
                tck = fitpack.insert(i, tck, per = True)
        return utility_tools.b_spline_to_bezier_series(tck, per = True)


    def resample(self, num_points, smoothing = 0, max_iters = 500, min_rms_change = 1e-6, step_size = 0.2):
        """Resample the contour to the given number of points, which will be spaced as evenly as possible.

        Parameters:
            - smoothing: the smoothing parameter for the spline fit used in resampling. See the to_spline documentation.
            - max_iters: the resampled points are evenly-spaced via an iterative process. This is the maximum number of iterations.
            - min_rms_change: if the points change by this amount or less, cease iteration.
            - step_size: amount to adjust the point spacing by, in the range [0,1]. Values too small slow convergence, but
                values too large introduce ringing. 0.2-0.6 is a generally safe range.

        Returns the number of iterations and the final RMS change.
        """
        # cache functions in inner loop as local vars for faster lookup
        splev = fitpack.splev
        norm, roll, clip, mean = utility_tools.norm, numpy.roll, numpy.clip, numpy.mean
        iters = 0
        ms_change = numpy.inf
        l = len(self.points)
        tck, u = self.to_spline(smoothing, spacing_corrected = True)
        positions = numpy.linspace(0, l, num_points, endpoint = False)
        min_ms_change = min_rms_change**2
        points = numpy.transpose(splev(positions, tck))
        while (iters < max_iters and ms_change > min_ms_change):
            forward_distances = norm(points - roll(points, -1, axis = 0))
            backward_distances = norm(points - roll(points, 1, axis = 0))
            arc_spans = (roll(positions, -1, axis = 0) - roll(positions, 1, axis = 0)) % (l+1)
            deltas = forward_distances - backward_distances
            units = arc_spans / (forward_distances + backward_distances)
            steps = step_size * deltas * units
            steps[0] = 0
            positions += steps
            positions = clip(positions, 0, l+1)
            iters += 1
            ms_change = mean((steps**2))
            points = numpy.transpose(splev(positions, tck))
        self.points = points
        return iters, numpy.sqrt(ms_change)

    def global_reorder_points(self, reference):
        """Find the point ordering that best aligns (in the RMSD sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        RMSD between the data points and the corresponding reference points
        (when the data points are so offset; see the offset_points method for details).
        A global binary search strategy works well because the RMSD-as-a-function-of-offset
        landscape is smooth and sinusoidal.
        (I have not proven this, but it is empirically so for simple cases of both
        convex and concave contours. Perhaps this is not valid for self-overlapping
        polygons, however.)

        Returns the final RMSD between the points (as best ordered) and the reference points.
        """
        best_offset = 0
        step = self.points.shape[0] // 2
        while(True):
            d = self.as_offset_points(best_offset).rms_distance_from(reference)
            dp = self.as_offset_points(best_offset + 1).rms_distance_from(reference)
            dn = self.as_offset_points(best_offset - 1).rms_distance_from(reference)
            if d < dp and d < dn: break
            elif dp < dn: direction = 1
            else: direction = -1
            best_offset += direction * step
            if step > 2:
                step //= 2
            else:
                step = 1
        self.offset_points(best_offset)
        return d


    def _local_point_ordering_search(self, reference, distance_function, max_iters = None):
        """Find the point ordering that best aligns the data points to the reference object's points.

        The quality of the alignment is evaluated by the provided distance function.
        A local search strategy is employed which takes unit steps in the most
        promising direction until a distance minima is reached.

        Note that the distance function might have side-effects on this object. This
        is desirable in the case that we want to transform the points as a part
        of finding the distance (e.g. we're looking at procustes distances).

        Returns the final distance value between the data points and the reference points.
        """
        if max_iters is None:
            max_iters = len(self.points)
        d = distance_function(self, reference)
        pos = self.as_offset_points(1)
        neg = self.as_offset_points(-1)
        dp = distance_function(pos, reference)
        dn = distance_function(neg, reference)
        if d < dp and d < dn: return d
        elif dp < dn:
            contour = pos
            d = dp
            direction = 1
        else:
            contour = neg
            d = dn
            direction = -1
        iters = 0
        while iters < max_iters:
            iters += 1
            ctr = contour.as_offset_points(direction)
            dp = distance_function(ctr, reference)
            if dp > d:
                break
            else:
                contour = ctr
                d = dp
        # now copy the metadata from the best contour to self.
        self.__init__(other = contour)
        return d

    def local_reorder_points(self, reference, max_iters = None):
        """Find the point ordering that best aligns (in the RMSD sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        RMSD between the data points and the corresponding reference points
        (when the data points are so offset; see the offset_points method for details).
        A local hill_climbing search strategy is used.

        This function will be slower than global_reorder_points unless the maxima
        is closer than log2(len(points)).

        Returns the final RMSD between the data points and the reference points.
        """
        return self._local_point_ordering_search(reference, self.__class__.rms_distance_from, max_iters)

    def local_best_alignment(self, reference, weights = None, allow_reflection = False,
            allow_scaling = True, allow_translation = True, max_iters = None):
        """Find the point ordering that best aligns (in the procustes distance sense) the data points to the reference object's points.

        The 'best ordering' is defined as the offset which produces the smallest
        procustes between the data points (when the points are so offset and then
        procustes aligned to the reference points; see offset_points, rigid_align, and
        procustes_distance_from for more details). A local hill_climbing search strategy is used.

        The 'weights', 'allow_reflection', 'allow_scaling', and 'allow_translation'
        parameters are equivalent to those from the rigid_align method; which see for
        details.

        Returns the final procustes distance between the data points and the reference points.
        """
        pdf = self.__class__.procustes_distance_from
        def find_distance(contour, reference):
            return pdf(contour, reference, True, weights, allow_reflection, allow_scaling, allow_translation)
        return self._local_point_ordering_search(reference, find_distance, max_iters)

    def global_best_alignment(self, reference, align_steps = 8, weights = None, allow_reflection = False,
            allow_scaling = True, allow_translation = True, allow_reversed_orientation = True, quick = False):
        """Perform a global search for the point ordering that allows the best rigid alignment between the data points and a reference.

        The 'align_steps' parameter controls the number of offsets that are
        initially examined. The contour will be offset 'align_steps' times, evenly
        spaced. If the 'quick' parameter is False (the default), a local search is
        performed at each step to find the closest maxima; if 'quick' is True then
        the distance at that offset (and not a nearby maxima) is recorded. In
        either case, one final local search is performed to refine the best fit
        previously found.

         The 'weights', 'allow_reflection', 'allow_scaling', and
        'allow_translation' parameters are equivalent to those from the
        rigid_align method; which see for details.

         If the 'allow_reversed_orientation' parameter is true, than at each step,
        the contour ordering is reversed to see if that provides a better fit.
        This is important in trying to fit a contour to a possibly-reflected form.
        """
        offsets = numpy.linspace(0, self.points.shape[0], align_steps, endpoint = False).astype(int)
        max_iters = int(numpy.ceil(0.1 * self.points.shape[0] / align_steps))
        best_distance = numpy.inf
        for offset in offsets:
            contour = self.as_offset_points(offset)
            if quick:
                distance = contour.procustes_distance_from(reference, True, weights, allow_reflection, allow_scaling, allow_translation)
            else:
                distance = contour.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation, max_iters)
            if allow_reversed_orientation:
                rev = self.as_reversed_orientation()
                rev.offset_points(offset)
                if quick:
                    r_distance = rev.procustes_distance_from(reference, True, weights, allow_reflection, allow_scaling, allow_translation)
                else:
                    r_distance = rev.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation, max_iters)
                if r_distance < distance:
                    contour = rev
                    distance = r_distance
            if distance < best_distance:
                best_offset = offset
                best_distance = distance
                best_contour = contour
        # copy best_contour to self
        self.__init__(other = best_contour)
        # now one last align step
        return self.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)

    def find_shape_intersections(self, ray_starts, ray_ends):
        """Find the closest points of intersection with the contour and a set of
        rays. Each ray must be represented as a start point and an end point.
        For each ray, two values are returned: the relative distance along the ray
        (as a fraction of the distance from start to end; could be negative) of
        the closest point, and the relative distance of the next-closest point
        that is on the ohter side of the contour. (That is, the point between the
        two returned intersection points is guaranteed to be INSIDE the contour.)
        If there are no intersection points, nans are returned.
        If the ray is exactly tangent to the contour, the results are undefined!
        No effort has been made to handle this uncommon case correctly.
        Also, the approximate contour position of the intersections is returned
        in a second array.
        """
        s0 = self.points
        s1 = numpy.roll(s0, -1, axis = 0)
        intersections = []
        point_numbers = []
        all_points = numpy.arange(len(self.points))
        for start, end in zip(ray_starts, ray_ends):
            radii, positions = utility_tools.line_intersections(start, end, s0, s1)
            intersects_in_segment = (positions <= 1) & (positions >= 0)
            intersect_radii = radii[intersects_in_segment]
            intersect_positions = (all_points + positions)[intersects_in_segment]
            if len(intersect_radii) == 0:
                intersections.append((None, None))
                point_numbers.append((None, None))
                continue
            pos_intersects = intersect_radii >= 0
            neg_intersects = ~(pos_intersects)
            pos_vals = intersect_radii[pos_intersects]
            neg_vals = intersect_radii[neg_intersects]
            pos_ord = numpy.argsort(pos_vals)
            neg_ord = numpy.argsort(neg_vals)
            pos = pos_vals[pos_ord]
            neg = neg_vals[neg_ord]
            pos_positions = intersect_positions[pos_intersects][pos_ord]
            neg_positions = intersect_positions[neg_intersects][neg_ord]
            closest = intersect_radii[numpy.absolute(intersect_radii).argmin()]
            if len(pos) % 2 == 1:
                # ray start is inside contour
                if closest >= 0:
                    closest_p = pos_positions[0]
                    next = neg[-1]
                    next_p = neg_positions[-1]
                else:
                    closest_p = neg_positions[-1]
                    next = pos[0]
                    next_p = pos_positions[0]
            else:
                # ray start is outside contour
                if closest >= 0:
                    closest_p = pos_positions[0]
                    next = pos[1]
                    next_p = pos_positions[1]
                else:
                    closest_p = neg_positions[-1]
                    next = neg[-2]
                    next_p = neg_positions[-2]
            intersections.append((closest, next))
            point_numbers.append((closest_p, next_p))
        return numpy.array(intersections, dtype=float), numpy.array(point_numbers, dtype=float)

    def find_nearest_point(self, point):
        """Find the position, in terms of the (fractional) contour parameter, of
        the point on the contour nearest to the given point."""
        s0 = self.points
        s1 = numpy.roll(s0, -1, axis = 0)
        closest_points, positions = utility_tools.closest_point_to_lines(point, s0, s1)
        positions.clip(0, 1)
        positions += numpy.arange(len(self.points))
        square_distances = ((point[:, numpy.newaxis] - closest_points)**2).sum(axis=1)
        return positions[square_distances.argmin()]

    def find_contour_midpoints(self, p1, p2):
        """Returns the two points midway between the given points along the
        contour, and the distances (in terms of the contour parameter)
        from the first point given to the two mid-points."""
        l = self.points.shape[0]
        if p2 < p1:
            p1, p2 = p2, p1
        ca = (p2 + p1)/2
        da = ca - p1
        return (ca, (ca - l/2.)%l), (da, da - l/2.)

    as_reversed_orientation = _copymethod(reverse_orientation)
    as_offset_points = _copymethod(offset_points)
    as_resampled = _copymethod(resample)
    as_globally_reordered_points = _copymethod(global_reorder_points)
    as_locally_reordered_points = _copymethod(local_reorder_points)
    as_locally_best_alignment = _copymethod(local_best_alignment)
    as_globally_best_alignment = _copymethod(global_best_alignment)

class ContourAndLandmarks(Contour):
    """Class for dealing with contour data that also has specific landmark points
    that should be taken account of when aligning with other contours."""
    _instance_data = dict(Contour._instance_data)
    _instance_data.update({'landmarks':numpy.zeros((0, 2)), 'weights':1})

    def _pack_landmarks_into_points(self):
        """Concatenate the list of landmarks to the list of points."""
        self.points = numpy.concatenate((self.points, self.landmarks))

    def _unpack_landmarks_from_points(self):
        """Unpack the list of landmarks from the list of points."""
        num_landmarks = len(self.landmarks)
        if num_landmarks == 0:
            return
        self.landmarks = self.points[-num_landmarks:]
        self.points = self.points[:-num_landmarks]

    def _get_points_and_landmarks(self):
        """Get the points and landmarks as a single concatenated list."""
        return numpy.concatenate((self.points, self.landmarks))

    def set_weights(self, landmark_weights):
        """Set the weights associted with the landmarks.

        In a landmark contour, each point and landmark can be associated with a
        weight, such that the total weight sums to one. This function is used to
        set the weights of the landmarks.

        If a single number (or list of size 1) is provided, then this weight is
        divided among all of the landmarks, and the remaining weight is divided
        among all of the contour points. Otherwise, the provided weights must be
        a list as long as the number of landmarks; each landmark will be given the
        corresponding weight, and the remaining weight will be divided among the
        contour points."""
        num_points = len(self.points)
        num_landmarks = len(self.landmarks)
        landmark_weights = numpy.asarray(landmark_weights, dtype=float)
        try:
            l = len(landmark_weights)
        except:
            l = 1
            landmark_weights = numpy.array([landmark_weights])
        if l == 1:
            landmark_weights /= num_landmarks
            landmark_weights = numpy.ones(num_landmarks) * landmark_weights
            l = num_landmarks
        elif l != num_landmarks:
            raise ValueError('Either one weight for all landmarks must be provided, or enough weights for each. (%d required, %d found)'%(num_landmarks, l))
        total_landmark_weight = landmark_weights.sum()
        if total_landmark_weight > 1:
            raise ValueError('The total weight assigned to the landmarks must not be greater than one.')
        point_weight = (1.0 - total_landmark_weight) / num_points
        point_weights = numpy.ones(num_points) * point_weight
        self.weights = numpy.concatenate((point_weights, landmark_weights))

    def transform(self, transform):
        self._pack_landmarks_into_points()
        Contour.transform(self, transform)
        self._unpack_landmarks_from_points()
    transform.__doc__ = Contour.transform.__doc__

    def rigid_align(self, reference, weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.rigid_align(self, reference, weights, allow_reflection, allow_scaling, allow_translation)
        if weights is None:
            weights = self.weights
        self._pack_landmarks_into_points()
        reference._pack_landmarks_into_points()
        Contour.rigid_align(self, reference, weights, allow_reflection, allow_scaling, allow_translation)
        self._unpack_landmarks_from_points()
        reference._unpack_landmarks_from_points()
    rigid_align.__doc__ = Contour.rigid_align.__doc__

    def rms_distance_from(self, reference):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.rms_distance_from(self, reference)
        return numpy.sqrt(((self.weights[:, numpy.newaxis] * (self._get_points_and_landmarks() - reference._get_points_and_landmarks()))**2).mean())
    rms_distance_from.__doc__ = Contour.rms_distance_from.__doc__

    def procustes_distance_from(self, reference, apply_transform = True,
            weights = None, allow_reflection = False, allow_scaling = False, allow_translation = True):
        if not isinstance(reference, ContourAndLandmarks):
            return Contour.procustes_distance_from(self, reference, apply_transform, weights,
                allow_reflection, allow_scaling, allow_translation)
        if weights is None:
            weights = self.weights
        self._pack_landmarks_into_points()
        reference._pack_landmarks_into_points()
        ret = Contour.procustes_distance_from(self, reference, apply_transform, weights,
            allow_reflection, allow_scaling, allow_translation)
        self._unpack_landmarks_from_points()
        reference._unpack_landmarks_from_points()
        return ret
    procustes_distance_from.__doc__ = Contour.procustes_distance_from.__doc__

    as_weighted = _copymethod(set_weights)

def calculate_mean_contour(contours):
    """Calculate the average of a set of contours, while retaining units and
    scaling information, if possible. If all contours have associated landmarks,
    then the average will be such a contour as well."""
    all_points = [c.points for c in contours]
    if not utility_tools.all_same_shape(all_points):
        raise ContourError('Cannot calculate mean of contours with different numbers of points.')
    mean_points = numpy.mean(all_points, axis=0)
    units = [c.units for c in contours]
    if not numpy.alltrue([u == units[0] for u in units]):
        raise ContourError('All contours must have the same units in order calculate their mean.')
    units = contours[0].units
    scales = [utility_tools.decompose_homogenous_transform(c.to_world_transform)[1] for c in contours]
    if numpy.alltrue([numpy.allclose(scales[0], s) for s in scales[1:]]):
        transform = utility_tools.make_homogenous_transform(transform=scales[0])
    else:
        transform = numpy.eye(3)
    if numpy.alltrue([isinstance(c, ContourAndLandmarks) for c in contours]):
        # if they're all landmark'd contours
        all_landmarks = [c.landmarks for c in contours]
        if not utility_tools.all_same_shape(all_landmarks):
            raise ContourError('Cannot calculate mean of contours with different numbers of landmarks.')
        mean_landmarks = numpy.mean(all_landmarks, axis=0)
        mean_weights = numpy.mean([c.weights for c in contours], axis=0)
        return ContourAndLandmarks(points=mean_points, units=units, landmarks=mean_landmarks,
            weights=mean_weights, to_world_transform=transform)
    else:
        return Contour(points=mean_points, units=units, to_world_transform=transform)
