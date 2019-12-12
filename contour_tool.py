import contour_class
import utility_tools as utility_tools
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
import numpy
import pandas as pd
from skimage import measure
from skimage.io import imread
from skimage.morphology import opening,closing

def find_contour_points(img_path,img_list,contour_value=0.5):
#     contour_points=[]
#     obj_arr=[]
    contour_points_and_obj=[]
    k=0
    while (k<len(img_list)):
        img=numpy.array(imread(img_path +'/'+ img_list[k]))    
        img=opening(img)
        #img=measure.label(img,connectivity=2)
        rps=measure.regionprops(img)
        r_labels=[r.label for r in rps]
        for label in r_labels:
            single_obj_img=(img==label)
#             print(label)
#             plt.imshow(single_obj_img)
#             plt.show()
            single_contour=measure.find_contours(single_obj_img,level=contour_value,fully_connected='low',positive_orientation='low')
            #print(len(single_contour))
            max_len=0
            i=0
            for i in range(len(single_contour)):
                if len(single_contour[i])>=max_len:
                    maj_i=i
                    max_len=len(single_contour[i])
#             contour_points.append(single_contour[maj_i])#need append the element of in single_contour instead of the whole array
#             obj_arr.append([k+1,label])
            
            contour_points_and_obj.append((single_contour[maj_i],[k+1,label]))
        k+=1
    return contour_points_and_obj


def df_find_contour_points(df,img_path,img_list,contour_value=0.5):
#     contour_points=[]
#     obj_arr=[]
    contour_points_and_obj=[]
    k=0
    while (k<len(img_list)):
        img=numpy.array(imread(img_path +'/'+ img_list[k]))    
        img=opening(img)
        #img=measure.label(img,connectivity=2)
        rps=measure.regionprops(img)
        r_labels=[r.label for r in rps]
        for label in r_labels:
            if numpy.asscalar(df.loc[(df['ImageNumber']==k+1)&(df['ObjectNumber']==label),'Cell_TrackObjects_Label'].values)!=-1:
                single_obj_img=(img==label)
                single_contour=measure.find_contours(single_obj_img,level=contour_value,fully_connected='low',positive_orientation='low')
                #print(len(single_contour))
                max_len=0
                i=0
                for i in range(len(single_contour)):
                    if len(single_contour[i])>=max_len:
                        maj_i=i
                        max_len=len(single_contour[i])
#                 contour_points.append(single_contour[maj_i])#need append the element of in single_contour instead of the whole array
#                 obj_arr.append([k+1,label])
                contour_points_and_obj.append((single_contour[maj_i],[k+1,label]))
        k+=1
    return contour_points_and_obj
    
def generate_contours(contour_points_and_obj,closed_only = True, min_area = None, max_area = None, axis_align = False):
    """Find the contours at a given image intensity level from an image.
    If multiple contours are found, they are returned in order of increasing area.

    Parameters:
    - closed_only: if True, then contours which touch the image edge (and thus
        are not cyclic) are discarded.
    - min_area, max_area: Minimum and maximum area (in pixels) of returned
        contours. Others will be discarded.
    - axis_align: if True, each contour will be aligned along its major axis.
    """


    if closed_only:
        contour_points_and_obj = [(p,obj) for p,obj in contour_points_and_obj if numpy.allclose(p[-1], p[0])]

    contours_and_obj = [(contour_class.Contour(points = p, units = 'pixels'),obj) for p,obj in contour_points_and_obj]
    areas_and_contours = []
    for c in contours_and_obj:
        area = c[0].signed_area()
        if area > 0:
            # Keep contours oriented in traditional (negative, counter-clockwise) orientation
            c[0].reverse_orientation()
            area = -area
        areas_and_contours.append((-area, c))
    if min_area is not None:
        areas_and_contours = [(a, c) for a, c in areas_and_contours if a >= min_area]
    if max_area is not None:
        areas_and_contours = [(a, c) for a, c in areas_and_contours if a <= max_area]
    if axis_align:
        for a, c in areas_and_contours:
            c[0].axis_align()
    areas_and_contours.sort(key=lambda x: x[0])
    sort_contours_and_obj=[c for a, c in areas_and_contours]
    contours=[c for c,obj in sort_contours_and_obj]
    sort_obj_arr=numpy.array([obj for c,obj in sort_contours_and_obj])
    return contours,sort_obj_arr


def _should_allow_reverse(contours, allow_reflection):
    # If we are to allow for reflections, we ought to allow for reversing
    # orientations too, because even if the contours start out oriented in the
    # same direction, reflections can change that.
    # Then check if all of the contours are oriented in the same direction.
    # If they're not, we need to allow for reversing their orientation in the
    # alignment process.
    if allow_reflection:
         return True
    orientations = numpy.array([numpy.sign(contour.signed_area()) for contour in contours])
    homogenous_orientations = numpy.alltrue(orientations == -1) or numpy.alltrue(orientations == 1)
    return not homogenous_orientations

def _compatibility_check(contours):
    if not utility_tools.all_same_shape([c.points for c in contours]):
        raise RuntimeError('All contours must have the same number of points in order to align them.')
    if numpy.alltrue([isinstance(c, contour_class.ContourAndLandmarks) for c in contours]):
        # if they're all landmark'd contours
        all_landmarks = [c.landmarks for c in contours]
        if not utility_tools.all_same_shape(all_landmarks):
            raise RuntimeError('If all contours have landmarks, they must all have the same number of landmarks.')


def align_contour_to(contour, reference, global_align = True, align_steps = 8, allow_reflection = False,
    allow_scaling = False, weights = None, quick = False):
    """Optimally align a contour to a reference contour. The input contour will be
    transformed IN PLACE to reflect this alignment.

    Parameters:
    - global_align: if True, the globally optimal point ordering and geometric
        alignment will be found to bring the contour into register with the
        reference. Otherwise only local hill-climbing will be used. Global
        alignment is slower than hill-climibing, however.
    - align_steps: if global_align is True, this is the number of different
        contour orientations to consider. For example, if align_steps = 8,
        then eight different (evenly-spaced) points will be chosen as the
        'first point' of the given contour, and then the fit to the reference
        will be locally optimized from that position. The best local fit is
        then treated as the global alignment.
    - allow_reflection: if True, then reflective transforms will be used if
        they make the alignment between the contour and reference better.
    - allow_scaling: if True, then the contour may be scaled to fit the
        reference better.
    - weights: if provided, this must be a list of weights, one for each
        point, for weighting the fit between the contour and reference.
    - quick: if global_align is True and quick is True, then no local optimization
       will be performed at each of the global search steps. This will provide
       a rough and sub-optimal, but fast, alignment.

    See celltool.contour_class.Contour.global_best_alignment and local_best_alignment,
    which are used internally by this function, for more details.
    """
    _compatibility_check([contour, reference])
    allow_reversed_orientation = _should_allow_reverse([contour], allow_reflection)
    allow_translation = True
    if global_align:
        # axis-align first, so that the align_steps correspond to similar locations for each contour
        contour.axis_align()
        contour.global_best_alignment(reference, align_steps, weights, allow_reflection,
                  allow_scaling, allow_translation, allow_reversed_orientation, quick)
    else:
        distance = contour.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)
        if allow_reversed_orientation:
            rev = self.as_reversed_orientation()
            r_distance = rev.local_best_alignment(reference, weights, allow_reflection, allow_scaling, allow_translation)
            if r_distance < distance:
                contour.__init__(other = rev)

def align_contours(contours, align_steps = 8, allow_reflection = False,
    allow_scaling = False, weights = None, max_iters = 10, min_rms_change = None,
    quick = False, iteration_callback = None):
    """Mutually align a set of contours to their mean in an expectation-maximization
    fashion. The input contous will be transformed IN PLACE to reflect this alignment.

    For each iteration, the mean contour is calculated, and then each contour is
    globally aligned to that mean with the celltool.contour_class.Contour.global_best_alignment
    method. Iteration continues until no contours are changed (beyond a given
    threshold), or the maximum number of iterations elapses.

    Parameters:
    - align_steps: The number of different contour orientations to consider
        when aligning each contour to the mean. For example, if align_steps = 8,
        then eight different (evenly-spaced) points will be chosen as the
        'first point' of the given contour, and then the fit to the mean
        will be locally optimized from that position. The best local fit is
        then treated as the global alignment.
    - allow_reflection: if True, then reflective transforms will be used if
        they make the alignment between the contour and reference better.
    - allow_scaling: if True, then the contour may be scaled to fit the
        reference better.
    - weights: if provided, this must be a list of weights, one for each
        point, for weighting the fit between the contour and reference.
    - max_iters: maximum number of alignment iterations.
    - min_rms_change: minimum RMS change between the contour points before
        and after alignment to the mean for that contour to be considered to
        have "changed". If no contours change, then iteration terminates;
        thus too stringent a criteria can prolong iteration, while too lax
        of one will produce sub-optimal results. If this parameter is None,
        then an appropriate value will be chosen.
    - quick: if True, then no local optimization will be performed at each of
       the global search steps. This will provide a rough and sub-optimal, but
       fast, alignment.
    - iteration_callback: if not None, this function is called after each
       contour is aligned, as follows: iteration_callback(iters, i, changed)
       where iters is the current iteration, i is the number of the contour
       that was just aligned, and changed is the number of contours changed
       so far during that iteration.

    See celltool.contour_class.Contour.global_best_alignment, which is used
    internally by this function, for more details.
    """

    _compatibility_check(contours)
    allow_reversed_orientation = _should_allow_reverse(contours, allow_reflection)
    allow_translation = True
    # roughly align the contours and make the point orderings correspond so that
    # the initial mean will be at all reasonable.
    for c in contours:
        c.axis_align()
        c.global_reorder_points(reference = contours[0])
    mean = contour_class.calculate_mean_contour(contours)
    if min_rms_change is None:
        # set the min RMSD to 0.01 of the largest dimension of the mean shape.
        min_rms_change = 0.01 * mean.size().max()
    min_ms_change = min_rms_change**2
    changed = 1
    iters = 0
    while changed != 0 and iters < max_iters:
        changed = 0
        for i, contour in enumerate(contours):
            original_points = contour.points[:]
            contour.global_best_alignment(mean, align_steps, weights, allow_reflection,
              allow_scaling, allow_translation, allow_reversed_orientation, quick)
            ms_change = ((contour.points - original_points)**2).mean()
            if ms_change > min_ms_change:
                changed += 1
            if iteration_callback is not None:
                iteration_callback(iters, i, changed)
        iters += 1
        mean = contour_class.calculate_mean_contour(contours)
    return mean,iters