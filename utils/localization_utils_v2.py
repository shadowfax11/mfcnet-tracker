import cv2 
import numpy as np 
from scipy import ndimage

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def calc_centroids(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(cnts) == 0:
        return [], []
    elif len(cnts)>2:
        cnts = cnts[:2]
    cX = []
    cY = []
    for c in cnts:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX.append(c[0][0][0])
            cY.append(c[0][0][1])
        else:
            cX.append(int(M["m10"] / M["m00"]))
            cY.append(int(M["m01"] / M["m00"]))
    return cX, cY

def determine_local_maxima_and_estimate_centroids(heatmap, blob, mask):
    heatmap_smoothed = ndimage.gaussian_filter(heatmap, 4)
    localmax = ndimage.maximum_filter(heatmap_smoothed, footprint=mask) == heatmap_smoothed
    localization = blob & localmax
    c_x, c_y = calc_centroids(255*localization.astype(np.uint8))
    return c_x, c_y

def centroid_error_3_classes(output, gt):
    mask = create_circular_mask(10,10).astype(np.float64)
    pred_classes = output.data.cpu().numpy().argmax(axis=1).squeeze()
    left_tip = pred_classes==2
    left_tip_heatmap = output[0,2,:,:].cpu().numpy()
    c_lt_x, c_lt_y = determine_local_maxima_and_estimate_centroids(left_tip_heatmap, left_tip, mask)
    right_tip = pred_classes==1
    right_tip_heatmap = output[0,1,:,:].cpu().numpy()
    c_rt_x, c_rt_y = determine_local_maxima_and_estimate_centroids(right_tip_heatmap, right_tip, mask)

    gt_classes = gt.cpu().numpy().squeeze()
    gt_left_tip = gt_classes==2
    gt_left_tip_heatmap = (gt_classes==2).astype(np.float32)
    c_gt_lt_x, c_gt_lt_y = determine_local_maxima_and_estimate_centroids(gt_left_tip_heatmap, gt_left_tip, mask)
    gt_right_tip = gt_classes==1
    gt_right_tip_heatmap = (gt_classes==1).astype(np.float32)
    c_gt_rt_x, c_gt_rt_y = determine_local_maxima_and_estimate_centroids(gt_right_tip_heatmap, gt_right_tip, mask)

    if len(c_gt_lt_x) == 0:
        c_gt_lt_x = [np.nan, np.nan]
        c_gt_lt_y = [np.nan, np.nan]
    elif len(c_gt_lt_x) == 1:
        c_gt_lt_x.append(c_gt_lt_x[0])
        c_gt_lt_y.append(c_gt_lt_y[0])
    if len(c_gt_rt_x) == 0:
        c_gt_rt_x = [np.nan, np.nan]
        c_gt_rt_y = [np.nan, np.nan]
    elif len(c_gt_rt_x) == 1:
        c_gt_rt_x.append(c_gt_rt_x[0])
        c_gt_rt_y.append(c_gt_rt_y[0])
    
    if len(c_lt_x) == 0:
        c_lt_x = [np.nan, np.nan]
        c_lt_y = [np.nan, np.nan]
    elif len(c_lt_x) == 1:
        c_lt_x.append(c_lt_x[0])
        c_lt_y.append(c_lt_y[0])
    
    if len(c_rt_x) == 0:
        c_rt_x = [np.nan, np.nan]
        c_rt_y = [np.nan, np.nan]
    elif len(c_rt_x) == 1:
        c_rt_x.append(c_rt_x[0])
        c_rt_y.append(c_rt_y[0])
    
    c_gt = [c_gt_rt_x, c_gt_rt_y, c_gt_lt_x, c_gt_lt_y]
    c_pred = [c_rt_x, c_rt_y, c_lt_x, c_lt_y] 

    p_gt_rc = not np.isnan(c_gt_rt_x[0])
    p_gt_lc = not np.isnan(c_gt_lt_x[0])
    p_gt = [p_gt_rc, p_gt_lc]
    p_rc = not np.isnan(c_rt_x[0])
    p_lc = not np.isnan(c_lt_x[0])
    p = [p_rc, p_lc]
    err_rc = np.minimum((np.sqrt((c_rt_x[0] - c_gt_rt_x[0])**2 + (c_rt_y[0] - c_gt_rt_y[0])**2) + np.sqrt((c_rt_x[1] - c_gt_rt_x[1])**2 + (c_rt_y[1] - c_gt_rt_y[1])**2))/2 , 
                        (np.sqrt((c_rt_x[0] - c_gt_rt_x[1])**2 + (c_rt_y[0] - c_gt_rt_y[1])**2) + np.sqrt((c_rt_x[1] - c_gt_rt_x[0])**2 + (c_rt_y[1] - c_gt_rt_y[0])**2))/2)
    err_lc = np.minimum((np.sqrt((c_lt_x[0] - c_gt_lt_x[0])**2 + (c_lt_y[0] - c_gt_lt_y[0])**2) + np.sqrt((c_lt_x[1] - c_gt_lt_x[1])**2 + (c_lt_y[1] - c_gt_lt_y[1])**2))/2, 
                        (np.sqrt((c_lt_x[0] - c_gt_lt_x[1])**2 + (c_lt_y[0] - c_gt_lt_y[1])**2) + np.sqrt((c_lt_x[1] - c_gt_lt_x[0])**2 + (c_lt_y[1] - c_gt_lt_y[0])**2))/2)
    return err_rc, err_lc, p_gt, p, c_gt, c_pred
    
def centroid_error_10_classes(output, gt): 
    mask = create_circular_mask(10,10).astype(np.float64)
    pred_classes = output.data.cpu().numpy().argmax(axis=1).squeeze()
    c_l1_x, c_l1_y = calc_centroids(255*(pred_classes==6).astype(np.uint8))
    c_l2_x, c_l2_y = calc_centroids(255*(pred_classes==7).astype(np.uint8))
    c_l3_x, c_l3_y = calc_centroids(255*(pred_classes==8).astype(np.uint8))
    c_l4_x, c_l4_y = calc_centroids(255*(pred_classes==9).astype(np.uint8))
    c_l5_x, c_l5_y = calc_centroids(255*(pred_classes==10).astype(np.uint8))
    c_r1_x, c_r1_y = calc_centroids(255*(pred_classes==1).astype(np.uint8))
    c_r2_x, c_r2_y = calc_centroids(255*(pred_classes==2).astype(np.uint8))
    c_r3_x, c_r3_y = calc_centroids(255*(pred_classes==3).astype(np.uint8))
    c_r4_x, c_r4_y = calc_centroids(255*(pred_classes==4).astype(np.uint8))
    c_r5_x, c_r5_y = calc_centroids(255*(pred_classes==5).astype(np.uint8))

    gt_classes = gt.cpu().numpy().squeeze()
    c_gt_l1_x, c_gt_l1_y = calc_centroids(255*(gt_classes==6).astype(np.uint8))
    c_gt_l2_x, c_gt_l2_y = calc_centroids(255*(gt_classes==7).astype(np.uint8))
    c_gt_l3_x, c_gt_l3_y = calc_centroids(255*(gt_classes==8).astype(np.uint8))
    c_gt_l4_x, c_gt_l4_y = calc_centroids(255*(gt_classes==9).astype(np.uint8))
    c_gt_l5_x, c_gt_l5_y = calc_centroids(255*(gt_classes==10).astype(np.uint8))
    c_gt_r1_x, c_gt_r1_y = calc_centroids(255*(gt_classes==1).astype(np.uint8))
    c_gt_r2_x, c_gt_r2_y = calc_centroids(255*(gt_classes==2).astype(np.uint8))
    c_gt_r3_x, c_gt_r3_y = calc_centroids(255*(gt_classes==3).astype(np.uint8))
    c_gt_r4_x, c_gt_r4_y = calc_centroids(255*(gt_classes==4).astype(np.uint8))
    c_gt_r5_x, c_gt_r5_y = calc_centroids(255*(gt_classes==5).astype(np.uint8))

    if len(c_gt_l1_x) == 0:
        c_gt_l1_x = [np.nan]; c_gt_l1_y = [np.nan]
    if len(c_gt_l2_x) == 0:
        c_gt_l2_x = [np.nan]; c_gt_l2_y = [np.nan]
    if len(c_gt_l3_x) == 0:
        c_gt_l3_x = [np.nan]; c_gt_l3_y = [np.nan]
    if len(c_gt_l4_x) == 0:
        c_gt_l4_x = [np.nan]; c_gt_l4_y = [np.nan]
    if len(c_gt_l5_x) == 0:
        c_gt_l5_x = [np.nan]; c_gt_l5_y = [np.nan]
    if len(c_gt_r1_x) == 0:
        c_gt_r1_x = [np.nan]; c_gt_r1_y = [np.nan]
    if len(c_gt_r2_x) == 0:
        c_gt_r2_x = [np.nan]; c_gt_r2_y = [np.nan]
    if len(c_gt_r3_x) == 0:
        c_gt_r3_x = [np.nan]; c_gt_r3_y = [np.nan]
    if len(c_gt_r4_x) == 0:
        c_gt_r4_x = [np.nan]; c_gt_r4_y = [np.nan]
    if len(c_gt_r5_x) == 0:
        c_gt_r5_x = [np.nan]; c_gt_r5_y = [np.nan]
    
    if len(c_l1_x) == 0:
        c_l1_x = [np.nan]; c_l1_y = [np.nan]
    if len(c_l2_x) == 0:
        c_l2_x = [np.nan]; c_l2_y = [np.nan]
    if len(c_l3_x) == 0:
        c_l3_x = [np.nan]; c_l3_y = [np.nan]
    if len(c_l4_x) == 0:
        c_l4_x = [np.nan]; c_l4_y = [np.nan]
    if len(c_l5_x) == 0:
        c_l5_x = [np.nan]; c_l5_y = [np.nan]
    if len(c_r1_x) == 0:
        c_r1_x = [np.nan]; c_r1_y = [np.nan]
    if len(c_r2_x) == 0:
        c_r2_x = [np.nan]; c_r2_y = [np.nan]
    if len(c_r3_x) == 0:
        c_r3_x = [np.nan]; c_r3_y = [np.nan]
    if len(c_r4_x) == 0:
        c_r4_x = [np.nan]; c_r4_y = [np.nan]
    if len(c_r5_x) == 0:
        c_r5_x = [np.nan]; c_r5_y = [np.nan]
    
    c_gt = [c_gt_r1_x, c_gt_r1_y, c_gt_r2_x, c_gt_r2_y, c_gt_r3_x, c_gt_r3_y, c_gt_r4_x, c_gt_r4_y, c_gt_r5_x, c_gt_r5_y, c_gt_l1_x, c_gt_l1_y, c_gt_l2_x, c_gt_l2_y, c_gt_l3_x, c_gt_l3_y, c_gt_l4_x, c_gt_l4_y, c_gt_l5_x, c_gt_l5_y]
    c_pred = [c_r1_x, c_r1_y, c_r2_x, c_r2_y, c_r3_x, c_r3_y, c_r4_x, c_r4_y, c_r5_x, c_r5_y, c_l1_x, c_l1_y, c_l2_x, c_l2_y, c_l3_x, c_l3_y, c_l4_x, c_l4_y, c_l5_x, c_l5_y]

    p_gt_r1 = not np.isnan(c_gt_r1_x[0]); p_gt_r2 = not np.isnan(c_gt_r2_x[0]); p_gt_r3 = not np.isnan(c_gt_r3_x[0]); p_gt_r4 = not np.isnan(c_gt_r4_x[0]); p_gt_r5 = not np.isnan(c_gt_r5_x[0])
    p_gt_l1 = not np.isnan(c_gt_l1_x[0]); p_gt_l2 = not np.isnan(c_gt_l2_x[0]); p_gt_l3 = not np.isnan(c_gt_l3_x[0]); p_gt_l4 = not np.isnan(c_gt_l4_x[0]); p_gt_l5 = not np.isnan(c_gt_l5_x[0])
    p_gt = [p_gt_r1, p_gt_r2, p_gt_r3, p_gt_r4, p_gt_r5, p_gt_l1, p_gt_l2, p_gt_l3, p_gt_l4, p_gt_l5]
    p_r1 = not np.isnan(c_r1_x[0]); p_r2 = not np.isnan(c_r2_x[0]); p_r3 = not np.isnan(c_r3_x[0]); p_r4 = not np.isnan(c_r4_x[0]); p_r5 = not np.isnan(c_r5_x[0])
    p_l1 = not np.isnan(c_l1_x[0]); p_l2 = not np.isnan(c_l2_x[0]); p_l3 = not np.isnan(c_l3_x[0]); p_l4 = not np.isnan(c_l4_x[0]); p_l5 = not np.isnan(c_l5_x[0])
    p = [p_r1, p_r2, p_r3, p_r4, p_r5, p_l1, p_l2, p_l3, p_l4, p_l5]

    err_r1 = np.sqrt((c_r1_x[0] - c_gt_r1_x[0])**2 + (c_r1_y[0] - c_gt_r1_y[0])**2)
    err_r2 = np.sqrt((c_r2_x[0] - c_gt_r2_x[0])**2 + (c_r2_y[0] - c_gt_r2_y[0])**2)
    err_r3 = np.sqrt((c_r3_x[0] - c_gt_r3_x[0])**2 + (c_r3_y[0] - c_gt_r3_y[0])**2)
    err_r4 = np.sqrt((c_r4_x[0] - c_gt_r4_x[0])**2 + (c_r4_y[0] - c_gt_r4_y[0])**2)
    err_r5 = np.sqrt((c_r5_x[0] - c_gt_r5_x[0])**2 + (c_r5_y[0] - c_gt_r5_y[0])**2)
    err_l1 = np.sqrt((c_l1_x[0] - c_gt_l1_x[0])**2 + (c_l1_y[0] - c_gt_l1_y[0])**2)
    err_l2 = np.sqrt((c_l2_x[0] - c_gt_l2_x[0])**2 + (c_l2_y[0] - c_gt_l2_y[0])**2)
    err_l3 = np.sqrt((c_l3_x[0] - c_gt_l3_x[0])**2 + (c_l3_y[0] - c_gt_l3_y[0])**2)
    err_l4 = np.sqrt((c_l4_x[0] - c_gt_l4_x[0])**2 + (c_l4_y[0] - c_gt_l4_y[0])**2)
    err_l5 = np.sqrt((c_l5_x[0] - c_gt_l5_x[0])**2 + (c_l5_y[0] - c_gt_l5_y[0])**2)
    err = [err_r1, err_r2, err_r3, err_r4, err_r5, err_l1, err_l2, err_l3, err_l4, err_l5]
    return err, p_gt, p, c_gt, c_pred

def centroid_error(output, gt, args): 
    if args.num_classes != 5:
        if args.num_classes == 3:
            err_rc, err_lc, p_gt, p, c_gt, c_pred = centroid_error_3_classes(output, gt)
            return err_rc, err_lc, p_gt, p, c_gt, c_pred
        else: 
            raise ValueError('Centroid error can only be computed for 5 classes')
    mask = create_circular_mask(10,10).astype(np.float64)
    pred_classes = output.data.cpu().numpy().argmax(axis=1).squeeze()
    left_base = pred_classes==3
    c_lb_x, c_lb_y = calc_centroids(255*left_base.astype(np.uint8))
    left_tip = pred_classes==4
    left_tip_heatmap = output[0,4,:,:].cpu().numpy()
    c_lt_x, c_lt_y = determine_local_maxima_and_estimate_centroids(left_tip_heatmap, left_tip, mask)
    right_base = pred_classes==1
    c_rb_x, c_rb_y = calc_centroids(255*right_base.astype(np.uint8))
    right_tip = pred_classes==2
    right_tip_heatmap = output[0,2,:,:].cpu().numpy()
    c_rt_x, c_rt_y = determine_local_maxima_and_estimate_centroids(right_tip_heatmap, right_tip, mask)

    # print("Pred: ", c_lb_x, c_lb_y, c_lt_x, c_lt_y, c_rb_x, c_rb_y, c_rt_x, c_rt_y)

    gt_classes = gt.cpu().numpy().squeeze()
    gt_left_base = gt_classes==3
    c_gt_lb_x, c_gt_lb_y = calc_centroids(255*gt_left_base.astype(np.uint8))
    gt_left_tip = gt_classes==4
    gt_left_tip_heatmap = (gt_classes==4).astype(np.float32)
    c_gt_lt_x, c_gt_lt_y = determine_local_maxima_and_estimate_centroids(gt_left_tip_heatmap, gt_left_tip, mask)
    gt_right_base = gt_classes==1
    c_gt_rb_x, c_gt_rb_y = calc_centroids(255*gt_right_base.astype(np.uint8))
    gt_right_tip = gt_classes==2
    gt_right_tip_heatmap = (gt_classes==2).astype(np.float32)
    c_gt_rt_x, c_gt_rt_y = determine_local_maxima_and_estimate_centroids(gt_right_tip_heatmap, gt_right_tip, mask)

    # print("GT: ", c_gt_lb_x, c_gt_lb_y, c_gt_lt_x, c_gt_lt_y, c_gt_rb_x, c_gt_rb_y, c_gt_rt_x, c_gt_rt_y)


    if len(c_gt_lt_x) == 0:
        c_gt_lt_x = [np.nan, np.nan]
        c_gt_lt_y = [np.nan, np.nan]
    elif len(c_gt_lt_x) == 1:
        c_gt_lt_x.append(c_gt_lt_x[0])
        c_gt_lt_y.append(c_gt_lt_y[0])
    if len(c_gt_lb_x) == 0:
        c_gt_lb_x = [np.nan]
        c_gt_lb_y = [np.nan]
    if len(c_gt_rt_x) == 0:
        c_gt_rt_x = [np.nan, np.nan]
        c_gt_rt_y = [np.nan, np.nan]
    elif len(c_gt_rt_x) == 1:
        c_gt_rt_x.append(c_gt_rt_x[0])
        c_gt_rt_y.append(c_gt_rt_y[0])
    if len(c_gt_rb_x) == 0:
        c_gt_rb_x = [np.nan]
        c_gt_rb_y = [np.nan]
    
    if len(c_lt_x) == 0:
        c_lt_x = [np.nan, np.nan]
        c_lt_y = [np.nan, np.nan]
    elif len(c_lt_x) == 1:
        c_lt_x.append(c_lt_x[0])
        c_lt_y.append(c_lt_y[0])
    if len(c_lb_x) == 0:
        c_lb_x = [np.nan]
        c_lb_y = [np.nan]
    
    if len(c_rt_x) == 0:
        c_rt_x = [np.nan, np.nan]
        c_rt_y = [np.nan, np.nan]
    elif len(c_rt_x) == 1:
        c_rt_x.append(c_rt_x[0])
        c_rt_y.append(c_rt_y[0])
    if len(c_rb_x) == 0:
        c_rb_x = [np.nan]
        c_rb_y = [np.nan]
    
    c_gt = [c_gt_rt_x, c_gt_rt_y, c_gt_rb_x, c_gt_rb_y, c_gt_lt_x, c_gt_lt_y, c_gt_lb_x, c_gt_lb_y]
    c_pred = [c_rt_x, c_rt_y, c_rb_x, c_rb_y, c_lt_x, c_lt_y, c_lb_x, c_lb_y]

    p_gt_rc = not np.isnan(c_gt_rt_x[0])
    p_gt_rb = not np.isnan(c_gt_rb_x[0])
    p_gt_lc = not np.isnan(c_gt_lt_x[0])
    p_gt_lb = not np.isnan(c_gt_lb_x[0])
    p_gt = [p_gt_rc, p_gt_rb, p_gt_lc, p_gt_lb]
    p_rc = not np.isnan(c_rt_x[0])
    p_rb = not np.isnan(c_rb_x[0])
    p_lc = not np.isnan(c_lt_x[0])
    p_lb = not np.isnan(c_lb_x[0])
    p = [p_rc, p_rb, p_lc, p_lb]
    err_rc = np.minimum((np.sqrt((c_rt_x[0] - c_gt_rt_x[0])**2 + (c_rt_y[0] - c_gt_rt_y[0])**2) + np.sqrt((c_rt_x[1] - c_gt_rt_x[1])**2 + (c_rt_y[1] - c_gt_rt_y[1])**2))/2 , 
                        (np.sqrt((c_rt_x[0] - c_gt_rt_x[1])**2 + (c_rt_y[0] - c_gt_rt_y[1])**2) + np.sqrt((c_rt_x[1] - c_gt_rt_x[0])**2 + (c_rt_y[1] - c_gt_rt_y[0])**2))/2)
    err_rb = (np.sqrt((c_rb_x[0] - c_gt_rb_x[0])**2 + (c_rb_y[0] - c_gt_rb_y[0])**2)*2)/2
    err_lc = np.minimum((np.sqrt((c_lt_x[0] - c_gt_lt_x[0])**2 + (c_lt_y[0] - c_gt_lt_y[0])**2) + np.sqrt((c_lt_x[1] - c_gt_lt_x[1])**2 + (c_lt_y[1] - c_gt_lt_y[1])**2))/2, 
                        (np.sqrt((c_lt_x[0] - c_gt_lt_x[1])**2 + (c_lt_y[0] - c_gt_lt_y[1])**2) + np.sqrt((c_lt_x[1] - c_gt_lt_x[0])**2 + (c_lt_y[1] - c_gt_lt_y[0])**2))/2)
    err_lb = (np.sqrt((c_lb_x[0] - c_gt_lb_x[0])**2 + (c_lb_y[0] - c_gt_lb_y[0])**2)*2)/2    
    return err_rc, err_rb, err_lc, err_lb, p_gt, p, c_gt, c_pred

