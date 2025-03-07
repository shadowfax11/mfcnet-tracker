import cv2
import numpy as np

def calc_weighted_centroids(output, mask): 
    r, c = output.shape
    r_ = np.linspace(0,r,r+1)
    c_ = np.linspace(0,c,c+1)
    x_m, y_m = np.meshgrid(c_, r_, sparse=False, indexing='xy')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # select at-most top two
    cX = []
    cY = [] 
    for c in cnts: 
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x,y,w,h = cv2.boundingRect(c)
        wts = output[y:y+h, x:x+w]
        wts[wts < 0.2] = 0
        roi_grid_x = x_m[y:y+h, x:x+w]
        roi_grid_y = y_m[y:y+h, x:x+w]
        wts_x = wts * roi_grid_x
        wts_y = wts * roi_grid_y
        if np.sum(wts) == 0:
            import pdb; pdb.set_trace()
        cX.append(int(np.sum(wts_x)/np.sum(wts)))
        cY.append(int(np.sum(wts_y)/np.sum(wts)))
    return cX, cY

def calc_base_centroid(mask): 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    cX = []
    cY = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX.append(int(M["m10"] / M["m00"]))
        cY.append(int(M["m01"] / M["m00"]))
    return cX, cY

def centroid_error(output, gt, args):
    if args.num_classes != 5:
        raise ValueError('Centroid error can only be computed for 5 classes')
    pred_classes = output.data.cpu().numpy().argmax(axis=1).squeeze()
    
    left_base = 255*(pred_classes==3).astype(np.uint8)
    left_tip = 255*(pred_classes==4).astype(np.uint8)
    left_tip_heatmap = output[0,4,:,:].cpu().numpy()
    left_tip_heatmap[left_tip==0] = 0
    c_lt_x, c_lt_y = calc_weighted_centroids(left_tip_heatmap, left_tip)
    c_lb_x, c_lb_y = calc_base_centroid(left_base)
    
    right_base = 255*(pred_classes==1).astype(np.uint8)
    right_tip = 255*(pred_classes==2).astype(np.uint8)
    right_tip_heatmap = output[0,2,:,:].cpu().numpy()
    right_tip_heatmap[right_tip==0] = 0
    c_rt_x, c_rt_y = calc_weighted_centroids(right_tip_heatmap, right_tip)
    c_rb_x, c_rb_y = calc_base_centroid(right_base)

    gt_classes = gt.cpu().numpy().squeeze()
    gt_left_base = 255*(gt_classes==3).astype(np.uint8)
    gt_left_tip = 255*(gt_classes==4).astype(np.uint8)
    gt_left_tip_heatmap = (gt_classes==4).astype(np.float32)
    c_gt_lt_x, c_gt_lt_y = calc_weighted_centroids(gt_left_tip_heatmap, gt_left_tip)
    c_gt_lb_x, c_gt_lb_y = calc_base_centroid(gt_left_base)

    gt_right_base = 255*(gt_classes==1).astype(np.uint8)
    gt_right_tip = 255*(gt_classes==2).astype(np.uint8)
    gt_right_tip_heatmap = (gt_classes==2).astype(np.float32)
    c_gt_rt_x, c_gt_rt_y = calc_weighted_centroids(gt_right_tip_heatmap, gt_right_tip)
    c_gt_rb_x, c_gt_rb_y = calc_base_centroid(gt_right_base)

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
    
    err_rc = (np.sqrt((c_rt_x[0] - c_gt_rt_x[0])**2 + (c_rt_y[0] - c_gt_rt_y[0])**2) + np.sqrt((c_rt_x[1] - c_gt_rt_x[1])**2 + (c_rt_y[1] - c_gt_rt_y[1])**2))/2
    err_rb = (np.sqrt((c_rb_x[0] - c_gt_rb_x[0])**2 + (c_rb_y[0] - c_gt_rb_y[0])**2)*2)/2
    err_lc = (np.sqrt((c_lt_x[0] - c_gt_lt_x[0])**2 + (c_lt_y[0] - c_gt_lt_y[0])**2) + np.sqrt((c_lt_x[1] - c_gt_lt_x[1])**2 + (c_lt_y[1] - c_gt_lt_y[1])**2))/2
    err_lb = (np.sqrt((c_lb_x[0] - c_gt_lb_x[0])**2 + (c_lb_y[0] - c_gt_lb_y[0])**2)*2)/2
    return err_rc, err_rb, err_lc, err_lb