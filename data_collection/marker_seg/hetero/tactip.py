from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
import cv2
import os
import argparse
from skimage import measure
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="efficientsam_ti")
    parser.add_argument("--img_dir",type=str,default="/path/to/imgdir")
    parser.add_argument("--save_dir",type=str,default="/path/to/savedir")
    parser.add_argument("--ori_root",type=str,default="/path/to/ori_root")
    parser.add_argument("--marker_root",type=str,default="/path/to/marker_root")
    parser.add_argument("--process_markers",type=str2bool, default='False')
    parser.add_argument("--hetero",type=str2bool, default='False')
    parser.add_argument("--see_img",type=str2bool, default='True')
    parser.add_argument("--save_img",type=str2bool, default='True')
    parser.add_argument("--save_npy",type=str2bool, default='False')
    parser.add_argument("--pair",type=str2bool, default='False')
    parser.add_argument("--mode",type=str, default='box')
    parser.add_argument("--white_color",type=str2bool, default='False')
    return parser

def show_img(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def normalize_brightness(image):  
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  
    yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])  
    normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  
    return normalized  

def poly_mask(img):
    mask = np.zeros_like(img)
    points = np.array([[148, 36], [470, 36], [620, 236], [505,430], [165,430], [13, 244]], np.int32)
    points = points.reshape((-1, 1, 2))  
    cv2.fillPoly(mask, [points], color=(255, 255, 255))  
    value = cv2.bitwise_and(mask,img)
    return value

def find_marker(cur,
                morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                mask_range=(100, 255), min_value:int=70,
                morphop_iter=2, morphclose_iter=1, dilate_iter=2):
    value =((cur[..., 2] <100 ) & (cur[..., 1] < 100) & (cur[..., 0] < 30)).astype(np.uint8) * 255  
    value = np.stack((value,value,value),axis=2)
    mask255_op = cv2.morphologyEx(value, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    if dilate_iter > 0:
        dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    else:
        dilate_mask = mask255_op
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    return morph_close

def find_center(img):
    img_ref_bin = img>254
    img_ref_labelled = measure.label(img_ref_bin)
    img_ref_props = measure.regionprops(img_ref_labelled)
    p0 = [[int(pro.centroid[1]),int(pro.centroid[0])] for pro in img_ref_props]
    p0 = np.array(p0,dtype=np.int32)
    return p0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

# --- OPTIMIZED BATCHED MARKER EXTRACTOR ---
def marker_extractor_box(img_dir, model, box_length=12, save_dir=None):
    """
    Batched marker extraction using SAM.
    """
    cur_ori = cv2.imread(img_dir)
    cur = poly_mask(cur_ori)
    cur = normalize_brightness(cur)
    cur = cv2.bitwise_not(cur, cur)
    marker_cv = find_marker(cur)
    input_points = find_center(marker_cv)
    num_points = len(input_points)
    if num_points == 0:
        print(f"No markers found in {img_dir}")
        return np.zeros(cur_ori.shape[:2], dtype=np.uint8)
    
    input_boxes = [[p[0] - box_length, p[1] - box_length, p[0] + box_length, p[1] + box_length] 
                   for p in input_points]
    input_boxes = torch.tensor(input_boxes, dtype=torch.int32).cuda()  # [n, 4]

    sample_image_tensor = transforms.ToTensor()(cur_ori).cuda()
    num_queries = len(input_boxes)
    input_labels = torch.tensor([2, 3])[None,None].repeat(1, num_queries, 1).cuda()
    input_boxes = input_boxes.unsqueeze(0).view(-1, num_queries, 2, 2).cuda() # [1, n, 2, 2]

    with torch.no_grad():
        try:
            predicted_logits, predicted_iou = model(
                sample_image_tensor[None, ...],   # [1, 3, H, W]
                input_boxes,                     # [1, n, 2, 2]
                input_labels,                    # [1, n, 2]
            )
        except torch.cuda.OutOfMemoryError as e:
            print("Cuda Out of Memory", e)
            return np.zeros(cur_ori.shape[:2], dtype=np.uint8)

    mask_segs = torch.ge(predicted_logits, 0).cpu().detach().numpy()[0,:,0,:,:] # [N, H, W]
    combined_mask = np.any(mask_segs, axis=0).astype(np.uint8) * 255

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.split(img_dir)[1]
        Image.fromarray(combined_mask).save(f"{save_dir}/marker_{img_name}")
    return combined_mask

def extractor(mode, img_dir, model, save_dir=None):
    if mode =='box':
        marker = marker_extractor_box(img_dir, model, save_dir=save_dir)
    else:
        raise NotImplementedError("Only 'box' mode is optimized for multi-marker batch.")
    return marker

def to_marker(ori_root, marker_root, model, mode='box', save_img=True, save_npy=False, pair=False):
    assert save_img!=save_npy, "have to choose one save format!"
    motions = ["normal_inc","shear_inc","shear_dec","normal_dec"]
    shape = os.listdir(ori_root)
    for s in shape:
        shape_name = s.split("-")[0]
        shape_dir_ouput = os.path.join(marker_root,shape_name)
        os.makedirs(shape_dir_ouput,exist_ok=True)
        shape_dir_input = os.path.join(ori_root,s)
        
        for i in tqdm(os.listdir(shape_dir_input)):
            if i.endswith(".jpg"): continue
            count = 0
            index_dir_input = os.path.join(shape_dir_input,i)
            index_dir_ouput = os.path.join(shape_dir_ouput,i)
            os.makedirs(index_dir_ouput,exist_ok=True)
            last_list = []

            for motion in motions:
                motion_dir_input = os.path.join(index_dir_input,motion)
                img_list_ordered = sorted(os.listdir(motion_dir_input),key=lambda x:int(x.split(".")[0]))
                for img in img_list_ordered:
                    print(f"processing - {shape_name} - {i} - {motion} - {img}")
                    img_dir_input = os.path.join(motion_dir_input,img)
                    marker = extractor(mode, img_dir_input, model)
                    marker = ((marker>0)*255).astype(np.uint8)
                    if save_img:
                        marker_dir_output = os.path.join(index_dir_ouput,f"{count:04d}.jpg")
                        Image.fromarray(marker).save(marker_dir_output)
                    else:
                        marker = (marker // 255).astype(np.uint8)
                        packed_marker = np.packbits(marker)   
                        marker_dir_output = os.path.join(index_dir_ouput,f"{count:04d}.npy")
                        np.save(marker_dir_output,packed_marker)
                        if pair:
                            if img == img_list_ordered[-1]:
                                last_list.append(marker_dir_output)
                    count += 1
            last_list_csv = os.path.join(index_dir_ouput,f"last.csv")
            np.savetxt(last_list_csv,last_list,fmt="%s",delimiter=",")
        print(f'{shape_name}-Done!')

def to_marker_heterogeneous(ori_root, marker_root, model, mode='box', save_img=False, save_npy=True, pair=True, white_color=False):
    assert save_img!=save_npy, "have to choose one save format!"
    imgs = sorted(os.listdir(ori_root))
    os.makedirs(marker_root,exist_ok=True)
    for img in tqdm(imgs):
        idex = os.path.splitext(img)[0]
        img_dir_input = os.path.join(ori_root,img)
        img_dir_output = os.path.join(marker_root,f"{idex}.npy")
        print(f"processing - {img}")
        marker = extractor(mode, img_dir_input, model)
        marker = ((marker>0)*255).astype(np.uint8)
        crop_x_min, crop_x_max = 150, 523
        crop_y_min, crop_y_max = 70, 350
        marker = marker[crop_y_min:crop_y_max,crop_x_min:crop_x_max]
        marker = cv2.resize(marker, (640,480), interpolation = cv2.INTER_LINEAR)
        if save_img:
            img_dir_output = os.path.join(marker_root,f"{idex}.jpg")
            Image.fromarray(marker).save(img_dir_output)
        else:
            marker = (marker // 255).astype(np.uint8)
            packed_marker = np.packbits(marker)   
            np.save(img_dir_output,packed_marker)
        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.model == "efficientsam_ti":
        model = build_efficient_sam_vitt().cuda()
    elif args.model == "efficientsam_s":
        with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
            zip_ref.extractall("weights")
        model = build_efficient_sam_vits().cuda()

    if args.process_markers:
        to_marker(args.ori_root, args.marker_root, model, mode=args.mode, save_img=args.save_img, save_npy=args.save_npy, pair=args.pair)
    if args.hetero:
        to_marker_heterogeneous(args.ori_root, args.marker_root, model, mode=args.mode, save_img=args.save_img, save_npy=args.save_npy)
    if args.see_img:
        for img in os.listdir(args.img_dir):
            if not img.endswith(".jpg"):continue
            img_dir = os.path.join(args.img_dir,img)
            extractor(args.mode, img_dir, model, args.save_dir)