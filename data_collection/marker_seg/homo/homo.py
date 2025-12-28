from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from PIL import Image, ImageEnhance, ImageFilter
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
    parser.add_argument("--model",type=str,default="efficientsam_ti") # efficientsam_ti, efficientsam_s

    parser.add_argument("--ori_root",type=str,default="data/sam_temp/hetero/gelsight") #all images
    parser.add_argument("--marker_root",type=str,default="data/sam_temp/hetero/gelsight/marker") #all images
    parser.add_argument("--process_markers",type=str2bool, default='False', help="if True, extract marker for all imgs") #all images

    parser.add_argument("--see_img",type=str2bool, default='True', help="if True, extract marker for single imgs")  #single image
    parser.add_argument("--img_dir",type=str,default="data/sam_temp/hetero/gelsight") #single image
    parser.add_argument("--save_dir",type=str,default="data/sam_temp/hetero/gelsight/marker") #single image
    
    parser.add_argument("--save_img",type=str2bool, default='False')  # save jpg
    parser.add_argument("--save_npy",type=str2bool, default='True')  # save npy
    parser.add_argument("--pair",type=str2bool, default='True') # save pair path
    parser.add_argument("--mode",type=str, default='box') # point, box
    parser.add_argument("--white_color",type=str2bool, default='False') # for white color gelsight
    return parser

def show_img(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def find_marker(frame,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        mask_range=(150, 255), min_value:int=70,
        morphop_iter=1, morphclose_iter=1, dilate_iter=1,white_color=False):
    """
    #mask range(160,255) for while color, (150,255) for rgb
    find markers in the tactile iamge
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # show_img(gray)
    img_lblur = cv2.GaussianBlur(gray, (15,15),5)  #5
    im_blur_sub = img_lblur - gray + 128
    # show_img(im_blur_sub)
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    # show_img((blur_mask*255).astype(np.uint8))
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    # show_img(value)
    value_mask = value < min_value
    # show_img((value_mask*255).astype(np.uint8))
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    # show_img(mask255)
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    if dilate_iter > 0:
        dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    else:
        dilate_mask = mask255_op
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    # show_img(morph_close)
    return morph_close

def dialated_img(img,
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)),
                 dilation_iter=5, #5
                 erosion_iter=3): #3
    img = cv2.dilate(img,kernel,iterations=dilation_iter)
    img = cv2.erode(img,kernel,iterations=erosion_iter)
    return img

def find_center(img):
    img_dialted = dialated_img(img)
    img_ref_bin = img_dialted>254
    img_ref_labelled = measure.label(img_ref_bin)
    img_ref_props = measure.regionprops(img_ref_labelled)
    p0 = [[pro.centroid[1],pro.centroid[0]] for pro in img_ref_props]
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

def openning_img(img,
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)),
                 dilation_iter=2,
                 erosion_iter=2):
    img = np.array(img,dtype=np.uint8)
    img = cv2.erode(img,kernel,iterations=erosion_iter)
    img = cv2.dilate(img,kernel,iterations=dilation_iter)
    img = img > 254
    return img

def marker_extractor_box(img_dir, model, box_length = 15, save_dir=None,white_color=False):
    """box length = diameter/2, d=30 px default 13
       change box length according to your marker size
    """
    cur = cv2.imread(img_dir)
    # show_img(cur,"ori")
    marker_cv = find_marker(cur,white_color=white_color)
    input_points = find_center(marker_cv)
    points = len(input_points)
    sample_image = Image.open(img_dir)
    sample_image_np = np.array(sample_image)
    sample_image_tensor = transforms.ToTensor()(sample_image_np).cuda() 
    mask = np.zeros((sample_image_np.shape[0],sample_image_np.shape[1]),dtype=np.uint8)
    for i in range(0,len(input_points),points):
        """3 boxes with 6 points"""
        if len(input_points)-i<points: 
            input_points_in = input_points[i:]
        else:
            input_points_in = input_points[i:i+points]
        input_box = [[p[0]-box_length,p[1]-box_length,p[0]+box_length,p[1]+box_length] for p in input_points_in]
        input_box = torch.tensor(input_box,dtype=torch.int32)
        num_queries = len(input_box)
        input_labels_in = torch.tensor([2, 3])  # top-left, bottom-right
        input_labels_in = input_labels_in[None, None].repeat(1, num_queries, 1).cuda() # [bs, num_queries, num_pts]
        input_points_in = torch.as_tensor(input_box).unsqueeze(0)      # [bs, num_queries, 4], bs = 1
        input_points_in = input_points_in.view(-1, num_queries, 2, 2).cuda()   # [bs, num_queries, num_pts, 2]
        predicted_logits, predicted_iou = model(
            sample_image_tensor[None, ...],
            input_points_in,
            input_labels_in,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        mask_segs = torch.ge(predicted_logits,0).cpu().detach().numpy()
        mask_segs = mask_segs[0,:,0,:,:]
        for seg in mask_segs:
             mask = np.logical_or(mask,seg)
    if save_dir is not None:
        img_name = os.path.split(img_dir)[1]
        os.makedirs(save_dir,exist_ok=True)
        Image.fromarray(mask).save(f"{save_dir}/marker_{img_name}")
    return mask

def mask_show(predicted_logits,i):
    mask = torch.ge(predicted_logits[0, 0, i, :, :], 0).cpu().detach().numpy()
    mask = mask[:,:,None]*255
    mask = np.concatenate((mask,mask,mask),axis=-1).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.show()

def to_marker_motions(ori_root,marker_root, model, mode='point', save_img=True,save_npy=False, pair=False, white_color=False):
    assert save_img!=save_npy, "have to choose one save format!"
    motions = ["normal_inc","shear_inc","shear_dec","normal_dec"]
    shape = os.listdir(ori_root)
    for s in shape:
        #output
        shape_name = s
        # shape_name = s.split("-")[0]
        # if "dotin" not in s: continue
        # shape_name = s.split("_")[-1]
        # if len(shape_name) < 3:
        #     shape_name = s.split("_")[-2]+"_"+s.split("_")[-1]
        # shape_name = s #array2
        # if not any(shape_name == x for x in ["cylinder_sh","hexagon"]): continue
        shape_dir_ouput = os.path.join(marker_root,shape_name)
        os.makedirs(shape_dir_ouput,exist_ok=True)
        #input
        shape_dir_input = os.path.join(ori_root,s)
        for i in tqdm(os.listdir(shape_dir_input)):
            if i.endswith(".jpg"): continue
            count = 0
            #input
            index_dir_input = os.path.join(shape_dir_input,i)
            #output
            index_dir_ouput = os.path.join(shape_dir_ouput,i)
            os.makedirs(index_dir_ouput,exist_ok=True)
            last_list = []
            for motion in motions:
                #input
                motion_dir_input = os.path.join(index_dir_input,motion)
                img_list_ordered = sorted(os.listdir(motion_dir_input),key=lambda x:int(x.split(".")[0]))
                for img in img_list_ordered:
                    # only extract path of the last image 
                    # if pair:
                    #     if img != img_list_ordered[-1]: 
                    #         count += 1
                    #         continue
                    #     else:
                    #         last_npy = os.path.join(index_dir_ouput,f"{count:04d}.npy")
                    #         last_list.append(last_npy)
                    #         count+=1
                    #         continue
                    img_dir_input = os.path.join(motion_dir_input,img)
                    print(f"processing - {shape_name} - {i} - {motion} - {img}")
                    marker = extractor(mode, img_dir_input, model, white_color=white_color)
                    marker = (marker*255).astype(np.uint8)
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
                    count+=1
            last_list_csv = os.path.join(index_dir_ouput,f"last.csv")
            np.savetxt(last_list_csv,last_list,fmt="%s",delimiter=",")
        print(f'{shape_name}-Done!')
            
def extractor(mode,img_dir,model,save_dir=None, white_color=False):
    if mode =='box':
        marker = marker_extractor_box(img_dir,model,save_dir=save_dir, white_color=white_color)
    return marker

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
        to_marker_motions(args.ori_root, args.marker_root, model, mode=args.mode, save_img=args.save_img, save_npy=args.save_npy, pair=args.pair, white_color=args.white_color)
    
    if args.see_img:
        for img in os.listdir(args.img_dir):
            if not img.endswith(".jpg"):continue
            img_dir = os.path.join(args.img_dir,img)
            extractor(args.mode, img_dir, model, args.save_dir, white_color=args.white_color)
