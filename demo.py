import os
from models.hrn import Reconstructor
import cv2
from tqdm import tqdm
import argparse

#This starts the HRN process, but program executes from main (below)
def run_hrn(args):
    #args includes: --checkpoints_dir, --name, --epoch, --input_type, --input_root, --output
    params = [
        '--checkpoints_dir', args.checkpoints_dir, #Where to store models
        '--name', args.name, #name of expirement
        '--epoch', args.epoch,#which pass of the training dataset through the algorithm (or model), to use
    ]

    #Reconstructor class is in hrn.py
    reconstructor = Reconstructor(params)

    #We sort the input images in the input_root directory
    names = sorted([name for name in os.listdir(args.input_root) if '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])

    #A message to tell the user which directory of input images we are predicting on
    print('predict', args.input_root)

    #tdqm and enumerate just help us display a progress bar
    #We iterate through all the input images
    for ind, name in enumerate(tqdm(names)):
        save_name = os.path.splitext(name)[0] #get the name of the image
        out_dir = os.path.join(args.output_root, save_name) #where to output the results
        os.makedirs(out_dir, exist_ok=True) #make the directory if it doesn't already exist
        img = cv2.imread(os.path.join(args.input_root, name)) #load the image from file 
        output = reconstructor.predict(img, visualize=True, save_name=save_name, out_dir=out_dir)#run predict function 


    #once the process is complete, tell the user where the results are
    print('results are saved to:', args.output_root)


def run_mvhrn(args):
    params = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--epoch', args.epoch,
    ]

    reconstructor = Reconstructor(params)

    names = sorted([name for name in os.listdir(args.input_root) if
                    '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])
    os.makedirs(args.output_root, exist_ok=True)

    print('predict', args.input_root)

    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)
    img_list = []
    for ind, name in enumerate(names):
        img = cv2.imread(os.path.join(args.input_root, name))
        img_list.append(img)
        # output = reconstructor.predict_base(img, save_name=save_name, out_dir=out_dir)
    output = reconstructor.predict_multi_view(img_list, visualize=True, out_dir=out_dir)

    print('results are saved to:', args.output_root)

#Execution Starts Here
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #We add arguments and then have th parser parse the command line arguments to fill out the ones we defined.
    parser.add_argument('--checkpoints_dir', type=str, default='assets/pretrained_models', help='models are saved here')
    parser.add_argument('--name', type=str, default='hrn_v1.1',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--epoch', type=str, default='10', help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--input_type', type=str, default='single_view',  # or 'multi_view'
                        help='reconstruct from single-view or multi-view')
    parser.add_argument('--input_root', type=str, default='./assets/examples/single_view_image',
                        help='directory of input images')
    parser.add_argument('--output_root', type=str, default='./assets/examples/single_view_image_results',
                        help='directory for saving results')

    args = parser.parse_args()

    if args.input_type == 'multi_view':
        run_mvhrn(args)
    else:
        #We use this one for now
        run_hrn(args)