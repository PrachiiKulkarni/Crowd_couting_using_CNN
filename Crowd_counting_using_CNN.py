from flask import Flask, render_template, request,send_file,Response
import os
import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image
from model import UNet, FCRN_A
from scipy.io import loadmat
import io
import base64
from werkzeug.utils import secure_filename
#from resizeimage import resizeimage

app = Flask(__name__,static_folder='E:/PRACHI/Advance Python/FLASK_AP/new_prog/')
app.config['UPLOAD_FOLDER'] = "E:/PRACHI/Advance Python/FLASK_AP/new_prog/"

annots_1 = loadmat(r'mall_gt.mat')
actual_count=[]
for i in range(len(annots_1['count'])):
    actual_count.append(annots_1['count'][i][0])

# img.shape = (4,3,480,640)
# image = np.array(Image.open(img_path), dtype=np.float32) / 255
#image = np.transpose(image, (2, 0, 1))


@app.route('/')
def upload():
    return render_template('upload_image_1.html')


@app.route('/prediction', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        filename = secure_filename(f.filename)

        im = Image.open(f)
        width, height = im.size
        print(width,'+',height)
        
        new_image = im.resize((640, 480))
        width_1, height_1 = new_image.size
        print(width_1,'+',height_1)
        new_image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'resized_image.jpg'))
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], 'resized_image.jpg')


        def infer(image_path,
          network_architecture,
          checkpoint,
          unet_filters,
          convolutions):
            """Run inference for a single image."""
            # use GPU if available
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # only UCSD dataset provides greyscale images instead of RGB
            input_channels = 3

            # initialize a model based on chosen network_architecture
            network = {
                'UNet': UNet
                    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)

            # load provided state dictionary
            # note: by default train.py saves the model in data parallel mode
            network = torch.nn.DataParallel(network)
            network.load_state_dict(torch.load(checkpoint,map_location=device))
            network.eval()

            img = Image.open(image_path)

            # network's output represents a density map
            density_map = network(TF.to_tensor(img).unsqueeze_(0))

            # note: density maps were normalized to 100 * no. of objects
            n_objects = torch.sum(density_map).item() / 100
            
            dmap = density_map.squeeze().cpu().detach().numpy()
    
            # keep the same aspect ratio as an input image
            fig, ax = plt.subplots(figsize=figaspect(1.0 * img.size[1] / img.size[0]))
            fig.subplots_adjust(0, 0, 1, 1)

            # plot a density map without axis
            ax.imshow(dmap, cmap="hot")
            plt.axis('off')
            fig.canvas.draw()

            # create a PIL image from a matplotlib figure
            dmap = Image.frombytes('RGB',
                                       fig.canvas.get_width_height(),
                                       fig.canvas.tostring_rgb())

            # add a alpha channel proportional to a density map value
            dmap.putalpha(dmap.convert('L'))

            # display an image with density map put on top of it
            new_image=Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size))
            
            return new_image,n_objects



        new_image,n_objects=infer(image_path=file_path,
              network_architecture='UNet',
              checkpoint='mall_UNet_1.pth',
              unet_filters=64,         
              convolutions=2)
        n_objects = round(n_objects)

        if 'seq_' in filename:
            x = int(filename[6:10])
            act_n_objects=int(actual_count[x-1])
        else:
            act_n_objects='not defined'
            
            
        output = io.BytesIO()
        rgb_im = new_image.convert('RGB')
        rgb_im.save(output,format='jpeg')
        encoded_img_data = base64.b64encode(output.getvalue())
        
        return render_template('Dmap_display_1.html', user_image = filename, density_map = encoded_img_data.decode('utf-8'), prediction_text='The no. of objects found using Density-Map are {},'.format(n_objects),
                               actual_count='the actual no. of objects in image are {}.'.format(act_n_objects))
    

if __name__ == '__main__':
    port = int(os.getenv('PORT'))
    app.run(debug = True)
