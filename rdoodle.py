#!/usr/bin/env python3
"""                         _       _                 _ _       
 _ __   ___ _   _ _ __ __ _| |   __| | ___   ___   __| | | ___  
| '_ \ / _ \ | | | '__/ _` | |  / _` |/ _ \ / _ \ / _` | |/ _ \ 
| | | |  __/ |_| | | | (_| | | | (_| | (_) | (_) | (_| | |  __/ 
|_| |_|\___|\__,_|_|  \__,_|_|  \__,_|\___/ \___/ \__,_|_|\___| 

"""
#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Doodle is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
# Research and Development sponsored by the nucl.ai Conference!
#   http://events.nucl.ai/
#   July 18-20, 2016 in Vienna/Austria.
#

import os
import sys
import math
import time
import pickle
import argparse
import itertools
import random


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--content',         default=None, type=str,         help='Subject image path to repaint in new style.')
add_arg('--style',           default=None, type=str,         help='Texture image path to extract patches from.')
add_arg('--layers',          default=None, nargs='+', type=str, help='The layers/scales to process.')
add_arg('--variety',         default=[0.2, 0.1, 0.0], nargs='+', type=float,   help='Bias selecting diverse patches')
add_arg('--previous-weight', default=[0.0, 0.2], nargs='+', type=float,        help='Weight of previous layer features.')
add_arg('--content-weight',  default=[0.0], nargs='+', type=float, help='Weight of input content features each layer.')
add_arg('--noise-weight',    default=[0.0], nargs='+', type=float, help='Weight of noise added into features.')
add_arg('--iterations',      default=[6,4,2], nargs='+', type=int, help='Number of iterations to run in each phase.')
add_arg('--shapes',          default=[3], nargs='+', type=int, help='Size of kernels used for patch extraction.')
add_arg('--semantic-ext',    default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=3.0, type=float,        help='Global weight of semantics vs. style features.')
add_arg('--output',          default='output.png', type=str, help='Filename or path to save output once done.')
add_arg('--output-size',     default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--frames',          default=False, action='store_true',   help='Render intermediate frames, takes more time.')
add_arg('--slices',          default=2, type=int,            help='Split patches up into this number of batches.')
add_arg('--device',          default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
add_arg('--randomize',       default=False, type=bool,       help='Generate param values randomly.')

args = parser.parse_args()

if args.randomize is not False:
    print ("randomizing...")

    if args.layers == None:
      layer_confs = [['6_1','5_1','4_1'],['5_1','4_1','3_1'],['5_1','4_1'],['4_1','3_1']]
      layers = random.choice(layer_confs)
      args.layers = layers 
    print("layers",args.layers)
    
    nlayers = len(args.layers)

    r_variety = [0] * nlayers
    r_ = random.randint(0,30) 
    for i in range(0, nlayers):
      r_variety[i] = r_ / 100
      r_ = random.randint(0, int(r_)) 
    print("variety", r_variety)
    args.variety = r_variety
    
    r_cweight = [0] * nlayers
    r_ = random.randint(1,50) 
    for i in range(0, nlayers):
      r_cweight[i] = r_ / 100
      r_ = random.randint(1, int(r_)) 
    print("content_weight", r_cweight)
    args.content_weight = r_cweight

    r_iterations = [0] * nlayers
    r_ = random.randint(1,8) 
    for i in range(0, nlayers):
      r_iterations[i] = r_ 
      r_ = random.randint(1, r_) 
    print("iterations", r_iterations)
    args.iterations = r_iterations

    r_shapes = [0] * nlayers
    r_ = random.randint(1,5) 
    for i in range(0, nlayers):
      r_shapes[i] = 1 + r_
      r_ = random.randint(0, int(r_)) 
    print("shapes", r_shapes)
    args.shapes = r_shapes

    r_pw = [0] * (nlayers - 1)
    r_ = random.randint(1,30) 
    for i in range(0, nlayers-1):
      r_pw[i] = r_ / 100
      r_ = random.randint(1, int(30)) 
    print("previous_weight", r_pw)
    args.previous_weight = r_pw
else:
    args.layers = ['5_1','4_1','3_1']


f = open('/var/www/html/test/params.txt', 'w')
f.write("layers: " + str(args.layers))
f.write(" - variety: " + str(r_variety))
f.write(" - content_weight: " + str(r_cweight))
f.write(" - iterations: " + str(r_iterations))
f.write(" - shapes: " + str(r_shapes))
f.write(" - previous_weight: " + str(r_pw))
f.close()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

def extend(lst): return itertools.chain(lst, itertools.repeat(lst[-1]))
def snap(value, grid=2**(int(args.layers[0][0])-1)): return int(grid * math.floor(value / grid))

print("""{}   {}High-quality image synthesis powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
import PIL.ImageOps
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer

print('{}  - Using the device `{}` for heavy computation.{}'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


#----------------------------------------------------------------------------------------------------------------------
# Convolutional Neural Network
#----------------------------------------------------------------------------------------------------------------------
class Model(object):
    """Store all the data related to the neural network (aka. "model"). This is currently based on VGG19.
    """

    def __init__(self):
        self.setup_model()
        self.load_data()

    def setup_model(self, previous=None):
        """Use lasagne to create a network of convolution layers, first using VGG19 as the framework
        and then adding augmentations for Semantic Style Transfer.
        """
        net, self.channels = {}, {}

        net['map'] = InputLayer((1, 1, None, None))
        for j in range(6):
            net['map%i'%(j+1)] = PoolLayer(net['map'], 2**j, mode='average_exc_pad')

        def DecvLayer(copy, previous, channels, **params):
            # Dynamically injects intermediate "pitstop" output layers in the decoder based on what the user
            # specified as layers. It's rather inelegant... Needs a rework!
            if copy in args.layers:
                if len(self.tensor_latent) > 0:
                    l = self.tensor_latent[-1][0]
                    if args.semantic_weight > 0.0:
                        net['out'+l] = ConcatLayer([previous, net['map%i'%(int(l[0])-1)]])
                    else:
                        net['out'+l] = previous

                self.tensor_latent.append((copy, T.tensor4()))
                net['lat'+copy] = InputLayer((1, previous.num_filters, None, None), var=self.tensor_latent[-1][1])
                previous = net['lat'+copy]

            dup = net['enc'+copy]
            return DeconvLayer(previous, channels, dup.filter_size, stride=dup.stride, crop=dup.pad,
                               nonlinearity=params.get('nonlinearity', lasagne.nonlinearities.elu))

        custom = {'nonlinearity': lasagne.nonlinearities.elu}
        # Encoder part of the neural network, takes an input image and turns it into abstract patterns.
        net['img']    = previous or InputLayer((1, 3, None, None))
        net['enc1_1'] = ConvLayer(net['img'],     32, 3, pad=1, **custom)
        net['enc1_2'] = ConvLayer(net['enc1_1'],  32, 3, pad=1, **custom)
        net['enc2_1'] = ConvLayer(net['enc1_2'],  64, 2, pad=0, stride=(2,2), **custom)
        net['enc2_2'] = ConvLayer(net['enc2_1'],  64, 3, pad=1, **custom)
        net['enc3_1'] = ConvLayer(net['enc2_2'], 128, 2, pad=0, stride=(2,2), **custom)
        net['enc3_2'] = ConvLayer(net['enc3_1'], 128, 3, pad=1, **custom)
        net['enc3_3'] = ConvLayer(net['enc3_2'], 128, 3, pad=1, **custom)
        net['enc3_4'] = ConvLayer(net['enc3_3'], 128, 3, pad=1, **custom)
        net['enc4_1'] = ConvLayer(net['enc3_4'], 256, 2, pad=0, stride=(2,2), **custom)
        net['enc4_2'] = ConvLayer(net['enc4_1'], 256, 3, pad=1, **custom)
        net['enc4_3'] = ConvLayer(net['enc4_2'], 256, 3, pad=1, **custom)
        net['enc4_4'] = ConvLayer(net['enc4_3'], 256, 3, pad=1, **custom)
        net['enc5_1'] = ConvLayer(net['enc4_4'], 512, 2, pad=0, stride=(2,2), **custom)
        net['enc5_2'] = ConvLayer(net['enc5_1'], 512, 3, pad=1, **custom)
        net['enc5_3'] = ConvLayer(net['enc5_2'], 512, 3, pad=1, **custom)
        net['enc5_4'] = ConvLayer(net['enc5_3'], 512, 3, pad=1, **custom)
        net['enc6_1'] = ConvLayer(net['enc5_4'], 768, 2, pad=0, stride=(2,2), **custom)

        # Decoder part of the neural network, takes abstract patterns and converts them into an image!
        self.tensor_latent = []
        net['dec6_1'] = DecvLayer('6_1', net['enc6_1'], 512)
        net['dec5_4'] = DecvLayer('5_4', net['dec6_1'], 512)
        net['dec5_3'] = DecvLayer('5_3', net['dec5_4'], 512)
        net['dec5_2'] = DecvLayer('5_2', net['dec5_3'], 512)
        net['dec5_1'] = DecvLayer('5_1', net['dec5_2'], 256)
        net['dec4_4'] = DecvLayer('4_4', net['dec5_1'], 256)
        net['dec4_3'] = DecvLayer('4_3', net['dec4_4'], 256)
        net['dec4_2'] = DecvLayer('4_2', net['dec4_3'], 256)
        net['dec4_1'] = DecvLayer('4_1', net['dec4_2'], 128)
        net['dec3_4'] = DecvLayer('3_4', net['dec4_1'], 128)
        net['dec3_3'] = DecvLayer('3_3', net['dec3_4'], 128)
        net['dec3_2'] = DecvLayer('3_2', net['dec3_3'], 128)
        net['dec3_1'] = DecvLayer('3_1', net['dec3_2'],  64)
        net['dec2_2'] = DecvLayer('2_2', net['dec3_1'],  64)
        net['dec2_1'] = DecvLayer('2_1', net['dec2_2'],  32)
        net['dec1_2'] = DecvLayer('1_2', net['dec2_1'],  32)
        net['dec1_1'] = DecvLayer('1_1', net['dec1_2'],   3, nonlinearity=lasagne.nonlinearities.tanh)
        net['dec0_0'] = lasagne.layers.ScaleLayer(net['dec1_1'])

        l = self.tensor_latent[-1][0]
        net['out'+l]  = lasagne.layers.NonlinearityLayer(net['dec0_0'], nonlinearity=lambda x: T.clip(127.5*(x+1.0), 0.0, 255.0))

        # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
        for j, i in itertools.product(range(6), range(4)):
            suffix = '%i_%i' % (j+1, i+1)
            if 'enc'+suffix not in net: continue

            self.channels[suffix] = net['enc'+suffix].num_filters
            if args.semantic_weight > 0.0:
                net['sem'+suffix] = ConcatLayer([net['enc'+suffix], net['map%i'%(j+1)]])
            else:
                net['sem'+suffix] = net['enc'+suffix]

            net['dup'+suffix] = InputLayer(net['sem'+suffix].output_shape)
            net['nn'+suffix] = ConvLayer(net['dup'+suffix], 1, 3, b=None, pad=0, flip_filters=False)

        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        data_file = os.path.join(os.path.dirname(__file__), 'gelu2_conv.pkl')
        if not os.path.exists(data_file):
            error("Model file with pre-trained convolution layers not found. Download from here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/gelu2_conv.pkl")

        data = pickle.load(open(data_file, 'rb'))
        for layer, values in data.items():
            #print("*** "+format(layer))
            assert layer in self.network, "Layer `{}` not found as expected.".format(layer)
            for p, v in zip(self.network[layer].get_params(), values):
                assert p.get_value().shape == v.shape, "Layer `{}` in network has size {} but data is {}."\
                                                       .format(layer, v.shape, p.get_value().shape)
                p.set_value(v.astype(np.float32))

    def setup(self, layers):
        """Setup the inputs and outputs, knowing the layers that are required by the optimization algorithm.
        """
        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}
        outputs = lasagne.layers.get_output([self.network[l] for l in layers], tensor_inputs)
        self.tensor_outputs = {k: v for k, v in zip(layers, outputs)}

    def get_outputs(self, type, layers):
        """Fetch the output tensors for the network layers.
        """
        return [self.tensor_outputs[type+l] for l in layers]

    def prepare_image(self, image):
        """Given an image loaded from disk, turn it into a representation compatible with the model. The format is
        (b,c,y,x) with batch=1 for a single image, channels=3 for RGB, and y,x matching the resolution.
        """
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
        image = image.astype(np.float32) / 127.5 - 1.0
        return image[np.newaxis]

    def finalize_image(self, image, resolution):
        """Convert network output into an image format that can be saved to disk, shuffling dimensions as appropriate.
        """
        image = np.swapaxes(np.swapaxes(image[::-1], 0, 1), 1, 2)
        image = np.clip(image, 0, 255).astype('uint8')
        return scipy.misc.imresize(image, resolution, interp='bicubic')


#----------------------------------------------------------------------------------------------------------------------
# Semantic Style Transfer
#----------------------------------------------------------------------------------------------------------------------
class NeuralGenerator(object):
    """This is the main part of the application that generates an image using optimization and LBFGS.
    The images will be processed at increasing resolutions in the run() method.
    """

    def __init__(self):
        """Constructor sets up global variables, loads and validates files, then builds the model.
        """
        self.start_time = time.time()

        # Prepare file output and load files specified as input.
        if args.frames is not False:
            os.makedirs('frames', exist_ok=True)
        if args.output is not None and os.path.isfile(args.output):
            os.remove(args.output)


        # Finalize the parameters based on what we loaded, then create the model.
        args.semantic_weight = math.sqrt(9.0 / args.semantic_weight) if args.semantic_weight else 0.0
        self.model = Model()


    #------------------------------------------------------------------------------------------------------------------
    # Helper Functions
    #------------------------------------------------------------------------------------------------------------------

    def rescale_image(self, img, scale):
        """Re-implementing skimage.transform.scale without the extra dependency. Saves a lot of space and hassle!
        """
        output = scipy.misc.toimage(img, cmin=0.0, cmax=255)
        return np.asarray(PIL.ImageOps.fit(output, [snap(dim*scale) for dim in output.size], PIL.Image.ANTIALIAS))

    def load_images(self, name, filename, scale=1.0):
        """If the image and map files exist, load them. Otherwise they'll be set to default values later.
        """
        basename, _ = os.path.splitext(filename)
        mapname = basename + args.semantic_ext
        img = scipy.ndimage.imread(filename, mode='RGB') if os.path.exists(filename) else None
        map = scipy.ndimage.imread(mapname) if os.path.exists(mapname) and args.semantic_weight > 0.0 else None

        shp = img.shape if img is not None else (map.shape if map is not None else '??')
        print('\n{}{} {}x{}{} at scale {:3.1f}'.format(ansi.BLUE_B, name.capitalize(), shp[1], shp[0], ansi.BLUE, 1.0))
        if img is not None: print('  - Loading `{}` for {} data.'.format(filename, name))
        if map is not None: print('  - Adding `{}` as semantic map.'.format(mapname))

        if img is not None and map is not None and img.shape[:2] != map.shape[:2]:
            error("The {} image and its semantic map have different resolutions. Either:".format(name),
                  "  - Resize {} to {}, or\n  - Resize {} to {}."\
                  .format(filename, map.shape[1::-1], mapname, img.shape[1::-1]))
        return [(self.rescale_image(i, scale) if i is not None else None) for i in [img, map]]

    def compile(self, arguments, function):
        """Build a Theano function that will run the specified expression on the GPU.
        """
        return theano.function(list(arguments), function, on_unused_input='ignore', allow_input_downcast=True)

    def compute_norms(self, backend, layer, array):
        ni = backend.sqrt(backend.sum(array[:,:self.model.channels[layer]] ** 2.0, axis=(1,), keepdims=True))
        ns = backend.sqrt(backend.sum(array[:,self.model.channels[layer]:] ** 2.0, axis=(1,), keepdims=True))
        return [ni, ns]

    def normalize_components(self, layer, array, norms):
        if args.semantic_weight > 0.0:
            array[:,self.model.channels[layer]:] /= (norms[1] * args.semantic_weight)
        array[:,:self.model.channels[layer]] /= (norms[0] * 3.0)


    #------------------------------------------------------------------------------------------------------------------
    # Initialization & Setup
    #------------------------------------------------------------------------------------------------------------------

    def prepare_style(self, scale=1.0):
        """Called each phase of the optimization, process the style image according to the scale, then run it
        through the model to extract intermediate outputs (e.g. sem4_1) and turn them into patches.
        """
        style_img_original, style_map_original = self.load_images('style', args.style, scale)

        if style_map_original is None:
            style_map_original = np.zeros(style_img_original.shape[:2]+(2,)) - 1.0
            args.semantic_weight = 0.0

        if style_img_original is None:
            error("Couldn't find style image as expected.",
                  "  - Try making sure `{}` exists and is a valid image.".format(args.style))

        self.style_img = self.model.prepare_image(style_img_original)
        self.style_map = style_map_original.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

        # Compile a function to run on the GPU to extract patches for all layers at once.
        layer_patches = self.do_extract_patches(args.layers, self.model.get_outputs('sem', args.layers), extend(args.shapes))
        extractor = self.compile([self.model.tensor_img, self.model.tensor_map], layer_patches)
        result = extractor(self.style_img, self.style_map)

        # Store all the style patches layer by layer, resized to match slice size and cast to 16-bit for size. 
        self.style_data = {}
        for layer, *data in zip(args.layers, result[0::3], result[1::3], result[2::3]):
            patches, l = data[0], self.model.network['nn'+layer]
            l.num_filters = patches.shape[0] // args.slices
            self.style_data[layer] = [d[:l.num_filters*args.slices].astype(np.float16) for d in data]\
                                   + [np.zeros((patches.shape[0],), dtype=np.float16)]
            print('  - Layer {} as {} patches {} in {:,}kb.'.format(layer, patches.shape[:2], patches.shape[2:], patches.size//1000))

    def prepare_content(self, scale=1.0):
        """Called each phase of the optimization, rescale the original content image and its map to use as inputs.
        """
        content_img_original, content_map_original = self.load_images('content', args.content or args.output, scale)

        if content_map_original is not None and self.style_map is None:
            basename, _ = os.path.splitext(args.style)
            error("Expecting a semantic map for the input style image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.style_map.max() >= 0.0 and content_map_original is None:
            basename, _ = os.path.splitext(args.content or args.output)
            error("Expecting a semantic map for the input content image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if content_map_original is None:
            if content_img_original is None and args.output_size:
                shape = tuple([int(i) for i in args.output_size.split('x')])
            else:
                if content_img_original is None:
                    shape = self.style_img.shape[2:]
                else:
                    shape = content_img_original.shape[:2]

            content_map_original = np.zeros(shape+(2,))
            args.semantic_weight = 0.0

        if content_img_original is None:
            print("  - No content image found; seed was set to random noise.")
            content_img_original = np.random.uniform(0, 255, content_map_original.shape[:2]+(3,)).astype(np.float32)

        if content_map_original.shape[2] != self.style_map.shape[1]:
            error("Mismatch in number of channels for style and content semantic map.",
                  "  - Make sure both images are RGB, RGBA, or L.")

        self.content_img = self.model.prepare_image(content_img_original)
        self.content_map = content_map_original.transpose((2, 0, 1))[np.newaxis].astype(np.float32)
        self.content_shape = content_img_original.shape

        # Feed-forward calculation only, returns the result of the convolution post-activation 
        self.compute_features = self.compile([self.model.tensor_img, self.model.tensor_map],
                                              self.model.get_outputs('sem', args.layers))

        self.content_features = self.compute_features(self.content_img, self.content_map)
        for layer, current in zip(args.layers, self.content_features):
            print('  - Layer {} as {} array in {:,}kb.'.format(layer, current.shape[1:], current.size//1000))

    def prepare_generation(self):
        """Layerwise synthesis images requires two sets of Theano functions to be compiled.
        """
        # Patch matching calculation that uses only pre-calculated features and a slice of the patches.
        self.matcher_tensors = {l: lasagne.utils.shared_empty(dim=4) for l in args.layers}
        self.matcher_history = {l: T.vector() for l in args.layers}
        self.matcher_inputs = {self.model.network['dup'+l]: self.matcher_tensors[l] for l in args.layers}
        nn_layers = [self.model.network['nn'+l] for l in args.layers]
        self.matcher_outputs = dict(zip(args.layers, lasagne.layers.get_output(nn_layers, self.matcher_inputs)))
        self.compute_matches = {l: self.compile([self.matcher_history[l]], self.do_match_patches(l)) for l in args.layers}

        # Decoding intermediate features into more specialized features and all the way to the output image.
        self.compute_output = []
        for layer, (_, tensor_latent) in zip(args.layers, self.model.tensor_latent):
            output = lasagne.layers.get_output(self.model.network['out'+layer],
                                              {self.model.network['lat'+layer]: tensor_latent,
                                               self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], output)
            self.compute_output.append(fn)


    #------------------------------------------------------------------------------------------------------------------
    # Theano Computation
    #------------------------------------------------------------------------------------------------------------------

    def do_extract_patches(self, layers, outputs, sizes, stride=1):
        """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
        from the intermediate outputs in the model.
        """
        results = []
        for layer, output, size in zip(layers, outputs, sizes):
            # Use a Theano helper function to extract "neighbors" of specific size, seems a bit slower than doing
            # it manually but much simpler!
            patches = theano.tensor.nnet.neighbours.images2neibs(output, (size, size), (stride, stride), mode='valid')
            # Make sure the patches are in the shape required to insert them into the model as another layer.
            patches = patches.reshape((-1, patches.shape[0] // output.shape[1], size, size)).dimshuffle((1, 0, 2, 3))
            # Calculate the magnitude that we'll use for normalization at runtime, then store...
            results.extend([patches] + self.compute_norms(T, layer, patches))
        return results

    def do_match_patches(self, layer):
        # Use node in the model to compute the result of the normalized cross-correlation, using results from the
        # nearest-neighbor layers called 'nn3_1' and 'nn4_1'.
        dist = self.matcher_outputs[layer]
        dist = dist.reshape((dist.shape[1], -1))
        # Compute the score of each patch, taking into account statistics from previous iteration. This equalizes
        # the chances of the patches being selected when the user requests more variety.
        offset = self.matcher_history[layer].reshape((-1, 1))
        scores = dist - offset
        # Pick the best style patches for each patch in the current image, the result is an array of indices.
        # Also return the maximum value along both axis, used to compare slices and add patch variety.
        return [scores.argmax(axis=0), scores.max(axis=0), dist.max(axis=1)]


    #------------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    #------------------------------------------------------------------------------------------------------------------

    def iterate_batches(self, *arrays, batch_size):
        """Break down the data in arrays batch by batch and return them as a generator.
        """ 
        total_size = arrays[0].shape[0]
        indices = np.arange(total_size)
        for index in range(0, total_size, batch_size):
            excerpt = indices[index:index + batch_size]
            yield excerpt, [a[excerpt] for a in arrays]

    def evaluate_slices(self, f, l, v):
        self.normalize_components(l, f, self.compute_norms(np, l, f))
        self.matcher_tensors[l].set_value(f)

        layer, data = self.model.network['nn'+l], self.style_data[l]
        history = data[-1]

        best_idx, best_val = None, 0.0
        for idx, (bp, bi, bs, bh) in self.iterate_batches(*data, batch_size=layer.num_filters):
            weights = bp.astype(np.float32)
            self.normalize_components(l, weights, (bi, bs))
            layer.W.set_value(weights)

            cur_idx, cur_val, cur_match = self.compute_matches[l](history[idx])
            if best_idx is None:
                best_idx, best_val = cur_idx, cur_val
            else:
                i = np.where(cur_val > best_val)
                best_idx[i] = idx[cur_idx[i]]
                best_val[i] = cur_val[i]
            history[idx] = cur_match * v

        return best_idx, best_val

    def evaluate(self, Xn):
        """Feed-forward evaluation of the output based on current image. Can be called multiple times.
        """
        frame = 0
        layer_params = [args.iterations, args.content_weight, args.previous_weight, args.noise_weight, args.variety]
        parameters = zip(args.layers, *[extend(a) for a in layer_params])

        # Iterate through each of the style layers one by one, computing best matches.
        previous_feature = self.content_features[0]
        self.render(frame, args.layers[0], self.content_features[0])

        for parameter, content_feature, compute in zip(parameters, self.content_features, self.compute_output):
            l, iterations, content_weight, previous_weight, noise_weight, variety = parameter
            desired_feature, patches = np.copy(previous_feature), self.style_data[l][0]
            assert previous_weight + content_weight < 1.0, "Previous and content weight should total below 1.0!"

            weights = 'c={:0.1f} p={:0.1f} n={:0.1f}'.format(content_weight, previous_weight, noise_weight)
            print('\n{}Phase {}{}: variety {}, weights {}, iterations {}.{}'\
                 .format(ansi.CYAN_B, l, ansi.CYAN, variety, weights, iterations, ansi.ENDC))
            channels, iter_time = self.model.channels[l], time.time()

            """"
            # Remap the content features onto the style range, helps with blending primarily as patch-matching is normalized.
            smin, smax = patches.min(axis=(0,2,3), keepdims=True), patches.max(axis=(0,2,3), keepdims=True)
            cmin, cmax = content_feature.min(axis=(0,2,3), keepdims=True), content_feature.max(axis=(0,2,3), keepdims=True)
            content_feature = (content_feature - cmin) / (cmax - cmin + 1E-9) * (smax - smin + 1E-9) + smin
            """

            for j in range(iterations):
                # Compute best matching patches this style layer, going through all slices.
                best_idx, best_val = self.evaluate_slices(desired_feature, l, variety)
                current_best = patches[best_idx].astype(np.float32)

                better_patches = current_best.transpose((0, 2, 3, 1))
                better_shape = desired_feature.shape[2:] + (desired_feature.shape[1],)
                better_features = reconstruct_from_patches_2d(better_patches, better_shape)
                better_features = better_features.astype(np.float32).transpose((2, 0, 1))[np.newaxis]

                # The new set of features is a blend of matched patches, input image, and previous layer.
                desired_feature = (1.0 - previous_weight - content_weight) * better_features \
                                + content_weight * content_feature + previous_weight * previous_feature \
                                + noise_weight * np.random.normal(0.0, 1.0, size=previous_feature.shape).astype(np.float32)

                used = 99.9 * len(set(best_idx)) / best_idx.shape[0]
                dups = 99.9 * len([v for v in np.bincount(best_idx) if v>1]) / best_idx.shape[0]
                err, frame = best_val.mean(), frame + 1
                print('{:>3}   {}patches{}  used {:2.0f}%  dups {:2.0f}%   {}error{} {:3.2e}   {}time{} {:3.1f}s'.format(frame, ansi.BOLD, ansi.ENDC, used, dups, ansi.BOLD, ansi.ENDC, err, ansi.BOLD, ansi.ENDC, time.time() - iter_time))

                self.render(frame, l, desired_feature)
                iter_time = time.time()

            previous_feature = compute(desired_feature[:,:channels], self.content_map)

        return previous_feature

    def render(self, frame, layer, features):
        """Decode features at a specific layer and save the result to disk for visualization. (Takes 50% more time.) 
        """
        if not args.frames: return
        for l, compute in list(zip(args.layers, self.compute_output))[args.layers.index(layer):]:
            features = compute(features[:,:self.model.channels[l]], self.content_map)

        output = self.model.finalize_image(features.reshape(self.content_img.shape[1:]), self.content_shape)
        filename = os.path.splitext(os.path.basename(args.output))[0]
        scipy.misc.toimage(output, cmin=0, cmax=255).save('frames/{}-{:03d}.png'.format(filename, frame))

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """

        self.model.setup(layers=['sem'+l for l in args.layers] + ['out'+l for l in args.layers])
        self.prepare_style()
        self.prepare_content()
        self.prepare_generation()

        Xn = self.evaluate((self.content_img[0] + 1.0) * 127.5)
        output = self.model.finalize_image(Xn.reshape(self.content_img.shape[1:]), self.content_shape)
        scipy.misc.toimage(output, cmin=0, cmax=255).save(args.output)

        print('\n{}Optimization finished in {:3.1f}s!{}\n'.format(ansi.CYAN, time.time()-self.start_time,  ansi.ENDC))


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
