### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import jittor as jt
from jittor import nn
import functools
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            jt.init.gauss_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            jt.init.constant_(m.bias, 0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'adain':
        netG = GlobalGenerator_adain(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.numel() != input.numel():
                self.real_label_var = jt.full_like(input, self.real_label).stop_grad()
            return self.real_label_var
        else:
            if self.fake_label_var is None or self.fake_label_var.numel() != input.numel():
                self.fake_label_var = jt.full_like(input, self.fake_label).stop_grad()
            return self.fake_label_var

    def execute(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids=None):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def execute(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].stop_grad())        
        return loss

##############################################################################
# Generator
##############################################################################

class GlobalGenerator_adain(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, style_dim=16, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect',without_act=False,input_nc2=None):
        assert(n_blocks >= 0)
        super(GlobalGenerator_adain, self).__init__()        
        activation = nn.ReLU()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='adain', padding_type=padding_type)]
        
        # print("dim:")
        # print(ngf * mult)
        
        ### upsample  
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [AdaptiveInstanceNorm2d(int(ngf * mult / 2))]
            model += [activation]
        if without_act:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), ]
        else:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        self.to_feature = nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)
        # style encoder
        if input_nc2==None:
            self.enc_style = StyleEncoder(5, input_nc, style_dim, self.get_num_adain_params(self.model), norm='none', activ='relu', pad_type='reflect')
        else:
            self.enc_style = StyleEncoder(5, input_nc2, style_dim, self.get_num_adain_params(self.model), norm='none', activ='relu', pad_type='reflect')
    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.reshape(-1)
                m.weight = std.reshape(-1)
                if adain_params.shape[1] > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def execute(self, image_content, image_style, adain_params = None):
        adain_params = self.enc_style(image_style)
        self.assign_adain_params(adain_params, self.model)
        return self.model(image_content),adain_params

    def feature_forward(self, image_content, image_style,return_adain=False):
        adain_params = self.enc_style(image_style)#1x19328x1x1
        # print(adain_params.shape)
        # import pdb
        # pdb.set_trace()
        self.assign_adain_params(adain_params, self.model)
        feature = None
        #print(len(self.model) - 3)
        for layer_id, layer in enumerate(self.model):
            image_content = layer(image_content)
            if layer_id == (len(self.model) - 2):
                feature = image_content
        if return_adain:
            return feature, image_content, adain_params
        return feature, image_content
        

# Define the Multiscale Discriminator.
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_execute(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def execute(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_execute(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Define the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def execute(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

# Vgg19 not used for inference - skipped (requires torchvision)
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        raise NotImplementedError("Vgg19 requires torchvision, not available in Jittor inference mode")

    def execute(self, X):
        pass

# style encode part
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        
        self.model = nn.Sequential(*self.model)
        
        self.output_dim = dim

    def execute(self, x):
        return self.model(x)

# Define the basic block
class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def execute(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def execute(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_type, padding_type, use_dropout)

    def build_conv_block(self, dim, norm_type, padding_type, use_dropout):
        conv_block = []
        conv_block += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm_type, activation='relu', pad_type=padding_type)]
        conv_block += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm_type, activation='none', pad_type=padding_type)]

        return nn.Sequential(*conv_block)

    def execute(self, x):
        out = x + self.conv_block(x)
        return out

class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv2 = nn.Conv2d(64, 64, 1) 
        self.SFT_shift_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv2 = nn.Conv2d(64, 64, 1)

    def execute(self, x):
        scale = self.SFT_scale_conv2(nn.leaky_relu(self.SFT_scale_conv1(x[1]), 0.1))
        shift = self.SFT_shift_conv2(nn.leaky_relu(self.SFT_shift_conv1(x[1]), 0.1))
        return x[0] * scale + shift

# Definition of normalization layer
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # running stats as plain attributes
        self.running_mean = jt.zeros(num_features)
        self.running_var = jt.ones(num_features)

    def execute(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.shape[0], x.shape[1]
        # Manual instance norm: compute mean/var per instance per channel
        x_flat = x.reshape(b, c, -1)
        mean = x_flat.mean(dim=2, keepdims=True)
        var = x_flat.var(dim=2, keepdims=True)
        x_norm = (x_flat - mean) / jt.sqrt(var + self.eps)
        x_norm = x_norm.reshape(x.shape)
        # Apply dynamic weight and bias
        weight = self.weight.reshape(b, c) if self.weight.ndim > 1 else self.weight.reshape(1, c)
        bias = self.bias.reshape(b, c) if self.bias.ndim > 1 else self.bias.reshape(1, c)
        out = x_norm * weight[:, :, None, None] + bias[:, :, None, None]
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = jt.rand(num_features)
            self.beta = jt.zeros(num_features)

    def execute(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (jt.norm(v) + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.shape[0]
        w_mat = w.reshape(height, -1)
        for _ in range(self.power_iterations):
            v_new = l2normalize(jt.matmul(w_mat.t(), u))
            u_new = l2normalize(jt.matmul(w_mat, v_new))
            v.update(v_new)
            u.update(u_new)

        sigma = jt.sum(u * jt.matmul(w_mat, v))
        setattr(self.module, self.name, w / sigma)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.shape[0]
        width = w.reshape(height, -1).shape[1]

        u = l2normalize(jt.randn(height)).stop_grad()
        v = l2normalize(jt.randn(width)).stop_grad()
        w_bar = w.clone()

        setattr(self.module, self.name + "_u", u)
        setattr(self.module, self.name + "_v", v)
        setattr(self.module, self.name + "_bar", w_bar)

    def execute(self, *args):
        self._update_u_v()
        return self.module(*args)


# Define a resnet block
class ResnetBlock2(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(), use_dropout=False):
        super(ResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def execute(self, x):
        out = x + self.conv_block(x)
        return out

class planes_Generator_single_conv_eg3d(nn.Module):
    def __init__(self, input_nc, output_nc=96, padding_type='reflect'):
        super(planes_Generator_single_conv_eg3d, self).__init__()        
        activation = nn.ReLU()        

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        model = []
        model += [nn.Conv2d(input_nc, 1024, kernel_size=3, stride=1, padding=1),
                    norm_layer(1024), activation]
        model += [ResnetBlock2(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model += [nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                    norm_layer(512), activation]
        #model += [nn.Conv2d(input_nc, 512, kernel_size=3, stride=1, padding=1),
        #            norm_layer(512), activation]
        model += [ResnetBlock2(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model += [nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                    norm_layer(256), activation]
        model += [nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(256), activation]
        model += [ResnetBlock2(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model += [nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                    norm_layer(128), activation]
        model += [ResnetBlock2(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
                    norm_layer(96), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(96, output_nc, kernel_size=7, padding=0)]        
        self.model = nn.Sequential(*model)
        
    def execute(self, input):
        return self.model(input)

