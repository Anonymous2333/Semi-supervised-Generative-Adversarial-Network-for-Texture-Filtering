import torch
from .base_model import BaseModel
from . import networks
import time
class DemoModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_label', 'D_label','G_no_label', 'D_no_label']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['label_A', 'label_B', 'no_label_A', 'no_label_B', 'label_fake_B', 'no_label_fake_B']
            #self.visual_names = ['label_A', 'label_B', 'label_fake_B', 'no_label_A', 'no_label_B', 'no_label_fake_B']
            #self.visual_names = ['label_A', 'label_B', 'label_fake_B']
        else:
            self.visual_names = ['real_A', 'fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            self.netVGGF = networks.define_VGGF(gpu_ids=self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionStyle = networks.StyleLoss(vgg_features=self.netVGGF, select_layers=opt.style_feat_layers)
            self.criterionContent = networks.ContentLoss(vgg_features=self.netVGGF, select_layers=opt.content_feat_layers)
            networks.print_network(self.netVGGF)

    def set_input(self, input):
        if self.isTrain:
            # get direciton
            AtoB = self.opt.direction == 'AtoB'
            # set input images
            self.label_A = input['label_A' if AtoB else 'label_B'].to(self.device)
            self.label_B = input['label_B' if AtoB else 'label_A'].to(self.device)
            self.no_label_A = input['no_label_A' if AtoB else 'no_label_B'].to(self.device)
            self.no_label_B = input['no_label_B' if AtoB else 'no_label_A'].to(self.device)
            self.label_image_paths = input['label_A_paths' if AtoB else 'label_B_paths']
            self.unlabel_image_paths = input['no_label_A_paths' if AtoB else 'no_label_B_paths']
        else:
            for path in input['A']:
                self.real_A = path.unsqueeze(0).to(self.device)
            self.image_paths = input['A_paths']

    def get_image_paths(self):
        if self.isTrain:
            return self.label_image_paths, self.no_label_image_paths
        else:
            return self.image_paths

    # forward pre-train procedures
    def forward_pretrain(self):
        #pretrain label part
        self.label_fake_B = self.netG(self.label_A)
        self.label_fake_data= self.label_A
        self.label_real_data = self.label_B
        #pretrain no label part
        self.no_label_fake_B = self.netG(self.no_label_A)
        self.no_label_fake_data = self.no_label_A
        self.no_label_real_data = self.no_label_B

    def forward(self):

        if self.isTrain:
            # train label part
            self.label_fake_B = self.netG(self.label_A)
            self.label_fake_data = self.label_A
            self.label_real_data = self.label_B
            # train no label part
            self.no_label_fake_B = self.netG(self.no_label_A)
            self.no_label_fake_data = self.no_label_A
            self.no_label_real_data = self.no_label_B
            self.forward_pretrain()
        else:
            # test part
            self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.label_A, self.label_fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # real
        real_AB = torch.cat((self.label_A, self.label_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_label = 0.5 * (self.loss_D_real + self.loss_D_fake)

        fake_AB = torch.cat((self.no_label_A, self.no_label_fake_B),1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.no_label_A, self.no_label_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_no_label = 0.5 * (self.loss_D_real + self.loss_D_fake)

        self.loss_D = 0.5 * (self.loss_D_no_label + self.loss_D_label)
        #self.loss_D = self.loss_D_no_label
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator 100 1 10 20
        label_fake_AB = torch.cat((self.label_A, self.label_fake_B), 1)
        label_pred_fake = self.netD(label_fake_AB)
        self.loss_G_GAN1 = 100 * self.criterionGAN(label_pred_fake, True)
        self.loss_G_content1 = 50 * self.criterionContent(self.label_fake_B, self.label_A)
        #self.loss_G_style1 = 0 * self.criterionStyle(self.label_fake_B, self.label_A)
        self.loss_G_style1 = 90 * self.criterionStyle(self.label_fake_B, self.label_A)
        self.loss_G_L1 = 1 * self.criterionL1(self.label_fake_B, self.label_B)
        self.loss_G_label = self.loss_G_GAN1 + self.loss_G_content1 + self.loss_G_style1 + self.loss_G_L1


        no_label_fake_AB = torch.cat((self.no_label_A, self.no_label_fake_B), 1)
        no_label_pred_fake = self.netD(no_label_fake_AB)
        self.loss_G_GAN2 = 100 * self.criterionGAN(no_label_pred_fake, True)
        self.loss_G_content2 = 50 * self.criterionContent(self.no_label_fake_B, self.no_label_A)
        #self.loss_G_style2 = 0 * self.criterionStyle(self.no_label_fake_B, self.no_label_A)
        self.loss_G_style2 = 90 * self.criterionStyle(self.no_label_fake_B, self.no_label_A)
        self.loss_G_no_label = self.loss_G_GAN2 + self.loss_G_content2 + self.loss_G_style2


        self.loss_G = 0.15 * self.loss_G_no_label + 0.85 * self.loss_G_label

        self.loss_G.backward()

    def optimize_parameters(self):
        #start = time.clock()
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step() # udpate G's weights
        #end = time.clock()
        #print('Running time: %s Seconds'%(end-start))
# 100 100
# 100 200 40*200 8000s
# 100 500 100*200
# 100 1000 200*200
# 100 1500 300*200
# 100 2000 400*200
# 100 3000 600*200