import torch
import networks
from models.model import BaseModel


class Pix2PixModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]

        if self.is_train:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        self.netG = self._define_G(
            config.input_nc,
            config.output_nc,
            config.ngf,
            config.netG,
            config.norm,
            not config.no_dropout,
            config.init_type,
            config.init_gain,
            self.gpu_ids,
        )

        if self.is_train:
            self.netD = self._define_D(
                config.input_nc + config.output_nc,
                config.ndf,
                config.netD,
                config.n_layers_D,
                config.norm,
                config.init_type,
                config.init_gain,
                self.gpu_ids,
            )

        if self.is_train:
            self.criterionGAN = networks.GANLoss(config.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def _define_G(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int,
        netG: str,
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
    ):
        net = None
        norm_layer = networks.get_norm_layer(norm_type=norm)
        if netG == "unet_128":
            net = networks.UnetGenerator(
                input_nc,
                output_nc,
                7,
                ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        elif netG == "unet_256":
            net = networks.UnetGenerator(
                input_nc,
                output_nc,
                8,
                ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        else:
            raise NotImplementedError(
                "Generator model name {} is not recognized".format(netG)
            )
        return networks.init_net(net, init_type, init_gain, gpu_ids)

    def _define_D(
        self,
        input_nc: int,
        ndf: int,
        netD: str,
        n_layers_D: int,
        norm="batch",
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
    ):
        net = None
        norm_layer = networks.get_norm_layer(norm_type=norm)
        if netD == "basic":
            net = networks.NLayerDiscriminator(
                input_nc, ndf, n_layers=3, norm_layer=norm_layer
            )
        elif netD == "n_layers":
            net = networks.NLayerDiscriminator(
                input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer
            )
        elif netD == "pixel":
            net = networks.PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        else:
            raise NotImplementedError(
                "Discriminator model name {} is not recognized".format(netD)
            )
        return networks.init_net(net, init_type, init_gain, gpu_ids)

    def set_input(self, input):
        AtoB = self.config.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self._backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self._backward_G()
        self.optimizer_G.step()

    def _backward_G(self):
        fake_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = (
            self.criterionL1(self.fake_B, self.real_B) * self.config.lambda_L1
        )
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def _backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
