# =====================
# COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction
# =====================
# Author: Yijie Lin
# Date: Mar, 2021
# E-mail: linyijie.gm@gmail.com,

# @inproceedings{lin2021completer,
#    title={COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction},
#    author={Lin, Yijie and Gou, Yuanbiao and Liu, Zitao and Li, Boyun and Lv, Jiancheng and Peng, Xi},
#    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month={June},
#    year={2021}
# }
# =====================

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torchviz import make_dot

from loss import crossview_contrastive_Loss
import evaluation
from util import next_batch


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 attation_dim_ht,
                 attation_dim_center,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        """
        添加以其他视图诱导生成的注意力权重
        """
        self.softmax = nn.Softmax(dim=1)
        self.spectral_fe = nn.Sequential(
            nn.Linear(attation_dim_ht[0], attation_dim_center[0]),
            nn.ReLU(),
            nn.Linear(attation_dim_center[0], attation_dim_center[1]),
            nn.ReLU(),
            nn.Linear(attation_dim_center[1], attation_dim_center[2]),
            nn.ReLU(),
            nn.Linear(attation_dim_center[2], attation_dim_center[3]),
            nn.ReLU(),
            nn.Linear(attation_dim_center[3], attation_dim_ht[-1]),
            nn.Sigmoid()
        )

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x, y):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.
            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        y_fe = self.spectral_fe(y)
        # attention = self.spectral_se(y_fe)
        latent = self.softmax(torch.mul(latent, y_fe))

        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class Completer():
    """COMPLETER module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['arch2'],config['Autoencoder']['attation_arch'],
                                        config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['arch1'],config['Autoencoder']['attation_arch'],
                                        config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)

    def train(self, config, logger, x1_train, x2_train, Y_list, mask, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]

        for epoch in range(config['training']['epoch']):
            X1, X2 = shuffle(train_view1, train_view2)
            loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre, loss_lse = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, config['training']['batch_size']):
                z_1 = self.autoencoder1.encoder(batch_x1, batch_x2)
                z_2 = self.autoencoder2.encoder(batch_x2, batch_x1)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2)
                reconstruction_loss = recon1 + recon2

                # Cross-view Contrastive_Loss
                cl_loss = crossview_contrastive_Loss(z_1, z_2, config['training']['alpha'])

                loss = cl_loss + reconstruction_loss * config['training']['lambda2']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_cl += cl_loss.item()

            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                         "===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec1, loss_rec2, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                scores = self.evaluation(config, logger, mask, x1_train, x2_train, Y_list, device)

        return scores['kmeans']['accuracy'], scores['kmeans']['NMI'], scores['kmeans']['ARI']

    def evaluation(self, config, logger, mask, x1_train, x2_train, Y_list, device):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()

            img_idx_eval = mask[:, 0] == 1
            txt_idx_eval = mask[:, 1] == 1

            imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval], x2_train[txt_idx_eval])
            txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval], x1_train[img_idx_eval])

            # representations
            latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                device)
            latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                device)

            latent_code_img_eval[img_idx_eval] = imgs_latent_eval
            latent_code_txt_eval[txt_idx_eval] = txts_latent_eval

            latent_fusion = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

            scores = evaluation.clustering([latent_fusion], Y_list[0])
            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m")

            self.autoencoder1.train(), self.autoencoder2.train()

        return scores
