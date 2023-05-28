"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        print('input_shape', input_shape)
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        print("pretrained feature_dimension ",  feature_dimensions )
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for images in data:
                if isinstance(images, dict):  
                    images = images["images"]  ####
                    print(images)
                with torch.no_grad():
                    input_image = images.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        #print('f1', features.shape)
        #print('fl2', features['layer2'].shape)
        #print('fl3', features['layer3'].shape)
        #print(features['2'].shape)
        if 'blocks' in self.layers_to_extract_from[0]  : 
            #features = [features[layer].unsqueeze(3) for layer in self.layers_to_extract_from]
            #features = [features[layer].unsqueeze(3) for layer in ['0','1', '2', '3', '4', '5']]
            features = [features[layer].reshape(features[layer].shape[:-1]+(32,32)) for layer in self.layers_to_extract_from]
        else:
            features = [features[layer] for layer in self.layers_to_extract_from]
        #features = [features[layer].unsqueeze(3) for layer in self.layers_to_extract_from]
        #features = [features[layer].unsqueeze(3) for layer in ['0']]
        #print(self.layers_to_extract_from)
        #print(len(features)) 
        #print(features[0].shape)
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        #print(len(features))
        #print('fff', features[0][0].shape)
        patch_shapes = [x[1] for x in features]
        #print('patch_shapes', patch_shapes)
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        #print('len ',  len(features))
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            # TODO(pgehler): Add comments
            #if i < 2:
            #    print('fff', _features.shape)
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            #print('_f1', _features.shape)
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            #print('_f2', _features.shape)
            #if i < 2:
            #    print('ddd', _features.shape)
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            #print('_f3', _features.shape)
            #if i < 2:
            #    print('eee', _features.shape)
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            #print('_f4', _features.shape)
            features[i] = _features
            #if i < 2:
            #    print('eee', _features.shape)
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        #for f in features:
        #    print('f shape', f.shape)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        #print('preprocessed f', features.shape)
        features = self.forward_modules["preadapt_aggregator"](features)
        #print('agg f', features.shape)
        
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                    #print(len(image), image)
                #for im in images:
                features.append(_image_to_features(image))
        print("features are computed")
        print('feature shape', np.array(features).shape)
        print(len(features))
        print(len(features[0]))
       # print(features[0].shape)
        #####features  = [ f[:1000] for f in features]
        features = np.concatenate(features, axis=0)
        print("concatanated", features.shape)
        features = self.featuresampler.run(features)
        print("fiting anomaly score", features.shape)
        #print(features.shape)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        #scores = {'image1': [], 'image2': [] , 'image3': []}
        #masks = {'mask1': [], 'mask2': [], 'mask3': []}
        #labels_gt = []
        #masks_gt = {'maskgt1': [], 'maskgt2': [], 'maskgt3': []}
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    mask =  image['mask']
                    masks_gt.extend(mask.numpy().tolist())
                    image = image["image"]
                    #print(image)
                #for i in range(len(images)):
                    _scores, _masks = self._predict(image)
                    #print(_masks)
                    for score, mask in zip(_scores, _masks):
                        scores.append(score)
                        masks.append(mask)
                    #print(masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        #print('Infer', images.shape)
        
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores, query_distances, query_nns, softmax_value = self.anomaly_scorer.predict([features])
            ##patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = patch_scores 
            #print("patch_scores", patch_scores, patch_scores.shape)
            #print('q distance', query_distances, query_distances.shape)
            #print('query nns', query_nns)
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            #patch_scores = patch_scores.reshape( 1 , scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            ######image_scores = softmax_value  ########################

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]

        """
        #print("---------------")
        #print('patchify features shape', features.shape)
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        ####print('unfold f.shape', unfolded_features.shape)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        ###print(unfolded_features.shape)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        ###print(unfolded_features.shape)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
