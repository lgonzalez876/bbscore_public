import numpy as np
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Sequence, Union

from data.utils import custom_collate


class FeatureExtractor:
    """Main wrapper for feature extraction from models with support for sequential models."""

    def __init__(
        self,
        model=None,
        layer_name: Union[str, List[str]] = None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        postprocess_fn=None,
        batch_size=32,
        num_workers=0,
        static=True,
        sequence_mode="all",  # Options: "last", "all", "concatenate"
        random_projection=None,  # Options: None, "sparse", "dense"
        target_dim=1024,  # Target dimensionality for random projection
        aggregation_mode="none",  # Options: "none", "concatenate", "stack"
    ):
        """
        Args:
            model: Pre-trained model.
            layer_name: Name of the layer(s) to extract features from. Can be str or List[str].
            device: Device to run the model on.
            postprocess_fn: A callable that postprocesses the captured features.
            sequence_mode: How to handle sequential outputs.
            random_projection: Type of random projection to apply.
            target_dim: Target dimensionality after projection.
            aggregation_mode: Determines how multiple layers are combined:
                - "none": Returns a dictionary {layer_name: features}.
                - "concatenate": Concatenates features along the feature dimension (axis -1) 
                  *before* random projection. Result: (B, [T], Sum(D)).
                - "stack": Stacks features along a new dimension (axis -2). 
                  Applies random projection *per layer* first to ensure uniform dimensions.
                  Result: (B, [T], L, Target_Dim).
        """
        self.device = device
        self.model = model
        self.static = static
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_mode = sequence_mode

        # Handle single vs multiple layers
        if isinstance(layer_name, str):
            self.layer_names = [layer_name]
        elif isinstance(layer_name, list):
            self.layer_names = layer_name
        else:
            raise ValueError(
                "layer_name must be a string or a list of strings")

        self.aggregation_mode = aggregation_mode
        if self.aggregation_mode not in ["none", "concatenate", "stack"]:
            raise ValueError(
                f"Unknown aggregation_mode: {aggregation_mode}. Must be 'none', 'concatenate', or 'stack'.")

        # Random projection settings
        self.random_projection = random_projection
        self.target_dim = target_dim
        self.projection_matrices = {}  # Storage for RP matrices

        if random_projection is not None and target_dim is None:
            raise ValueError(
                "target_dim must be specified when using random projection")

        if random_projection not in [None, "sparse", "dense"]:
            raise ValueError(
                f"random_projection must be None, 'sparse', or 'dense', got {random_projection}")

        # Validation for Stack mode (Warning only, as attributes might change post-init)
        if self.aggregation_mode == "stack" and self.random_projection is None:
            print("WARNING: aggregation_mode='stack' usually requires random_projection to ensure all layers have the same feature dimension. If layers differ in size, this will fail at runtime.")

        # Check if layers exist
        available_layers = dict(model.named_modules())
        for l_name in self.layer_names:
            if not any(name == l_name for name in available_layers):
                raise ValueError(
                    f"Layer {l_name} not found in model {model.__class__.__name__}. Here is \n"
                    f"Here is a list of available layers: {model}"
                )

        self.model.eval()
        self.model.to(device)
        self.postprocess_fn = postprocess_fn

        # Storage: Dictionary mapping layer_name -> list of outputs
        self.features = {l: [] for l in self.layer_names}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture features."""
        if self.model is None or not self.layer_names:
            return

        def get_module_by_name(model, layer_name):
            for name, module in model.named_modules():
                if name == layer_name:
                    return module
            raise ValueError(f"Layer {layer_name} not found in model")

        def hook_fn_factory(layer_id):
            """Creates a hook specific to a layer name."""
            def hook_fn(module, input, output):
                # Handle HuggingFace/diffusers model outputs (dataclasses)
                if hasattr(output, 'sample'):
                    output = output.sample
                elif hasattr(output, 'last_hidden_state'):
                    output = output.last_hidden_state
                elif isinstance(output, tuple):
                    output = output[0]
                self.features[layer_id].append(output.detach())
            return hook_fn

        for l_name in self.layer_names:
            target_layer = get_module_by_name(self.model, l_name)

            # Check if the target layer is a ModuleList
            if isinstance(target_layer, torch.nn.ModuleList):
                last_module = target_layer[-1]
                print(
                    f"WARNING: Target layer '{l_name}' is a ModuleList. Using the last module: {type(last_module).__name__}")
                last_module.register_forward_hook(hook_fn_factory(l_name))
            else:
                target_layer.register_forward_hook(hook_fn_factory(l_name))

    def _initialize_projection_matrix(self, original_dim, key):
        """Initialize the random projection matrix for a specific key."""
        if key in self.projection_matrices:
            return

        torch.manual_seed(42)
        np.random.seed(42)

        if self.random_projection == "sparse":
            # Sparse random projection
            density = 1.0 / np.sqrt(original_dim)
            density = min(1.0, max(0.001, density))
            s = 1.0 / density
            mask = torch.rand(original_dim, self.target_dim) < density
            values = torch.randint(
                0, 2, (original_dim, self.target_dim), dtype=torch.float32) * 2 - 1
            projection = torch.where(
                mask, values * np.sqrt(s), torch.zeros_like(values))

        elif self.random_projection == "dense":
            # Dense Gaussian random projection
            projection = torch.randn(
                original_dim, self.target_dim, dtype=torch.float32)
            projection /= np.sqrt(self.target_dim)

        self.projection_matrices[key] = projection.to(self.device)
        print(
            f"Initialized {self.random_projection} random projection for '{key}': {original_dim} -> {self.target_dim}")

    def _apply_random_projection(self, features, key):
        """Apply random projection to features."""
        if self.random_projection is None:
            return features

        original_shape = features.shape
        is_temporal = len(original_shape) == 3
        original_dim = original_shape[-1]

        # Check/Initialize matrix
        if key not in self.projection_matrices:
            # Skip if dim is too small, UNLESS we are stacking (need uniform dim)
            if self.target_dim >= original_dim and self.aggregation_mode != "stack":
                self.projection_matrices[key] = None  # Mark as skipped
                print(
                    f"WARNING: target_dim ({self.target_dim}) >= original_dim ({original_dim}) for '{key}'. Skipping RP.")
                return features
            self._initialize_projection_matrix(original_dim, key)

        proj_mat = self.projection_matrices[key]
        if proj_mat is None:  # Previously skipped
            return features

        if is_temporal:
            N, T, D = original_shape
            features = features.reshape(-1, D)

        projection_matrix = proj_mat.to(dtype=features.dtype)
        projected = torch.matmul(features, projection_matrix)

        if is_temporal:
            projected = projected.reshape(N, T, self.target_dim)

        return projected

    def get_activations(self, inputs):
        """Apply optional preprocessing, move inputs to device, and pass them through the model."""
        # Clear previous features
        for l in self.layer_names:
            self.features[l] = []

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            return self.model(inputs)

    @classmethod
    def _set_weights(self, outputs: List[torch.Tensor], shape: Sequence[int]) -> torch.Tensor:
        """Helper to align shapes."""
        rank = len(shape)
        if rank == 5:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2], : shape[3], : shape[4]] for out in outputs], dim=1)
        elif rank == 4:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2], : shape[3]] for out in outputs], dim=1)
        elif rank == 3:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2]] for out in outputs], dim=1)
        elif rank == 2:
            return torch.cat([out[: shape[0], None, : shape[1]] for out in outputs], dim=1)
        elif rank == 1:
            return torch.cat([out[: shape[0], None] for out in outputs], dim=1)
        else:
            raise ValueError(f"Unexpected target shape rank: {shape}")

    def _process_sequence_features(self, features_list):
        if not features_list:
            return None
        processed_features = []
        for feat in features_list:
            if isinstance(feat, tuple):
                feat = feat[0]
            processed_features.append(feat)

        if self.sequence_mode == "last":
            return processed_features[-1]
        elif self.sequence_mode == "all":
            if (len(processed_features) == 1 and isinstance(processed_features[0], (list, tuple)) and all(isinstance(t, torch.Tensor) for t in processed_features[0])):
                nested = processed_features[0]
                min_shape = [min(t.shape[d] for t in nested)
                             for d in range(nested[0].ndim)]
                return self._set_weights(nested, min_shape)
            elif all(isinstance(f, torch.Tensor) for f in processed_features):
                return torch.stack(processed_features, dim=1)
            else:
                return np.array(processed_features)
        elif self.sequence_mode == "concatenate":
            if all(isinstance(f, torch.Tensor) for f in processed_features):
                flat_features = [f.flatten(1) for f in processed_features]
                return torch.cat(flat_features, dim=1)
            else:
                flat_features = [np.array(f).reshape(
                    f.shape[0], -1) for f in processed_features]
                return np.concatenate(flat_features, axis=1)
        else:
            raise ValueError(f"Unknown sequence_mode: {self.sequence_mode}")

    def extract_features(self, dataset, downsample_factor=1.0, new_batch_size=None):
        """
        Returns:
            If aggregation_mode is "concatenate" or "stack":
                (all_features_array, all_labels)
            If aggregation_mode is "none":
                (dict_of_arrays, all_labels) where dict is {layer_name: features}
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        self.model.eval()

        # Storage structure
        if self.aggregation_mode in ["concatenate", "stack"]:
            all_features = []
        else:
            all_features = {l: [] for l in self.layer_names}

        all_labels = []

        def process_batch(inputs, labels=None):
            B, T = None, None

            # 1) Handle static, temporal input reshaping
            if self.static and hasattr(inputs, 'shape') and len(inputs.shape) >= 5:
                B, T = inputs.shape[0], inputs.shape[1]
                inputs = inputs.reshape(-1, *inputs.shape[2:])

            # 2) Forward pass
            _ = self.get_activations(inputs)

            # Dictionary to hold processed tensors for this batch
            batch_layer_features = {}

            # 3) Process individual layers
            for l_name in self.layer_names:
                raw_list = self.features[l_name]
                if not raw_list:
                    raise ValueError(
                        f"No features captured for layer {l_name}")

                features_val = self._process_sequence_features(raw_list)

                if not isinstance(features_val, torch.Tensor):
                    features_val = torch.from_numpy(features_val)

                # Downsample (Spatial)
                if downsample_factor < 1.0:
                    axes = list(range(1, features_val.ndim))
                    max_axis = max(axes, key=lambda a: features_val.shape[a])

                    key = (downsample_factor, max_axis, l_name)
                    if not hasattr(self, '_pp_indices'):
                        self._pp_indices = {}

                    if key not in self._pp_indices:
                        D_full = features_val.shape[max_axis]
                        D_keep = max(
                            1, int(np.floor(D_full * downsample_factor)))
                        torch.manual_seed(42)
                        perm = torch.randperm(
                            D_full, device=features_val.device)
                        idx = perm[:D_keep]
                        idx, _ = torch.sort(idx)
                        self._pp_indices[key] = idx

                    features_val = torch.index_select(
                        features_val, dim=max_axis, index=self._pp_indices[key])

                # Restore static/time shape if needed
                if self.static and B is not None and features_val.ndim > 2:
                    new_shape = (B, T) + features_val.shape[1:]
                    features_val = features_val.reshape(new_shape)

                # Postprocess (Flatten spatial dims usually)
                if self.postprocess_fn is not None:
                    features_val = self.postprocess_fn(features_val)

                # Safety net: if features are still high-dimensional (>2D), force global average pool
                if isinstance(features_val, torch.Tensor) and features_val.ndim > 2:
                    print(f"  [WARNING] Features still {features_val.ndim}D after postprocess: "
                          f"shape={features_val.shape}. Applying global average pooling.")
                    # Squeeze any singleton dims first
                    while features_val.ndim > 4:
                        features_val = features_val.squeeze(1)
                    if features_val.ndim == 4:
                        features_val = torch.nn.functional.adaptive_avg_pool2d(
                            features_val, (1, 1)
                        ).flatten(1)
                    elif features_val.ndim == 3:
                        features_val = features_val.mean(dim=1)

                if not isinstance(features_val, torch.Tensor):
                    features_val = torch.from_numpy(
                        features_val).to(self.device)

                batch_layer_features[l_name] = features_val

            # 4) Aggregation & Random Projection Logic
            if self.aggregation_mode == "concatenate":
                # CONCAT: (B, T, D_total)
                # Concatenate along feature dimension (last axis)
                to_concat = [batch_layer_features[l]
                             for l in self.layer_names]
                combined = torch.cat(to_concat, dim=-1)

                # Apply Random Projection to the combined result
                combined = self._apply_random_projection(
                    combined, key="combined")

                if isinstance(combined, torch.Tensor):
                    combined = combined.cpu().numpy()
                all_features.append(combined)

            elif self.aggregation_mode == "stack":
                # STACK: (B, T, L, D_target)
                # Apply RP per layer first to ensure D matches
                stack_list = []
                for l in self.layer_names:
                    f = batch_layer_features[l]
                    f = self._apply_random_projection(f, key=l)
                    stack_list.append(f)

                # Stack along new dimension (axis -2)
                # If (B, T, D) -> (B, T, L, D). If (B, D) -> (B, L, D)
                try:
                    combined = torch.stack(stack_list, dim=-2)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Stacking failed. Layers likely have different dimensions. "
                        f"Set random_projection to 'sparse' or 'dense' to normalize dimensions. Error: {e}"
                    )

                if isinstance(combined, torch.Tensor):
                    combined = combined.cpu().numpy()
                all_features.append(combined)

            else:
                # NONE (Separate)
                for l_name in self.layer_names:
                    f = batch_layer_features[l_name]
                    # Apply RP per layer
                    f = self._apply_random_projection(f, key=l_name)

                    if isinstance(f, torch.Tensor):
                        f = f.cpu().numpy()
                    all_features[l_name].append(f)

            # Label processing
            if labels is not None:
                def _process_labels(lbl):
                    processed = []
                    if isinstance(lbl, (list, tuple)) and len(lbl) >= 2 \
                       and hasattr(lbl[0], "__len__") and not isinstance(lbl[0], str):
                        keys = list(lbl[0])
                        arrays = []
                        for arr in lbl[1:]:
                            if torch.is_tensor(arr):
                                arr = arr.cpu().numpy()
                            if isinstance(arr, np.ndarray):
                                arr = arr.tolist()
                            elif isinstance(arr, (float, int)):
                                arr = [arr] * len(keys)
                            else:
                                arr = list(arr)
                            arrays.append(arr)
                        for tup in zip(keys, *arrays):
                            processed.append(tup)
                    else:
                        if torch.is_tensor(lbl):
                            lbl = lbl.cpu().numpy()
                        if isinstance(lbl, (float, int)):
                            processed.append(lbl)
                        elif not isinstance(lbl, (list, np.ndarray)):
                            processed.append(lbl)
                        else:
                            processed.extend(lbl)
                    return processed
                batch_labels = _process_labels(labels)
                all_labels.extend(batch_labels)

            # Free GPU cache after each batch to reduce fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size if new_batch_size is None else new_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=custom_collate
            )
            for batch in tqdm(dataloader, desc="Extracting Features", unit="batch"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs, labels = batch, None
                process_batch(inputs, labels)

        except TypeError as e:
            raise ValueError(f"DataLoader failed: {e}")

        # Final Formatting
        if self.aggregation_mode in ["concatenate", "stack"]:
            if all_features:
                return np.vstack(all_features), (all_labels if all_labels else None)
            else:
                return None, None
        else:
            # Stack each layer's list
            result_dict = {}
            for l_name, feats in all_features.items():
                if feats:
                    result_dict[l_name] = np.vstack(feats)
            return result_dict, (all_labels if all_labels else None)
