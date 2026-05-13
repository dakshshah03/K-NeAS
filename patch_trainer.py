import sys

trainer_file = "/home/daksh/neas_ablations/NeAS-2/src/trainer.py"
with open(trainer_file, "r") as f:
    code = f.read()

# Fix dataset instantiation
old_dataset = "train_dset = dataset_class(cfg[\"exp\"][\"datadir\"], cfg[\"train\"][\"n_rays\"], \"train\", device, num_views=num_views, n_mask_rays=self.n_mask_rays)"
new_dataset = "train_dset = dataset_class(cfg[\"exp\"][\"datadir\"], cfg[\"train\"][\"n_rays\"], \"train\", device, num_views=num_views, n_mask_rays=self.n_mask_rays, floater_reg=cfg[\"train\"].get(\"floater_reg\", True))"
code = code.replace(old_dataset, new_dataset)

# Fix imports
old_imp = "shared_att_freq_mlp, shared_att_hash_mlp,"
new_imp = "shared_att_freq_mlp, shared_att_hash_mlp, IndependentAttenuationMLP, hard_material_selector,"
code = code.replace(old_imp, new_imp)

# Fix network instantiation
old_k_freq = """            else:  # K >= 2: shared latent space
                self.sdf_model = sdf_freq_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim, multires=multires).to(device)
                self.att_model = shared_att_freq_mlp(input_dim=feature_dim,
                                                     material_activations=material_configs).to(device)"""

new_k_freq = """            else:  # K >= 2
                self.sdf_model = sdf_freq_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim, multires=multires).to(device)
                if cfg["network"].get("shared_latent", True):
                    self.att_model = shared_att_freq_mlp(input_dim=feature_dim,
                                                         material_activations=material_configs).to(device)
                else:
                    self.att_model = IndependentAttenuationMLP(input_dim=feature_dim, material_activations=material_configs, encoding_type='freq').to(device)"""

code = code.replace(old_k_freq, new_k_freq)

old_k_hash = """            else:  # K >= 2: shared latent space
                self.sdf_model = sdf_hash_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim,
                                                 num_levels=num_levels, level_dim=level_dim,
                                                 base_resolution=base_resolution,
                                                 log2_hashmap_size=log2_hashmap_size).to(device)
                self.att_model = shared_att_hash_mlp(input_dim=feature_dim,
                                                     material_activations=material_configs).to(device)"""

new_k_hash = """            else:  # K >= 2
                self.sdf_model = sdf_hash_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim,
                                                 num_levels=num_levels, level_dim=level_dim,
                                                 base_resolution=base_resolution,
                                                 log2_hashmap_size=log2_hashmap_size).to(device)
                if cfg["network"].get("shared_latent", True):
                    self.att_model = shared_att_hash_mlp(input_dim=feature_dim,
                                                         material_activations=material_configs).to(device)
                else:
                    self.att_model = IndependentAttenuationMLP(input_dim=feature_dim, material_activations=material_configs, encoding_type='hash').to(device)"""

code = code.replace(old_k_hash, new_k_hash)

# Update renderer kwargs
old_render = "intensity, att_coeff = render_image("
code = code.replace("intensity, att_coeff = render_image(", "soft_selector = self.conf['network'].get('soft_selector', True)\n                intensity, att_coeff = render_image(")

old_render_call = """                    att_model=self.att_model1 if self.num_materials == 1 else self.att_model,
                    s=self.s_val,
                    n_samples=self.n_samples,
                    chunk_size=self.n_rays,
                    tau=self.tau,
                    num_materials=self.num_materials
                )"""
new_render_call = """                    att_model=self.att_model1 if self.num_materials == 1 else self.att_model,
                    s=self.s_val,
                    n_samples=self.n_samples,
                    chunk_size=self.n_rays,
                    tau=self.tau,
                    num_materials=self.num_materials,
                    soft_selector=soft_selector
                )"""
code = code.replace(old_render_call, new_render_call)

old_eval_call = """                        att_model=self.att_model1 if self.num_materials == 1 else self.att_model,
                        s=self.s_val,
                        n_samples=self.val_n_samples,
                        chunk_size=self.val_chunk_size,
                        tau=self.tau,
                        num_materials=self.num_materials
                    )"""
new_eval_call = """                        att_model=self.att_model1 if self.num_materials == 1 else self.att_model,
                        s=self.s_val,
                        n_samples=self.val_n_samples,
                        chunk_size=self.val_chunk_size,
                        tau=self.tau,
                        num_materials=self.num_materials,
                        soft_selector=soft_selector
                    )"""
code = code.replace(old_eval_call, new_eval_call)

with open(trainer_file, "w") as f:
    f.write(code)
