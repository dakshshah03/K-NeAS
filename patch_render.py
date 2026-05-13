import sys

render_file = "/home/daksh/neas_ablations/NeAS-2/src/render/render.py"
with open(render_file, "r") as f:
    code = f.read()

old_render_sig = "def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size, tau=None, num_materials=1):"
new_render_sig = "def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size, tau=None, num_materials=1, soft_selector=True):"
code = code.replace(old_render_sig, new_render_sig)

old_volume_call = "chunk_intensity, chunk_att_coeff = volume_render_intensity("
new_volume_call = "chunk_intensity, chunk_att_coeff = volume_render_intensity("

code = code.replace("tau=tau, num_materials=num_materials)", "tau=tau, num_materials=num_materials, soft_selector=soft_selector)")


old_vol_sig = "def volume_render_intensity(rays, sdf_model, att_model, s, n_samples, chunk_size=1024, tau=None, num_materials=1):"
new_vol_sig = "def volume_render_intensity(rays, sdf_model, att_model, s, n_samples, chunk_size=1024, tau=None, num_materials=1, soft_selector=True):"
code = code.replace(old_vol_sig, new_vol_sig)

old_km = """            if is_km:
                # KM-NeAS: K SDFs + shared attenuation + nested selector
                distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
                boundary_values = [surface_boundary_function(d, s) for d in distances]
                raw_attenuations = att_model(feature_vector)
                att_coeff = nested_material_selector(boundary_values, raw_attenuations)"""

new_km = """            if is_km:
                from ..network import nested_material_selector, hard_material_selector
                distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
                boundary_values = [surface_boundary_function(d, s) for d in distances]
                raw_attenuations = att_model(feature_vector)
                
                if soft_selector:
                    att_coeff = nested_material_selector(boundary_values, raw_attenuations)
                else:
                    att_coeff = hard_material_selector(distances, raw_attenuations, boundary_values)"""

code = code.replace(old_km, new_km)

with open(render_file, "w") as f:
    f.write(code)
