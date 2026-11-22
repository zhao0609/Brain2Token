import torch
import torch.nn.functional as F


def get_visual_module(model):
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen3-VL visual module.")


def get_language_module(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError("Could not find Qwen3-VL language module.")


@torch.no_grad()
def extract_vision_block_features(model, pixel_values, image_grid_thw, target_layers=(8,)):
    """Return raw Qwen3-VL vision-block outputs before patch merger.

    The output tensors are flattened across images in the same layout as Qwen's
    visual encoder: [sum(t*h*w), hidden_size]. For NSD 256x256 images this is
    [batch*256, 1152].
    """
    visual = get_visual_module(model)
    target_layers = set(target_layers)
    max_target = max(target_layers)

    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype))
    hidden_states = hidden_states + visual.fast_pos_embed_interpolate(image_grid_thw)

    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    outputs = {}
    for layer_idx, block in enumerate(visual.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_idx in target_layers:
            outputs[layer_idx] = hidden_states
        if layer_idx >= max_target:
            break

    return outputs


@torch.no_grad()
def continue_vision_from_block(model, block_hidden_states, image_grid_thw, start_layer):
    """Continue Qwen3-VL visual encoder from a raw block output.

    Returns:
        main_embeds: [sum merged visual tokens, 4096]
        deepstack_embeds: list of [sum merged visual tokens, 4096] for deepstack
            layers whose source layer is >= start_layer.

    If start_layer is 8, the returned deepstack list contains the layer 8, 16,
    and 24 branches, which is the complete Qwen3-VL visual injection package.
    """
    visual = get_visual_module(model)
    hidden_states = block_hidden_states

    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    seq_len, _ = hidden_states.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_features = []
    if start_layer in visual.deepstack_visual_indexes:
        merger_idx = visual.deepstack_visual_indexes.index(start_layer)
        deepstack_features.append(visual.deepstack_merger_list[merger_idx](hidden_states))

    for layer_idx in range(start_layer + 1, len(visual.blocks)):
        hidden_states = visual.blocks[layer_idx](
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_idx in visual.deepstack_visual_indexes:
            merger_idx = visual.deepstack_visual_indexes.index(layer_idx)
            deepstack_features.append(visual.deepstack_merger_list[merger_idx](hidden_states))

    main_embeds = visual.merger(hidden_states)
    return main_embeds, deepstack_features
