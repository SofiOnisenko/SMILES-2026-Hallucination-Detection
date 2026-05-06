from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts the representation of the last non-padding token
    from the final transformer layer
    """

    final_layer = hidden_states[-1]

    last_pos = attention_mask.nonzero(as_tuple=False)[-1].item()

    return final_layer[last_pos]


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Geometric features are intentionally disabled in the final solution
    """

    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:

    features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(
            hidden_states,
            attention_mask,
        )
        features = torch.cat([features, geo_features], dim=0)

    return features