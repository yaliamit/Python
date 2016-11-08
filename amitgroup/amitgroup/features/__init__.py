
from .bedges import bedges, change_saturation, bedges_from_image, bspread
from .spread_patches import spread_patches, spread_patches_new
from .code_parts import (code_parts,
                         code_parts_many,
                         convert_partprobs_to_feature_vector,
                         convert_part_to_feature_vector,
                         code_parts_as_features)
from .hog import hog

from .binary_descriptor import BinaryDescriptor
from .edge_descriptor import EdgeDescriptor
from .parts_descriptor import PartsDescriptor


# TODO: Experimental
from .code_parts import (code_parts_new,
                         extract_parts,
                         code_parts_support_mask,
                         code_parts_INDICES,
                         code_parts_as_features_INDICES,
                         extract_parts_adaptive_EXPERIMENTAL)
