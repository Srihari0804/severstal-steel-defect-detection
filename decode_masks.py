import numpy as np

def rle_to_mask(rle_string: str, height: int = 256, width: int = 1600) -> np.ndarray:
    """
    Decodes an RLE string into a binary mask.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    
    if not isinstance(rle_string, str) or not rle_string.strip():
        return mask.reshape((height, width), order='F')

    rle_numbers = [int(x) for x in rle_string.split()]
    starts = np.array(rle_numbers[0::2]) - 1  # Convert to 0-based indexing
    lengths = np.array(rle_numbers[1::2])

    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    # Reshape using column-major order
    return mask.reshape((height, width), order='F')
