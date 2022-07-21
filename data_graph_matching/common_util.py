import numpy as np

def is_numeric_text(s):
    if len(s) == 0:
        return False
    # a special case: caused by errors in the OCR
    if len(s.lstrip('-')) == 0:
        return True
    try:
        r = float(s)
        return True
    except:
        return False

def is_numeric_text_strict(s):
    try:
        r = float(s)
        return True
    except:
        return False


def run_length_encoding(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def compute_iou_score(res1, res2):
    intersection = np.logical_and(res1, res2)
    union = np.logical_or(res1, res2)
    if np.sum(union) > 0:
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    else:
        return 0.0