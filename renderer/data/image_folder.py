import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_video_dataset(dir, max_n_sequences=None):
    images = []
    if dir:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = sorted(os.walk(dir))
        for fname in sorted(fnames):
            paths = []
            root = fname[0]
            for f in sorted(fname[2]):
                if is_image_file(f):
                    paths.append(os.path.join(root, f))
            if len(paths) > 0:
                images.append(paths)
        if max_n_sequences is not None:
            images = images[:max_n_sequences]
    return images

def assert_valid_pairs(A_paths, B_paths):
    assert len(A_paths) > 0 and len(B_paths) > 0, 'No sequences found.'
    assert len(A_paths) == len(B_paths), 'Number of NMFC sequences different than RGB sequences.'
    for i in range(len(A_paths)):
        assert len(A_paths[i]) == len(B_paths[i]), 'Number of NMFC frames in sequence different than corresponding RGB frames.'
