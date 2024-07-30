import os
import tarfile

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# base_dir = "./pix2pix_test02a/dataset"
# source_dir = f"{base_dir}/direct-component-set"
# output_filename = f"{base_dir}/archive.tar.gz"
# make_tarfile(output_filename, source_dir)