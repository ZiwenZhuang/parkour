""" The script to clear the dataset, but maintain the metadata.json file """
import os
import os.path as osp
import shutil
import argparse

def main(args):
    for data_dir in args.data_dirs:
        if osp.isfile(osp.join(data_dir, "metadata.json")):
            shutil.copy2(
                osp.join(data_dir, "metadata.json"),
                osp.join(osp.dirname(data_dir), data_dir.split("/")[-1] + ".json")
            )
            print(f"Moved metadata.json to {osp.join(osp.dirname(data_dir), data_dir.split('/')[-1] + '.json')}")
        # removing the directory
        if osp.isdir(data_dir) and (not osp.islink(data_dir)):
            print("Removing directory: ", data_dir)
            if not args.only_files:
                shutil.rmtree(data_dir)
            else:
                for file_name in os.listdir(data_dir):
                    os.remove(osp.join(data_dir, file_name))
        print(f"Finished clearing {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs",
        type= str,
        nargs= "+",
    )
    parser.add_argument("--only_files",
        action= "store_true",
    )
    args = parser.parse_args()
    main(args)
